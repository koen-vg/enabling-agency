# SPDX-FileCopyrightText: 2023 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Simple utility functions for working with PyPSA networks."""

import copy
import logging
import os
import time
from collections import OrderedDict
from numbers import Real
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pypsa
from pypsa.components import component_attrs, components
from pypsa.descriptors import get_active_assets, get_extendable_i
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.descriptors import nominal_attrs
from pypsa.linopf import ilopf, network_lopf
from pypsa.linopt import (
    define_constraints,
    get_var,
    linexpr,
    write_bound,
    write_objective,
)

from solve_network import extra_functionality as sec_extra_functionality
from solve_network import prepare_network

marginal_attr = {"Generator": "p", "Link": "p", "Store": "p", "StorageUnit": "p"}


def get_basis_variables(n: pypsa.Network, basis: dict) -> OrderedDict:
    """Return the decision variables of `n` making up the basis.

    Specifically, a dictionary is returned, indexed the same as
    `basis`. Each item of the dictionary is a pandas DataFrame with
    the columns "coeffs" and "vars" containing the coefficients and
    names of the variables making up the corresponding dimension of
    the basis.

    This function can only be called inside the `extra_functionality`
    function passed to the PyPSA network solver.

    """
    proj_basis = OrderedDict()

    for key, dim in basis.items():
        summands = []
        for spec in dim:
            # First, get the respecified variables for _all_
            # components of the given type, for example p_nom for
            # _all_ generators.
            vars = get_var(n, spec["c"], spec["v"])

            # Now, we filter down to a desired subset of variables,
            # following keywords given in the specification. First,
            # narrow down to a given carrier.
            if "carrier" in spec:
                vars = vars.loc[
                    n.df(spec["c"])
                    .loc[n.df(spec["c"])["carrier"] == spec["carrier"]]
                    .index
                ]

            # If specified, narrow down to a given region. The
            # "region" will be given as a list of countries, where
            # each country as represented as a two-letter code. To
            # check the country of a component, we check the bus name;
            # each bus will start with a country code. For example, a
            # bus might be called "UK0 0 H2"; this is a hydrogen bus
            # in the UK.

            # There are two potential complications. First, a bus
            # might represent multiple aggregated countries, in which
            # case its name will look like "UK0_IR0_... H2" (for
            # example). Secondly, if the component at hand is a link
            # or line, it might be located "between" two countries,
            # and might have one side located inside the focus region
            # and one side outside of the focus region. In this case,
            # we include the component in the basis if either side is
            # located in the focus region.
            if "regions" in spec:
                region = spec["regions"]
                bus_columns = (
                    ["bus0", "bus1"] if spec["c"] in ["Link", "Line"] else ["bus"]
                )

                # Extract a list of countries associated to each variable.
                var_countries = (
                    n.df(spec["c"])
                    .loc[vars.index]
                    .apply(
                        lambda l: [
                            c.strip("0123456789")  # E.g. "UK0" -> "UK"
                            for bc in bus_columns
                            for c in l[bc].split(" ")[0].split("_")
                        ],
                        axis="columns",
                    )
                )

                # Filter down to variables that are connected to the region.
                vars_in_region = var_countries.apply(
                    lambda cs: len(set(cs).intersection(set(region))) > 0
                )
                vars = vars.loc[vars_in_region]

            # Extract coefficients.
            coeffs = pd.Series(1, index=vars.index)
            if "weight" in spec:
                # We allow spec["weight"] to be either a string or an
                # iterable of strings. Either way, the variable
                # coefficients are multiplied by the values in the
                # component dataframe columns given by the(se) string(s).
                if isinstance(spec["weight"], str):
                    factors = [spec["weight"]]
                else:
                    factors = spec["weight"]
                for f in factors:
                    coeffs *= n.df(spec["c"]).loc[vars.index, f]

            if "scale_by_years" in spec:
                # In this case we scale the coefficients down by the
                # number of years over which the network is defined.
                # This can be useful for example when the coefficients
                # are given by costs which can depend on the
                # time-frame of the network. Disregard leap years.
                coeffs /= n.snapshot_weightings.objective.sum() / 8760

            if "regions" in spec and spec["c"] in ["Link", "Line"]:
                # For links and lines where one side is in the region
                # and the other side outside the region, we divide the
                # coefficients by two. Get the countries associated to
                # each side of the line/link, and check if they are in
                # the region.
                bus0_countries = (
                    n.df(spec["c"])
                    .loc[vars.index]
                    .apply(
                        lambda l: [
                            c.strip("0123456789")  # E.g. "UK0" -> "UK"
                            for c in l["bus0"].split(" ")[0].split("_")
                        ],
                        axis="columns",
                    )
                )
                bus1_countries = (
                    n.df(spec["c"])
                    .loc[vars.index]
                    .apply(
                        lambda l: [
                            c.strip("0123456789")  # E.g. "UK0" -> "UK"
                            for c in l["bus1"].split(" ")[0].split("_")
                        ],
                        axis="columns",
                    )
                )

                # Note that some links lead to buses like "co2
                # atmosphere"; not associated to any country. We count
                # these as being in the region.
                bus0_in_region = bus0_countries.apply(
                    lambda cs: len(cs) == 0
                    or len(set(cs).intersection(set(region))) > 0
                )
                bus1_in_region = bus1_countries.apply(
                    lambda cs: len(cs) == 0
                    or len(set(cs).intersection(set(region))) > 0
                )
                # Divide the coefficients by two if one side is in the
                # region and the other side is outside the region
                # (using XOR, ^).
                coeffs.loc[bus0_in_region ^ bus1_in_region] /= 2

            # Append the coefficients and variables to the complete linear summand.
            expr = pd.concat([coeffs, vars], axis="columns")
            expr.columns = ["coeffs", "vars"]
            summands.append(expr)
        proj_basis[key] = pd.concat(summands, axis="rows")
    return proj_basis


def get_basis_values(n: pypsa.Network, basis: OrderedDict, use_opt=True) -> OrderedDict:
    """Get the coordinates of a solved PyPSA network `n` in the given basis.

    If `use_opt` is set to True, it uses the optimal capacities *_nom_opt
    instead of *_nom capacities.
    """
    basis_caps = BasisCapacities(basis=basis, init=n, use_opt=use_opt)
    return basis_caps.project_to_coordinates()


def get_basis_values_by_bus(
    n: pypsa.Network, basis: OrderedDict, labels: list, use_opt=True
) -> OrderedDict:
    """Get the coordinates of a solved PyPSA network `n` in the given basis.

    Only the coordinates given by the labels `labels` are returned;
    this allows the user to select only coordinates which can actually
    be resolved down to buses. If `use_opt` is set to True, it uses
    the optimal capacities *_nom_opt instead of *_nom capacities.

    """
    basis_caps = BasisCapacities(basis=basis, init=n, use_opt=use_opt)
    return basis_caps.project_to_coordinates_by_bus(labels)


def scale_caps(
    n: pypsa.Network, factor: Real, only_extendable: bool = False
) -> pypsa.Network:
    """Return a network with uniformly scaled capacities based on `n`.

    If the `only_extendable` argument is set, only extendable
    capacities are scaled.

    Both nominal and optimal capacities are scaled by the same factor,
    `factor`.

    This function returns a copy of the given PyPSA network where
    capacities are scaled, and does not modify the argument.

    """
    result = copy.deepcopy(n)
    for c, attr in nominal_attrs.items():
        idx = get_extendable_i(result, c) if only_extendable else result.df(c).index
        result.df(c).loc[idx, attr] *= factor
        result.df(c).loc[idx, attr + "_opt"] *= factor
    return result


def add_caps(
    n: pypsa.Network, m: pypsa.Network, only_extendable: bool = False
) -> pypsa.Network:
    """Add the capacities of one network to another.

    The capacities of `m` are added to `n`, and the resulting new
    network returned. If the `only_extendable` argument is set to
    True, only extendable capacities of `m` are added to `n`. Both
    nominal and optimal capacities (i.e. *_nom and *_nom_opt) are
    added together.

    This function returns a new network, and does not modify its
    arguments.

    """
    result = copy.deepcopy(n)
    for c, attr in nominal_attrs.items():
        idx = get_extendable_i(result, c) if only_extendable else result.df(c).index
        result.df(c).loc[idx, attr] += m.df(c).loc[idx, attr]
        result.df(c).loc[idx, attr + "_opt"] += m.df(c).loc[idx, attr + "_opt"]
    return result


def apply_caps(
    n_source: pypsa.Network, n_target: pypsa.Network, only_extendable: bool = True
) -> None:
    """Apply the capacities of one network, `n_source` to another, `n_target`.

    If the `only_extendable` argument is set to True, only extendable
    capacities are applied.

    It modifies the argument `n_target`.

    """
    for c, attr in nominal_attrs.items():
        idx = get_extendable_i(n_source, c) if only_extendable else n_source.df(c).index
        for v in [attr, attr + "_opt"]:
            n_target.df(c).loc[idx, v] = n_source.df(c).loc[idx, v]


class BasisCapacities:
    """A datastructure for storing a set of capacities of a PyPSA network."""

    def __init__(self, basis: OrderedDict, init: pypsa.Network = None, use_opt=False):
        """Initialise from a basis and optionally a PyPSA network."""
        self._basis = basis
        self._caps = {}
        if init is not None:
            # Make a copy of the initialising network, in case `init`
            # ever gets modified.
            n = copy.deepcopy(init)
            self._n = n
            for key, dim in basis.items():
                self._caps[key] = {}
                for spec in dim:
                    # Initialise the dataframe of capacities.
                    v = spec["v"] if not use_opt else spec["v"] + "_opt"

                    if spec["c"] not in self._caps[key]:
                        self._caps[key][spec["c"]] = pd.DataFrame(columns=["coeffs", v])

                    # Extract the capacities.
                    caps = n.df(spec["c"])[v]

                    # Filter by carrier and region
                    if "carrier" in spec:
                        caps = n.df(spec["c"]).loc[
                            n.df(spec["c"])["carrier"] == spec["carrier"]
                        ][v]

                    # Deal with regions similarly as in the
                    # basis_variables function.
                    if "regions" in spec:
                        region = spec["regions"]
                        bus_columns = (
                            ["bus0", "bus1"]
                            if spec["c"] in ["Link", "Line"]
                            else ["bus"]
                        )

                        # Extract a list of countries associated to each variable.
                        cap_countries = (
                            n.df(spec["c"])
                            .loc[caps.index]
                            .apply(
                                lambda l: [
                                    c.strip("0123456789")  # E.g. "UK0" -> "UK"
                                    for bc in bus_columns
                                    for c in l[bc].split(" ")[0].split("_")
                                ],
                                axis="columns",
                            )
                        )

                        # Filter down to variables that are connected to the region.
                        caps_in_region = cap_countries.apply(
                            lambda cs: len(set(cs).intersection(set(region))) > 0
                        )
                        caps = caps.loc[caps_in_region]

                    # Extract the coefficients.
                    coeffs = pd.Series(1, index=caps.index, name="coeffs")
                    if "weight" in spec:
                        if isinstance(spec["weight"], str):
                            factors = [spec["weight"]]
                        else:
                            factors = spec["weight"]
                        for f in factors:
                            coeffs *= n.df(spec["c"]).loc[caps.index, f]

                    if "scale_by_years" in spec:
                        coeffs /= n.snapshot_weightings.objective.sum() / 8760

                    if "regions" in spec and spec["c"] in ["Link", "Line"]:
                        # Similarly to the basis_variables function,
                        # we only count links and lines between the
                        # focus region and the outside by half.
                        bus0_countries = (
                            n.df(spec["c"])
                            .loc[caps.index]
                            .apply(
                                lambda l: [
                                    c.strip("0123456789")  # E.g. "UK0" -> "UK"
                                    for c in l["bus0"].split(" ")[0].split("_")
                                ],
                                axis="columns",
                            )
                        )
                        bus1_countries = (
                            n.df(spec["c"])
                            .loc[caps.index]
                            .apply(
                                lambda l: [
                                    c.strip("0123456789")  # E.g. "UK0" -> "UK"
                                    for c in l["bus1"].split(" ")[0].split("_")
                                ],
                                axis="columns",
                            )
                        )

                        # Note that some links lead to buses like "co2
                        # atmosphere"; not associated to any country. We count
                        # these as being in the region.
                        bus0_in_region = bus0_countries.apply(
                            lambda cs: len(cs) == 0
                            or len(set(cs).intersection(set(region))) > 0
                        )
                        bus1_in_region = bus1_countries.apply(
                            lambda cs: len(cs) == 0
                            or len(set(cs).intersection(set(region))) > 0
                        )
                        # Divide the coefficients by two if one side is in the
                        # region and the other side is outside the region
                        # (using XOR, ^).
                        coeffs.loc[bus0_in_region ^ bus1_in_region] /= 2

                    # Store everything in the capacity datastructure.
                    df = pd.concat([coeffs, caps], axis="columns")
                    self._caps[key][spec["c"]] = pd.concat(
                        [self._caps[key][spec["c"]], df], axis="index"
                    )

    def project_to_coordinates(self) -> OrderedDict:
        """Project the capacities down to a space defined by the basis specification."""
        proj_values = OrderedDict()
        for key, caps in self._caps.items():
            value = 0
            for comp in caps.values():
                value += (comp.iloc[:, 0] * comp.iloc[:, 1]).sum()
            proj_values[key] = value

        return proj_values

    def project_to_coordinates_by_bus(self, labels) -> OrderedDict:
        """Project the capacities down to a space defined by the basis specification."""
        proj_values = pd.DataFrame(index=self._n.buses.index, columns=labels)
        proj_values.loc[:, :] = 0
        for key in labels:
            for comp in self._caps[key].values():
                values = pd.DataFrame(comp.iloc[:, 0] * comp.iloc[:, 1], columns=["x"])
                # This is a hacky and bad way to extract the bus for
                # each row, and only works for PyPSA-Eur bus naming
                # conventions, and probably if there are only less
                # than 10 buses per country.
                values["bus"] = [i[:5] for i in values.index]
                values = values.groupby("bus").sum()
                proj_values.loc[values.index, key] += values.x

        return proj_values

    def apply_to_network(self, n: pypsa.Network, set_opt: bool = True) -> None:
        """Apply the stored capacities to the given network `n`.

        If `set_opt` is activated, then set the *_nom_opt values in
        `n` according to the stored capacities.

        """
        # Go through every "dimension" in the projected space...
        for dim in self._caps.values():
            # Each dimension can contain capacities from different technologies.
            for c, caps in dim.items():
                # Here, `c` is a component such as 'Generator' or
                # 'Line', and `caps` is a dataframe, indexed over
                # component names, with columns 'coeffs' and '*_nom'
                # (e.g. 'p_nom', 's_nom', etc.). We want to apply the
                # capacities (stored in the second column of `caps`)
                # to the corresponding dataframe in `n`, including to
                # '*_nom_opt' depending on the `set_opt` argument.
                v = caps.columns[1]
                n.df(c).loc[caps.index, v] = caps[v]
                if set_opt:
                    n.df(c).loc[caps.index, v + "_opt"] = caps[v]

    def __add__(self, other: "BasisCapacities") -> "BasisCapacities":
        """Add two sets of capacities."""
        if self._basis != other._basis:
            raise ValueError(
                "Cannot add two BasisCapacities objects with different bases."
            )
        result = BasisCapacities(self._basis)
        for key in self._basis:
            result._caps[key] = {}
            for c in self._caps[key]:
                caps_a = self._caps[key][c]
                caps_b = other._caps[key][c]
                # Check that the coefficients are the same; otherwise
                # the addition is not defined.
                if not np.array_equal(caps_a.coeffs, caps_b.coeffs):
                    raise ValueError(
                        "Cannot add two BasisCapacities objects with differing"
                        " coefficients."
                    )
                caps = caps_a.iloc[:, 1] + caps_b.iloc[:, 1]
                result._caps[key][c] = pd.concat([caps_a.coeffs, caps], axis="columns")
        return result

    def __mul__(self, factor: Real) -> "BasisCapacities":
        """Scale the capacities by a given factor."""
        if isinstance(factor, (int, float)):
            result = BasisCapacities(self._basis)
            result._caps = copy.deepcopy(self._caps)
            for dim in result._caps.values():
                for caps in dim.values():
                    caps.iloc[:, 1] *= factor
            return result
        else:
            return NotImplemented

    def __rmul__(self, factor: Real) -> "BasisCapacities":
        """Scale the capacities by a given factor."""
        if isinstance(factor, (int, float)):
            return self * factor
        else:
            return NotImplemented

    def __truediv__(self, factor: Real) -> "BasisCapacities":
        """Divide the capacities by a given factor."""
        return self * (1 / factor)

    def scale(self, factors: dict) -> "BasisCapacities":
        """Scale capacities dimension-by-dimension.

        The `factors` dictionary gives the factors by which each
        dimension of `self` (according to the basis over which `self`
        is defined) is scaled.

        For example, if the BasisCapacities have two dimensions,
        'wind' and 'solar', and `factors` is
        `{'wind': 1.2, 'solar': 0.9}`, then all capacities in the
        'wind' dimension are scaled up by 20%, while all capacities in
        the 'solar' dimension are scaled down by 10%.

        This function returns a copy of `self` with scaled capacities,
        and does not modify `self`.

        """
        result = BasisCapacities(self._basis)
        result._caps = copy.deepcopy(self._caps)
        for key, dim in result._caps.items():
            for caps in dim.values():
                caps.iloc[:, 1] *= factors[key]
        return result

    def shift(self, distances: dict) -> "BasisCapacities":
        """Shift total capacities by given distances in each dimension.

        The `distances` dictionary gives the distances over which each
        dimension of `self` (according to the basis over which `self`
        is defined) is changed in total. The change is distributed in
        a relative way among all capacities in each dimension.

        For example, suppose the BasisCapacities (`self`) object has
        two dimensions, 'wind' and 'solar', with total capacities of
        4GW and 10GW respectively, and we have

            `distances == {'wind': 2.0e3, 'solar': -1.0e3}`.

        (Here, we assume we're working in MWs, so the distances are
        2GW and -1GW.) Then `shift` will increase all capacities in
        the 'wind' dimension by 50% and decrease all capacities in the
        'solar' dimension by 10% in order to achieve total capacities
        of 6GW and 9GW respectively.

        This function returns a copy of `self` with scaled capacities,
        and does not modify `self`.

        """
        p = self.project_to_coordinates()
        factors = {}
        for d in self._basis:
            factors[d] = max(1 + distances[d] / p[d], 0)
        return self.scale(factors)


def set_nom_to_opt(n: pypsa.Network) -> None:
    """Overwrite nominal extendable capacities of `n` by optimal capacities.

    Modifies the argument `n`.

    """
    for c, attr in nominal_attrs.items():
        i = get_extendable_i(n, c)
        # Note: we make sure not to set any nominal capacities to NaN
        # by using `dropna()`.
        n.df(c).loc[i, attr] = n.df(c).loc[i, attr + "_opt"].dropna()


def solve_network_in_direction(
    n: pypsa.Network,
    direction: Sequence[float],
    basis: OrderedDict = None,
    max_obj: float = np.inf,
) -> None:
    """Solve the network `n` with custom objective function.

    Specifically, solve `n` so as to find the extreme point in
    direction `direction` of the reduced near-optimal feasible space
    (given by `basis` and `max_obj`).

    Parameters
    ----------
    n : pypsa.Network
        Network to be solved.
    direction : Sequence[float]
        Direction used for solving `n` (in the low-dimensional projection).
    basis : OrderedDict
        Basis defining the direction in the higher-dimensional space.
    max_obj : float
        Constraint defining the near-optimal space.

    """
    # Retrieve solver options from n.
    solving_options = n.config["solving"]["options"]
    solver_options = n.config["solving"]["solver"].copy()
    solver_name = solver_options.pop("name")
    tmpdir = n.config["solving"].get("tmpdir", None)
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    # TODO: make the prepare_network and extra_functionality work for
    # pypsa-eur as well as pypsa-eur-sec.
    n = prepare_network(n, solving_options, config=n.config)

    def extra_functionality(n, snapshots):
        """Enforce near-optimality and define a custom objective."""
        # Add constraint to make solution near-optimal.
        define_constraints(
            n, get_objective(n, n.snapshots), "<=", max_obj, "Near_optimal"
        )

        # Now, modify the objection function to point in the given
        # direction.
        basis_variables = get_basis_variables(n, basis)
        obj = pd.concat(
            [
                linexpr((c * b.coeffs, b.vars))
                for (c, b) in zip(direction, basis_variables.values())
            ]
        )
        write_objective(n, obj)

        # Set a constant objective, which is useless in this case but expected by `ilopf`.
        n.objective_constant = 0

        # Run the additional extra functionality from the
        # sector-coupled model.
        sec_extra_functionality(n, n.snapshots)

    # Solve the network.
    if solving_options.get("skip_iterations", False):
        status, termination_condition = network_lopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            skip_objective=True,
            solver_dir=tmpdir,
            extra_functionality=extra_functionality,
            keep_references=True,
        )
    else:
        ilopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            skip_objective=True,
            solver_dir=tmpdir,
            extra_functionality=extra_functionality,
            track_iterations=solving_options.get("track_iterations", False),
            min_iterations=solving_options.get("min_iterations", 1),
            max_iterations=solving_options.get("max_iterations", 6),
        )
        # `ilopf` doesn't give us any optimisation status or
        # termination condition, and simply crashes if any
        # optimisation fails.
        status, termination_condition = "ok", "optimal"

    # Return the status and termination condition of the network
    # solve, allowing the caller to deal with suboptimal and failed
    # solves.
    return status, termination_condition


# This function is adapted from pypsa.linopf
def get_objective(n, sns):
    """Return the objective function as a linear expression."""
    if n._multi_invest:
        period_weighting = n.investment_period_weightings.objective[
            sns.unique("period")
        ]

    if n._multi_invest:
        weighting = n.snapshot_weightings.objective.mul(period_weighting, level=0).loc[
            sns
        ]
    else:
        weighting = n.snapshot_weightings.objective.loc[sns]

    total = ""

    # constant for already done investment
    nom_attr = nominal_attrs.items()
    constant = 0
    for c, attr in nom_attr:
        ext_i = get_extendable_i(n, c)
        cost = n.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        if n._multi_invest:
            active = pd.concat(
                {
                    period: get_active_assets(n, c, period)[ext_i]
                    for period in sns.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        constant += cost @ n.df(c)[attr][ext_i]

    object_const = write_bound(n, constant, constant)
    total += linexpr((-1, object_const), as_pandas=False)[0]

    # marginal cost
    for c, attr in marginal_attr.items():
        cost = (
            get_as_dense(n, c, "marginal_cost", sns)
            .loc[:, lambda ds: (ds != 0).all()]
            .mul(weighting, axis=0)
        )
        if cost.empty:
            continue
        terms = linexpr((cost, get_var(n, c, attr).loc[sns, cost.columns])).sum().sum()
        total += terms

    # investment
    for c, attr in nominal_attrs.items():
        ext_i = get_extendable_i(n, c)
        cost = n.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        if n._multi_invest:
            active = pd.concat(
                {
                    period: get_active_assets(n, c, period)[ext_i]
                    for period in sns.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        caps = get_var(n, c, attr).loc[ext_i]
        terms = linexpr((cost, caps)).sum()
        total += terms

    return total


# This function is adapted from pypsa.linopf
def get_total_cost(n):
    """Return the total cost of the network."""
    sns = n.snapshots

    if n._multi_invest:
        period_weighting = n.investment_period_weightings.objective[
            sns.unique("period")
        ]

    if n._multi_invest:
        weighting = n.snapshot_weightings.objective.mul(period_weighting, level=0).loc[
            sns
        ]
    else:
        weighting = n.snapshot_weightings.objective.loc[sns]

    total = 0

    # constant for already done investment
    nom_attr = nominal_attrs.items()
    for c, attr in nom_attr:
        ext_i = get_extendable_i(n, c)
        cost = n.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        if n._multi_invest:
            active = pd.concat(
                {
                    period: get_active_assets(n, c, period)[ext_i]
                    for period in sns.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        total += cost @ n.df(c)[attr][ext_i]

    # marginal cost
    for c, attr in marginal_attr.items():
        cost = (
            get_as_dense(n, c, "marginal_cost", sns)
            .loc[:, lambda ds: (ds != 0).all()]
            .mul(weighting, axis=0)
        )
        if cost.empty:
            continue
        terms = linexpr((cost, get_var(n, c, attr).loc[sns, cost.columns])).sum().sum()
        total += terms

    # investment
    for c, attr in nominal_attrs.items():
        ext_i = get_extendable_i(n, c)
        cost = n.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        if n._multi_invest:
            active = pd.concat(
                {
                    period: get_active_assets(n, c, period)[ext_i]
                    for period in sns.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        caps = get_var(n, c, attr).loc[ext_i]
        terms = linexpr((cost, caps)).sum()
        total += terms

    return total


def compute_feasibility_criteria(
    n: pypsa.Network,
    name: str,
) -> pd.DataFrame:
    """Compute feasibility in terms of load curtailment.

    Compute the total curtailment, the percent of load curtailed,
    accounting for numerical instabilities (curtailment starts with
    load shedding above 1 MW).

    Parameters
    ----------
    n: pypsa.Network
        Network to be validated.
    name: str
        Name of network.

    Returns
    -------
    feasibility: pd.DataFrame
        Dataframe storing the values obtained from the feasibility criteria.

    """
    # Read out the total load curtailed.
    load_shedding = n.generators_t["p"].filter(like="load").sum(axis=1)
    # Read out the total load in the network.
    total_load = n.loads_t["p"].sum().sum()
    # Filter out load curtailment above 1 MW in a node to avoid taking
    # numerical instability into account.
    filtered_shedding = load_shedding[load_shedding > 1]
    total_curtailment = filtered_shedding.sum()
    relative_curtailment = total_curtailment / total_load

    columns = [
        "Total curtailment",
        "Relative curtailment",
    ]
    values = [
        total_curtailment,
        relative_curtailment,
    ]
    feasibility = pd.DataFrame(columns=columns)
    feasibility.loc[name] = values
    return feasibility


def total_emissions(n: pypsa.Network) -> float:
    """Compute the total emissions in the given PyPSA network."""
    total = 0
    for c in n.carriers.index:
        gen_i = n.generators.carrier == c
        if n.generators.loc[gen_i].empty:
            continue
        prod = n.generators_t["p"].loc[:, gen_i].sum(axis="columns")
        total_prod = (prod * n.snapshot_weightings.objective).sum()
        emissions_factor = n.carriers.at[c, "co2_emissions"]
        efficiency = n.generators.loc[gen_i, "efficiency"][0]
        total += total_prod * emissions_factor / efficiency
    return total


def annual_investment_cost(
    n: pypsa.Network, only_extendable: bool = False, use_opt: bool = False
) -> float:
    """Compute the annual investment costs in a PyPSA network.

    This function assumes that capital costs in `n` are proportional to
    the length of time that `n` is defined over. Hence, the total
    investment costs in `n` are scaled down by the number of years `n`
    is defined over in order to get an annual figure.

    If `only_extendable` is set, only include extendable technologies in the
    calculation.

    If `use_opt` is set, use the *_nom_opt capacities instead of the
    *_nom capacities.

    """
    total = 0
    for c, attr in nominal_attrs.items():
        i = get_extendable_i(n, c) if only_extendable else n.df(c).index
        v = attr + "_opt" if use_opt else attr
        total += (n.df(c).loc[i, "capital_cost"] * n.df(c).loc[i, v]).sum()
    # Divide by the number of years the network is defined over.
    # Disregard leap years.
    total /= n.snapshot_weightings.objective.sum() / 8760
    return total


def annual_variable_cost(
    n: pypsa.Network,
) -> float:
    """Compute the annual variable costs in a PyPSA network `n`."""
    weighting = n.snapshot_weightings.objective
    total = 0
    # Add variable costs for generators
    total += (
        n.generators_t.p[n.generators.index].multiply(weighting, axis=0).sum(axis=0)
        * n.generators.marginal_cost
    ).sum()
    # Add variable costs for links (lines have none), in our model all 0 though.
    total += (
        n.links_t.p0[n.links.index].abs().multiply(weighting, axis=0).sum(axis=0)
        * n.links.marginal_cost
    ).sum()
    # Add variable costs for stores
    total += (
        n.stores_t.p[n.stores.index].abs().multiply(weighting, axis=0).sum(axis=0)
        * n.stores.marginal_cost
    ).sum()
    # Add variable costs for storage units
    total += (
        n.storage_units_t.p[n.storage_units.index]
        .abs()
        .multiply(weighting, axis=0)
        .sum(axis=0)
        * n.storage_units.marginal_cost
    ).sum()
    # Divide by the number of years the network is defined over. Disregard
    # leap years.
    total /= n.snapshot_weightings.objective.sum() / 8760
    return total


def ratio_share_national_imports(
    n: pypsa.Network,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes the share of national production compared to imports and their ratio.

    This function computes the ratio of national energy produced to
    imported energy, as well as the share of national production. It
    relies on the assumption that efficiency (conversion) losses do
    not differ diametrally between the energy coming from local
    production and the energy imported.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network.

    Returns
    -------
    pd.DataFrame
        The ratio of national production to imports.
    pd.DataFrame
        The share of national production of energy available.
    """
    # Compute generation from renewable sources (excluding hydro and biomass).
    var_renew_idx = n.generators.loc[
        n.generators.carrier.isin(
            [
                "onwind",
                "offwind-ac",
                "offwind-dc",
                "solar",
                "ror",
                "residential rural solar thermal",
                "services rural solar thermal",
                "residential urban decentral solar thermal",
                "services urban decentral solar thermal",
                "urban central solar thermal",
                "solar rooftop",
            ]
        )
    ].index
    var_renew = (
        (n.snapshot_weightings.generators @ n.generators_t.p.loc[:, var_renew_idx])
        .groupby(n.generators.bus.map(n.buses.location).map(n.buses.country))
        .sum()
    )
    # Compute generation from hydro reservoirs.
    hydro_idx = n.storage_units.loc[n.storage_units.carrier == "hydro"].index
    var_hydro = (
        (n.snapshot_weightings.stores @ n.storage_units_t.p)
        .loc[hydro_idx]
        .groupby(n.storage_units.bus.map(n.buses.location).map(n.buses.country))
        .sum()
    )
    # Compute generation from biomass. NB: Available biomass is the storage level in the first snapshot.
    biomass_idx = n.stores.loc[
        n.stores.carrier.isin(
            [
                "biogas",
                "solid biomass",
            ]
        )
    ].index
    biomass_first_e = n.stores_t.e.loc[n.snapshots[0], biomass_idx]
    biomass_last_e = n.stores_t.e.loc[n.snapshots[-1], biomass_idx]
    biomass = (
        -(biomass_last_e - biomass_first_e)
        .groupby(n.stores.bus.map(n.buses.location).map(n.buses.country))
        .sum()
    )

    # Compute nuclear generation.
    nuclear_idx = n.links.loc[n.links.carrier == "nuclear"].index
    nuclear = (
        (
            (
                pd.DataFrame(
                    np.outer(
                        n.snapshot_weightings.generators.values,
                        n.links.loc[nuclear_idx, "efficiency"].values,
                    ),
                    index=n.snapshots,
                    columns=nuclear_idx,
                )
                * n.links_t.p0[nuclear_idx]
            ).sum(axis="rows")
        )
        .groupby(n.links.bus1.map(n.buses.location).map(n.buses.country))
        .sum()
    )

    # Ambient heat for heat pumps.
    heat_pump_idx = n.links.filter(like="heat pump", axis="rows").index
    from_ambient = n.links_t["efficiency"].loc[:, heat_pump_idx] - 1
    ambient_heat_coeffs = from_ambient.mul(
        n.snapshot_weightings.generators, axis="rows"
    )
    ambient_heat = (
        (ambient_heat_coeffs * n.links_t.p0[heat_pump_idx])
        .groupby(
            n.links.bus1.map(n.buses.location).map(n.buses.country), axis="columns"
        )
        .sum()
        .sum(axis="rows")
    )

    # National energy production
    local_energy_prod = (
        pd.concat(
            [var_renew, var_hydro, biomass, nuclear, ambient_heat], axis="columns"
        )
        .fillna(0)
        .sum(axis="columns")
    )

    # Imports
    # Electricity imports
    lines_in = {
        c: n.lines.loc[
            (n.lines.bus1.map(n.buses.country) == c)
            & (n.lines.bus0.map(n.buses.country) != c)
        ].index
        for c in n.buses.country
    }
    lines_out = {
        c: n.lines.loc[
            (n.lines.bus0.map(n.buses.country) == c)
            & (n.lines.bus1.map(n.buses.country) != c)
        ].index
        for c in n.buses.country
    }
    elec_imports = {
        c: (n.snapshot_weightings.generators @ n.lines_t.p0)[lines_in[c]].sum()
        if len(lines_in[c]) > 0
        else 0
        for c in local_energy_prod.index
    }
    elec_exports = {
        c: -(n.snapshot_weightings.generators @ n.lines_t.p0)[lines_out[c]].sum()
        if len(lines_out[c]) > 0
        else 0
        for c in local_energy_prod.index
    }
    elec_net_imports = {
        c: elec_imports[c] + elec_exports[c] for c in local_energy_prod.index
    }
    elec_net_imports = pd.Series(elec_net_imports)

    pipeline_carriers = [
        "H2 pipeline",
        "H2 pipeline retrofitted",
        "gas pipeline",
        "gas pipeline new",
        "solid biomass transport",  # Not a pipeline but functionally equivalent
        "DC",  # Ditto
    ]
    into_country = lambda c: (
        (n.links.bus1.map(n.buses.location).map(n.buses.country) == c)
        & (n.links.bus0.map(n.buses.location).map(n.buses.country) != c)
    )
    out_of_country = lambda c: (
        (n.links.bus0.map(n.buses.location).map(n.buses.country) == c)
        & (n.links.bus1.map(n.buses.location).map(n.buses.country) != c)
    )
    links_in = {
        c: n.links.loc[
            into_country(c) & (n.links.carrier.isin(pipeline_carriers))
        ].index
        for c in n.buses.country
    }
    links_out = {
        c: n.links.loc[
            out_of_country(c) & (n.links.carrier.isin(pipeline_carriers))
        ].index
        for c in n.buses.country
    }

    pipeline_imports = {
        c: (n.snapshot_weightings.generators @ n.links_t.p0)[links_in[c]].sum()
        if len(links_in[c]) > 0
        else 0
        for c in local_energy_prod.index
    }
    pipeline_exports = {
        c: -(n.snapshot_weightings.generators @ n.links_t.p0)[links_out[c]].sum()
        if len(links_out[c]) > 0
        else 0
        for c in local_energy_prod.index
    }

    pipeline_net_imports = {
        c: pipeline_imports[c] + pipeline_exports[c] for c in local_energy_prod.index
    }
    pipeline_net_imports = pd.Series(pipeline_net_imports)

    # Gas imports (LNG, pipeline, production)

    gas_import = (
        (
            n.snapshot_weightings.generators
            @ n.generators_t.p.loc[:, n.generators.carrier == "gas"]
        )
        .groupby(n.generators.bus.map(n.buses.location).map(n.buses.country))
        .sum()
    )
    gas_import = gas_import.reindex(local_energy_prod.index).fillna(0)

    # Total net imports
    net_imports = elec_net_imports + pipeline_net_imports + gas_import

    # Compute ratio of national production to imports. Note this could be infinity if no energy is imported.
    ratio = (
        local_energy_prod
        - elec_net_imports.clip(upper=0)
        - pipeline_net_imports.clip(upper=0)
        - gas_import.clip(upper=0)
    ) / (elec_net_imports.clip(0) + pipeline_net_imports.clip(0) + gas_import.clip(0))

    # Compute share of national production of energy available. Note this can be larger than one for net exporters.
    local_share = local_energy_prod / (local_energy_prod + net_imports)

    return ratio, local_share


# This function is taken from pypsa-eur-sec/scripts/helper.py:
def override_component_attrs(directory):
    """Tell PyPSA that links can have multiple outputs by
    overriding the component_attrs. This can be done for
    as many buses as you need with format busi for i = 2,3,4,5,....
    See https://pypsa.org/doc/components.html#link-with-multiple-outputs-or-inputs

    Parameters
    ----------
    directory : string
        Folder where component attributes to override are stored
        analogous to ``pypsa/component_attrs``, e.g. `links.csv`.

    Returns
    -------
    Dictionary of overriden component attributes.
    """

    attrs = dict({k: v.copy() for k, v in component_attrs.items()})

    for component, list_name in components.list_name.items():
        fn = f"{directory}/{list_name}.csv"
        if os.path.isfile(fn):
            overrides = pd.read_csv(fn, index_col=0, na_values="n/a")
            attrs[component] = overrides.combine_first(attrs[component])

    return attrs
