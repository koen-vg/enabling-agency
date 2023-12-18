# SPDX-FileCopyrightText: 2023 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compute a robust solution based on the most expensive year considered.

We add investment costs uniformly on all technologies (on the optimal
solution of the most expensive year) to match investment costs in the
exact solution. This serves as a baseline for comparing the allocation
heuristics.

"""
import copy
import logging
import os
import warnings
from collections import OrderedDict
from itertools import islice, product
from multiprocessing import Pool, get_context
from pathlib import Path

# Suppress future warnings from pandas, as this should be fixed in
# most recent PyPSA versions.
warnings.simplefilter("ignore", FutureWarning)

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging
from geometry import filter_vectors_auto, uniform_random_hypersphere_sampler
from pypsa.linopf import ilopf, network_lopf
from pypsa.linopt import define_constraints, linexpr
from solve_network import extra_functionality as sec_extra_functionality
from solve_network import prepare_network
from utilities import get_basis_variables, override_component_attrs
from workflow_utilities import parse_net_spec


def get_intersection_vertices(
    intersection: pd.DataFrame, num_vertices: int, dim_groups: list[list[str]]
) -> list[pd.Series]:
    """Return a number of spread-out vertices of the given intersection.

    The vertices will include first those which min- and maximise the
    sums of the respective groups of coordinates given in
    `dim_groups`, and then a random sample of spread-out vertices.
    """
    # Check that num_vertices is high enough
    min_num_robust = 2 ** len(dim_groups) + 2 * len(dim_groups)
    if num_vertices < min_num_robust:
        raise ValueError(
            f"Number of robust networks specified ({num_vertices}) is"
            f" lower than the minimum of {min_num_robust}."
        )

    directions = []

    # First add directions with all combinations of -1, 1 in the
    # dimension groups given in `dim_groups`.
    dim_groups_min_max = list(product([-1, 1], repeat=len(dim_groups)))
    for dim_group_coords in dim_groups_min_max:
        v = pd.Series(0, index=intersection.columns, dtype=float)
        for c, g in zip(dim_group_coords, dim_groups):
            v.loc[g] = c
        directions.append(v)

    # Then include directions where each dimension group is min/maxed
    # individually.
    for g in dim_groups:
        for sign in [-1, 1]:
            v = pd.Series(0, index=intersection.columns, dtype=float)
            v.loc[g] = sign
            directions.append(v)

    # Finally generate the rest of the directions randomly
    directions.extend(
        map(
            lambda v: pd.Series(v, index=intersection.columns),
            islice(
                filter_vectors_auto(
                    uniform_random_hypersphere_sampler(len(intersection.columns)),
                    init_angle=90,
                ),
                num_vertices - len(directions),
            ),
        )
    )

    # Now, for each direction, find the vertex of the intersection
    # furthest in that direction.
    vertices = []
    for d in directions:
        i = np.matmul(intersection.values, d.values).argmax()
        vertices.append(intersection.iloc[i])

    return vertices


def compute_robust(
    n: pypsa.Network,
    point: pd.Series,
    centre: pd.Series,
    basis: OrderedDict,
    out_dir: str,
    num_iter: int,
) -> pypsa.Network:
    m = copy.deepcopy(n)
    m.opts = n.opts
    m.config = n.config
    solving_options = n.config["solving"]["options"]
    solver_options = n.config["solving"]["solver"].copy()
    solver_name = solver_options.pop("name")
    tmpdir = n.config["solving"].get("tmpdir", None)
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    m = prepare_network(m, solving_options, config=n.config)

    # Take a point that's a little towards to the centre of the intersection
    buffer = 0.01
    point = (1 - buffer) * point + buffer * centre

    def set_coordinates(n, snapshots):
        """Extra functionality to set total capacities on pypsa-eur network."""
        basis_variables = get_basis_variables(n, basis)
        # Add constraint to make the solution near-optimal.
        for key in basis_variables:
            x = basis_variables[key]
            b = point[key]
            scaling_factor = float(100 / b)
            # Define lower and upper constraints in the given basis
            # dimension, fixing the coordinates.
            name = f"fixed_investment_{key}"
            n.add(
                "GlobalConstraint",
                name,
                sense="==",
                type="investment",
                carrier_attribute=key,
            )
            define_constraints(
                n,
                linexpr((scaling_factor * x.coeffs, x.vars)).sum(),
                "==",
                scaling_factor * b,
                name="GlobalConstraint",
                attr="mu",
                axes=pd.Index([name]),
                spec=name,
            )

        # Run the additional extra functionality from the
        # sector-coupled model.
        sec_extra_functionality(n, n.snapshots)

    if solving_options.get("skip_iterations", False):
        status, _ = network_lopf(
            m,
            solver_name=solver_name,
            solver_options=solver_options,
            solver_dir=tmpdir,
            extra_functionality=set_coordinates,
            keep_references=True,
            keep_shadowprices=True,
        )
    else:
        ilopf(
            m,
            solver_name=solver_name,
            solver_options=solver_options,
            solver_dir=tmpdir,
            extra_functionality=set_coordinates,
            track_iterations=solving_options.get("track_iterations", False),
            min_iterations=solving_options.get("min_iterations", 1),
            max_iterations=solving_options.get("max_iterations", 6),
            keep_references=True,
            keep_shadowprices=True,
        )
        # `ilopf` doesn't give us any optimisation status or
        # termination condition, and simply crashes if any
        # optimisation fails.
        status = "ok"
    if status == "ok":
        p_str = "_".join(map(lambda t: f"{t[0]}:{t[1]:.3f}", point.items()))
        print(f"Optimisation {num_iter} successful, exporting to {p_str}.nc")
        # Note that `num_iter` depends on when it was an input, and
        # does not necessarily reflect the order in which the outputs
        # are generated.
        m.export_to_netcdf(os.path.join(out_dir, p_str + ".nc"))


if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # Load the network
    overrides = override_component_attrs(snakemake.params.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    n.config = snakemake.config["pypsa-eur-sec"]
    n.opts = parse_net_spec(snakemake.wildcards.spec)["sector_opts"].split("-")

    # Load the intersection
    intersection = pd.read_csv(snakemake.input.intersection, index_col=0)
    centre = pd.read_csv(snakemake.input.centre, index_col=0).squeeze()

    # Load the basis
    basis = snakemake.config["projection"]

    # Calculate at which points to compute robust networks
    vertices = get_intersection_vertices(
        intersection, snakemake.params.num_robust, snakemake.params.dim_groups
    )

    # Make sure the output directory exists
    if not os.path.exists(snakemake.params.networks_dir):
        os.makedirs(snakemake.params.networks_dir)

    logging.info(f"Writing robust networks to {snakemake.params.networks_dir}")

    # Compute robust networks in parallel
    with get_context("spawn").Pool(snakemake.params.num_parallel) as p:
        p.starmap(
            compute_robust,
            [
                (n, v, centre, basis, snakemake.params.networks_dir, num_iter)
                for num_iter, v in enumerate(vertices)
            ],
        )

    # Touch the output flag file to indicate that we are done
    Path(snakemake.output).touch()
