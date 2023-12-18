# SPDX-FileCopyrightText: 2023 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Optimising a PyPSA network with respect to its total system costs."""

import logging
import time
import warnings
from pathlib import Path

from _helpers import configure_logging
from pypsa.linopf import ilopf, network_lopf
from solve_network import extra_functionality, prepare_network
from utilities import get_basis_values, override_component_attrs
from workflow_utilities import parse_net_spec

# Ignore futurewarnings raised by pandas from inside pypsa, at least
# until the warning is fixed. This needs to be done _before_ pypsa and
# pandas are imported; ignore the warning this generates.
warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd  # noqa: E402
import pypsa  # noqa: E402

if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # Load the network and solving options.
    overrides = override_component_attrs(snakemake.params.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
    solving_options = snakemake.config["pypsa-eur"]["solving"]["options"]
    solver_options = snakemake.config["pypsa-eur"]["solving"]["solver"].copy()
    solver_name = solver_options.pop("name")
    tmpdir = snakemake.config["pypsa-eur"]["solving"].get("tmpdir", None)
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    # Add to network for extra_functionality.
    n.config = snakemake.config["pypsa-eur-sec"]
    n.opts = parse_net_spec(snakemake.wildcards.spec)["sector_opts"].split("-")

    # TODO: make the following few lines work for pypsa-eur and not just pypsa-eur-sec
    n = prepare_network(n, solving_options, config=snakemake.config["pypsa-eur-sec"])

    # Solve the network for the cost optimum and then get its
    # coordinates in the basis.
    logging.info("Compute initial, optimal solution.")
    t = time.time()
    if solving_options.get("skip_iterations", False):
        status, _ = network_lopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            solver_dir=tmpdir,
            extra_functionality=extra_functionality,
            keep_shadowprices=solving_options.get("keep_shadowprices", True),
        )
    else:
        ilopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            solver_dir=tmpdir,
            track_iterations=solving_options.get("track_iterations", False),
            min_iterations=solving_options.get("min_iterations", 1),
            max_iterations=solving_options.get("max_iterations", 6),
            extra_functionality=extra_functionality,
            keep_shadowprices=solving_options.get("keep_shadowprices", True),
        )
        # `ilopf` doesn't give us any optimisation status or
        # termination condition, and simply crashes if any
        # optimisation fails.
        status = "ok"
    logging.info(f"Optimisation took {time.time() - t:.2f} seconds.")

    # Check if the optimisation succeeded; if not we don't output
    # anything in order to make snakemake fail. Not checking for this
    # would result in an invalid (non-optimal) network being output.
    if status == "ok":
        # Write the result to the given output files. Save the objective
        # value for further processing.
        n.export_to_netcdf(snakemake.output.optimum)
        opt_point = get_basis_values(n, snakemake.config["projection"])
        pd.DataFrame(opt_point, index=[0]).to_csv(snakemake.output.optimal_point)
        with open(snakemake.output.obj, "w") as f:
            f.write(str(n.objective))
