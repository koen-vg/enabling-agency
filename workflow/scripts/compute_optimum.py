# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Optimising a PyPSA network with respect to its total system costs."""

import logging
from pathlib import Path

import pandas as pd
import pypsa
from _helpers import configure_logging
from pypsa.linopf import network_lopf
from utilities import get_basis_values

if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # Load the network and solving options.
    n = pypsa.Network(snakemake.input.network)
    solver_options = snakemake.config["pypsa-eur"]["solving"]["solver"]
    solver_name = solver_options.pop("name")
    tmpdir = snakemake.config["pypsa-eur"]["solving"].get("tmpdir", None)
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    # Solve the network for the cost optimum and then get its
    # coordinates in the basis.
    logging.info("Compute initial, optimal solution.")
    network_lopf(
        n,
        solver_name=solver_name,
        solver_options=solver_options,
        solver_dir=tmpdir,
    )
    opt_point = get_basis_values(n, snakemake.config["projection"])

    # Write the result to the given output files. Save the objective
    # value for further processing.
    n.export_to_netcdf(snakemake.output.optimum)
    pd.DataFrame(opt_point, index=[0]).to_csv(snakemake.output.optimal_point)
    with open(snakemake.output.obj, "w") as f:
        f.write(str(n.objective))