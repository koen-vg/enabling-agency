import warnings

warnings.simplefilter("ignore", FutureWarning)

import hashlib
import os
import re
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import yaml

# Dirty hack to allow importing a script from a parent directory.
sys.path.insert(1, os.path.join(sys.path[0], "../scripts"))

from scripts.utilities import override_component_attrs
from scripts.workflow_utilities import build_intersection_scenarios


def average_every_nhours(n, offset, drop_leap_day=False):
    m = n.copy(with_time=False)

    snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                if c.list_name == "stores" and k == "e_max_pu":
                    pnl[k] = df.resample(offset).min()
                elif c.list_name == "stores" and k == "e_min_pu":
                    pnl[k] = df.resample(offset).max()
                else:
                    pnl[k] = df.resample(offset).mean()

    if drop_leap_day:
        sns = m.snapshots[~((m.snapshots.month == 2) & (m.snapshots.day == 29))]
        m.set_snapshots(sns)

    return m


def filter_renewables(n, countries_in):
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
    reg_renew = var_renew[countries_in]
    return reg_renew


def filter_hydro(n, countries_in):
    hydro_idx = n.storage_units.loc[n.storage_units.carrier == "hydro"].index
    var_hydro = (
        (n.snapshot_weightings.stores @ n.storage_units_t.p)
        .loc[hydro_idx]
        .groupby(n.storage_units.bus.map(n.buses.location).map(n.buses.country))
        .sum()
    )
    reg_hydro, cs = [], []
    for c in countries_in:
        if c in var_hydro.index:
            reg_hydro.append(var_hydro[c])
            cs.append(c)
    reg_hydro = pd.DataFrame(reg_hydro, index=cs)
    return reg_hydro


def filter_biomass(n, countries_in):
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
    reg_biomass, cs = [], []
    for c in countries_in:
        if c in biomass.index:
            reg_biomass.append(biomass[c])
            cs.append(c)
    reg_biomass = pd.DataFrame(reg_biomass, index=cs)
    return reg_biomass


def filter_nuclear(n, countries_in):
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
    reg_nuclear, cs = [], []
    for c in countries_in:
        if c in nuclear.index:
            reg_nuclear.append(nuclear[c])
            cs.append(c)
    reg_nuclear = pd.DataFrame(reg_nuclear, index=cs)
    return reg_nuclear


def filter_heat(n, countries_in):
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
    reg_heat = ambient_heat[countries_in]
    return reg_heat


def compute_elec_imports(n, countries_in):
    lines_in = {
        c: n.lines.loc[
            (n.lines.bus1.map(n.buses.country) == c)
            & (n.lines.bus0.map(n.buses.country) != c)
        ].index
        for c in countries_in
    }
    lines_out = {
        c: n.lines.loc[
            (n.lines.bus0.map(n.buses.country) == c)
            & (n.lines.bus1.map(n.buses.country) != c)
        ].index
        for c in countries_in
    }
    elec_imports = {
        c: (n.snapshot_weightings.generators @ n.lines_t.p0)[lines_in[c]].sum()
        if len(lines_in[c]) > 0
        else 0
        for c in countries_in
    }
    elec_exports = {
        c: -(n.snapshot_weightings.generators @ n.lines_t.p0)[lines_out[c]].sum()
        if len(lines_out[c]) > 0
        else 0
        for c in countries_in
    }
    pipeline_carriers = [
        "DC",
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
        for c in countries_in
    }
    links_out = {
        c: n.links.loc[
            out_of_country(c) & (n.links.carrier.isin(pipeline_carriers))
        ].index
        for c in countries_in
    }

    dc_imports = {
        c: (n.snapshot_weightings.generators @ n.links_t.p0)[links_in[c]].sum()
        if len(links_in[c]) > 0
        else 0
        for c in countries_in
    }
    dc_exports = {
        c: -(n.snapshot_weightings.generators @ n.links_t.p0)[links_out[c]].sum()
        if len(links_out[c]) > 0
        else 0
        for c in countries_in
    }

    dc_net_imports = {c: dc_imports[c] + dc_exports[c] for c in countries_in}
    dc_net_imports = pd.Series(dc_net_imports).sum()

    ac_net_imports = {c: elec_imports[c] + elec_exports[c] for c in countries_in}
    ac_net_imports = pd.Series(ac_net_imports).sum()

    elec_net_imports = ac_net_imports + dc_net_imports
    return elec_net_imports


def compute_nonelec_imports(n, countries_in):
    pipeline_carriers = [
        "H2 pipeline",
        "H2 pipeline retrofitted",
        "gas pipeline",
        "gas pipeline new",
        "solid biomass transport",  # Not a pipeline but functionally equivalent
        "Fischer-Tropsch",
        "biomass to liquid",
        "residential rural oil boiler",
        "services rural oil boiler",
        "residential urban decentral oil boiler",
        "services urban decentral oil boiler",
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
        for c in countries_in
    }
    links_out = {
        c: n.links.loc[
            out_of_country(c) & (n.links.carrier.isin(pipeline_carriers))
        ].index
        for c in countries_in
    }

    pipeline_imports = {
        c: (n.snapshot_weightings.generators @ n.links_t.p0)[links_in[c]].sum()
        if len(links_in[c]) > 0
        else 0
        for c in countries_in
    }
    pipeline_exports = {
        c: -(n.snapshot_weightings.generators @ n.links_t.p0)[links_out[c]].sum()
        if len(links_out[c]) > 0
        else 0
        for c in countries_in
    }

    pipeline_net_imports = {
        c: pipeline_imports[c] + pipeline_exports[c] for c in countries_in
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
    gas_import = gas_import.reindex(countries_in).fillna(0)

    nonelec_net_imports = pipeline_net_imports + gas_import
    return nonelec_net_imports.sum()


def compute_local_prod(n, countries_in):
    reg_renew = filter_renewables(n, countries_in)
    reg_hydro = filter_hydro(n, countries_in)
    reg_biomass = filter_biomass(n, countries_in)
    reg_nuclear = filter_nuclear(n, countries_in)
    reg_heat = filter_heat(n, countries_in)

    local_energy_prod = (
        pd.concat(
            [reg_renew, reg_hydro, reg_biomass, reg_nuclear, reg_heat], axis="columns"
        )
        .fillna(0)
        .sum(axis="columns")
        .sum(axis="rows")
    )
    return local_energy_prod


if __name__ == "__main__":
    config_names = [
        # "PL+balt-focus",
        # "nordics-focus",
        # "adriatic-focus",
        # "DE-focus",
        # "FR-focus",
        "british-isles-focus",
        "iberia-focus",
    ]

    # results_dir = "results_backup"
    results_dir = "results"

    num_robust = 300

    processing_dir = "processing/robust_networks"

    # Make sure the processing directory exists
    Path(processing_dir).mkdir(exist_ok=True)

    configs = {}
    for config_name in config_names:
        with open(f"../../config/config-{config_name}.yaml") as f:
            configs[config_name] = yaml.safe_load(f)

    scenarios = {
        name: build_intersection_scenarios(config["intersection_scenarios"])
        for name, config in configs.items()
    }

    overrides = override_component_attrs(
        "../modules/pypsa-eur-sec/data/override_component_attrs"
    )

    for config_name, scenario_list in scenarios.items():
        print(f"Starting to compute statistics for {config_name}")
        eps = configs[config_name]["intersection_scenarios"]["eps"][0]
        scenario_hash = hashlib.md5("".join(scenario_list).encode()).hexdigest()[:8]
        config = configs[config_name]
        robust_base = next(
            filter(
                lambda s: s.startswith(config["robust_networks"]["robust_base"]),
                scenarios[config_name],
            )
        )
        if "opts_substitutions" in config["robust_networks"]:
            for orig, replacement in config["robust_networks"][
                "opts_substitutions"
            ].items():
                robust_base = re.sub(
                    orig, replacement, robust_base, flags=re.IGNORECASE
                )
        dir_name = (
            f"../../{results_dir}/{config_name}/robust_networks/"
            f"{robust_base}_e{eps}_{scenario_hash}"
        )
        first_n = sorted(
            [
                os.path.join(dir_name, file)
                for file in os.listdir(dir_name)
                if file.endswith(".nc")
            ],
            key=os.path.getmtime,
        )[:num_robust]

        robust_networks = {}
        for file in first_n:
            try:
                n = pypsa.Network(
                    file, override_component_attrs=overrides
                )
                # Test if the network makes sense
                if "country" not in n.buses.columns:
                    raise ValueError("No country attribute in buses")
                robust_networks[os.path.basename(file)] = n
            except Exception:
                pass

        print(f"Loaded {len(robust_networks)} networks.")

        # Get inside countries. Need to remove Montenegro manually, as
        # it ended up erroneously in the config file.
        countries_in = [
            c for c in configs[config_name]["projection_regions"]["in"] if c != "MT"
        ]

        stats = pd.DataFrame(
            columns=[
                "net_elec_exports",
                "net_nonelec_exports",
                "net_exports",
                "local_prod",
            ],
            index=robust_networks.keys(),
        )

        stats["net_elec_exports"] = (
            pd.Series(
                {
                    f: -compute_elec_imports(n, countries_in)
                    for f, n in robust_networks.items()
                }
            )
            / 1e6
        )

        stats["net_nonelec_exports"] = (
            pd.Series(
                {
                    f: -compute_nonelec_imports(n, countries_in)
                    for f, n in robust_networks.items()
                }
            )
            / 1e6
        )

        stats["net_exports"] = stats["net_elec_exports"] + stats["net_nonelec_exports"]

        stats["local_prod"] = (
            pd.Series(
                {
                    f: compute_local_prod(n, countries_in)
                    for f, n in robust_networks.items()
                }
            )
            / 1e6
        )

        stats.to_csv(
            os.path.join(
                processing_dir,
                config_name,
                f"{robust_base}_e{eps}_{scenario_hash}.csv",
            )
        )

        # Ensure the output dir exists
        Path(
            os.path.join(
                processing_dir,
                config_name,
                f"{robust_base}_e{eps}_{scenario_hash}",
            )
        ).mkdir(exist_ok=True)
        for f, n in robust_networks.items():
            m = average_every_nhours(n, "8760H", drop_leap_day=True)
            m.export_to_netcdf(
                os.path.join(
                    processing_dir,
                    config_name,
                    f"{robust_base}_e{eps}_{scenario_hash}",
                    f,
                )
            )
