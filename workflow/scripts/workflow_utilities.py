# SPDX-FileCopyrightText: 2023 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Simple utilities aiding the Snakemake workflow."""


import copy
import hashlib
import json
import os
import re
from os.path import join

import yaml


def validate_config(config):
    """Check for common mistakes in the config file."""
    # Make sure that the config has a name.
    if "name" not in config:
        raise ValueError(f"Config file does not have a name key.")

    # Check that the config doesn't have both a "scenario" and
    # "intersection_scenarios" section.
    if "scenario" in config and "intersection_scenarios" in config:
        raise ValueError(
            f"Config file has both a 'scenario' and an "
            "'intersection_scenarios' section. Only one of these is allowed."
        )

    # Check for an easy mistake in projection specification: not
    # enabling the `scale_by_years` option.
    try:
        warn = False
        years = config["scenario"]["year"]
        if any("-" in y or "+" in y for y in years):
            for dim in config["projection"].values():
                for spec in dim:
                    if "capital_cost" in spec.values() and "scale_by_years" not in spec:
                        warn = True
        if warn:
            print(
                "\n=====WARNING=====\nDid you forget to add `scale_by_years` to"
                f" projection config in config-{name}?\n"
            )
    except (IndexError, KeyError):
        pass


def translate_projection_regions(
    proj_config: dict, regions_config: dict, all_countries: list[str]
) -> dict:
    """Translate named regions into explicit lists of countries.

    For each `regions: "foo"` option in a projection specification in
    `proj_config`, replace the string "foo" with the contents of
    `regions_config["foo"]`. Replace "not-foo" with a list of all
    countries excluding those in `regions_config["foo"]`. This allows
    regions and their complements to be specified cleanly without
    repetition for each dimension.

    """
    new_proj_config = copy.deepcopy(proj_config)
    for spec in new_proj_config.values():
        for comp in spec:
            if "regions" in comp:
                if isinstance(comp["regions"], str):
                    # We have something to replace! Is it a named
                    # region or its complement?
                    if comp["regions"].startswith("not-"):
                        region_name = comp["regions"][4:]
                        complement = True
                    else:
                        region_name = comp["regions"]
                        complement = False
                    if region_name not in regions_config:
                        raise ValueError(
                            f"Used region name {region_name} in projection config without"
                            " defining it in the `projection_regions` section of the config!"
                        )
                    countries = regions_config[region_name]
                    if complement:
                        countries = list(set(all_countries).difference(set(countries)))
                    comp["regions"] = countries
    return new_proj_config


def build_intersection_scenarios(scenarios_config: dict) -> list[str]:
    scenarios: list[str] = []
    common_wildcards: dict = scenarios_config["common"]
    for name, varying_wildcards in scenarios_config["scenarios"].items():
        # Check that the common and varying wildcards don't overlap
        # much; we only allow the {opts} and {sector_opts} wildcards
        # to specified in both sections.
        repeated_wildcards = set(common_wildcards).intersection(set(varying_wildcards))
        repeated_wildcards.discard("opts")
        repeated_wildcards.discard("sector_opts")
        if repeated_wildcards:
            raise ValueError(
                "In the intersection scenarios configuration, the following wildcards "
                f"were repeated between the common and scenario parts: {repeated_wildcards}"
            )
        # Merge wildcard dictionaries; the right-most dictionary takes
        # precendence in case of conflicts.
        w = common_wildcards | varying_wildcards

        # Include some special handling of {opts} and {sector_opts}
        # wildcards if they are present in both static and common
        # wildcards.
        for o in ["opts", "sector_opts"]:
            if o in common_wildcards and o in varying_wildcards:
                w[o] = varying_wildcards[o] + "-" + common_wildcards[o]

        # Normalise some wildcard names across pypsa-eur and pypsa-eur-sec
        w["ll"] = w["lv"] if "lv" in w else w["ll"]
        w["opts"] = w["sector_opts"] if "sector_opts" in w else w["opts"]

        # Build up the complete name of each scenario.
        scenario = (
            f"{name}_{w['year']}_{w['simpl']}_{w['clusters']}_{w['ll']}_{w['opts']}"
        )
        if "planning_horizons" in w:
            scenario = scenario + "_" + w["planning_horizons"]
        scenarios.append(scenario)

    return scenarios


def hash_config(configuration: dict):
    """Compute hash of most config file contents.

    The hash helps optimisation runs initiated with the same config
    file to be reused, specifically in the `compute_near_opt`
    snakemake rule. However, we exclude some specific keywords in the
    near_opt_approx configuration from the hash computation, since
    they are only parameters but do not influence the results of the
    algorithm beyong accuracy.

    """
    config_to_be_hashed = copy.deepcopy(configuration)
    # Remove the following keywords from the near_opt_approx
    # configuration in the config file.
    for i in [
        "iterations",
        "conv_epsilon",
        "conv_iterations",
        "directions_angle_separation",
        "num_parallel_solvers",
        "qhull_options",
        "intersection_pre_cluster",
    ]:
        if i in config_to_be_hashed["near_opt_approx"]:
            del config_to_be_hashed["near_opt_approx"][i]
    if "solving" in config_to_be_hashed["pypsa-eur"]:
        del config_to_be_hashed["pypsa-eur"]["solving"]
    if "scenario" in config_to_be_hashed:
        del config_to_be_hashed["scenario"]
    if "robust_networks" in config_to_be_hashed:
        del config_to_be_hashed["robust_networks"]
    return hashlib.md5(
        str(json.dumps(config_to_be_hashed, sort_keys=True)).encode()
    ).hexdigest()[:8]


def parse_net_spec(spec: str) -> dict:
    """Parse a network specification and return it as a dictionary."""
    # Define the individual regexes for all the different wildcards.
    rs = {
        "year": r"([0-9]+)(-[0-9]+)?(\+([0-9]+)(-[0-9]+)?)*",
        "simpl": r"[a-zA-Z0-9]*|all",
        "clusters": r"[0-9]+m?|all|[0-9]+-[0-9]+-[0-9]+",
        "ll": r"(v|c)([0-9\.]+|opt|all)|all",
        "lv": r"[a-z0-9\.]+",
        "sector_opts": r"[-+a-zA-Z0-9\.]*",
        "planning_horizon": r"[0-9]{4}|",
        "eps": r"[0-9\.]+(uni([0-9]+)(-[0-9]+)?(\+([0-9]+)(-[0-9]+)?)*)?",
    }
    # Make named groups out of the individual groups, such that e.g.
    # r"[0-9\.]+(uni)?" becomes r"(?P<eps>[0-9\.]+(uni)?)".
    G = {n: f"(?P<{n}>{r})" for n, r in rs.items()}
    # Build the complete regex out of the individual groups
    full_regex_pypsa_eur_sec = (
        f"({G['year']}_)?"
        f"{G['simpl']}_{G['clusters']}_{G['lv']}_{G['sector_opts']}_{G['planning_horizon']}"
        f"(_e{G['eps']})?"
    )
    m = re.search(full_regex_pypsa_eur_sec, spec)
    if m is not None:
        return m.groupdict()

    raise ValueError(f"Could not parse network space {spec}")


def parse_year_wildcard(w):
    """
    Parse a {year} wildcard to a list of years.

    The wildcard can be of the form `1980+1990+2000-2002`; a set of
    ranges (two years joined by a `-`) and individual years all
    separated by `+`s. The above wildcard is parsed to the list [1980,
    1990, 2000, 2001, 2002].
    """
    years = []
    for rng in w.split("+"):
        try:
            if "-" in rng:
                # `rng` is a range of years.
                [start, end] = rng.split("-")
                # Check that the range is well-formed.
                if end < start:
                    raise ValueError(f"Malformed range of years {rng}.")
                # Add the range (inclusive) to the set of years.
                years.extend(range(int(start), int(end) + 1))
            else:
                # `rng` is just a single year.
                years.append(int(rng))
        except ValueError:
            raise ValueError(f"Illegal range of years {rng} encountered.")
    # Sort the years before returning.
    return sorted(years)
