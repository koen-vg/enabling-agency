import hashlib
import os
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from annoy import AnnoyIndex
from scipy.spatial import ConvexHull
from scipy.stats import gmean
from scripts.geometry import ch_centre, slice_hull
from scripts.workflow_utilities import build_intersection_scenarios


def slice_stats(
    I: ConvexHull,
    min_slice=0.01,
    max_slice=0.99,
    num_slices=99,
):
    """Compute slice volumes along different dimensions at given ranges."""

    # Define slice values
    min_val = I.points[:, 0].min()
    max_val = I.points[:, 0].max()
    slice_values = (
        np.linspace(min_slice, max_slice, num_slices) * (max_val - min_val) + min_val
    )
    stats = pd.DataFrame(index=slice_values, columns=["mean_width"], dtype="float64")

    for slice_val in slice_values:
        s = slice_hull(
            I,
            0,  # Note: we assume slicing on the first dimension!
            slice_val,
            qhull_options="QJ",
        )
        if s is not None:
            # We get s as an array of points; this allows us to
            # compute widths but not volume or chebyshev radius.
            stats.loc[slice_val, "mean_width"] = np.mean(s.max(axis=0) - s.min(axis=0))

    return stats


def projected_hull(
    I: pd.DataFrame,
    dims: list[str],
):
    """Compute convex hull of points projected to given dimensions"""

    # In order to use qhull to approximate this operation well, we
    # need to scale down to the unit cube.
    scaling_factors = I[dims].max(axis=0)
    t = time.time()
    print("Starting convex hull computation...")
    projected_I = ConvexHull(
        I[dims] / scaling_factors,
        qhull_options="Qt W0.015 C0.001",
        # qhull_options="QJ W0.1 C0.001",
    )
    print(f"Done: {time.time() - t:.0f} seconds, {len(projected_I.vertices)} vertices.")
    points = pd.DataFrame(projected_I.points[projected_I.vertices], columns=dims)
    return scaling_factors * points


if __name__ == "__main__":
    config_names = [
        "PL+balt-focus",
        "nordics-focus",
        "adriatic-focus",
        "DE-focus",
        "FR-focus",
        "british-isles-focus",
        "iberia-focus",
    ]

    # Do we compute convex hulls and store them, or try to read convex
    # hulls from storage?
    compute_hulls = True

    # results_dir = "results_backup"
    results_dir = "results"

    processing_dir = "processing/projected_intersection_hulls"

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

    intersections = {}
    for config_name, scenario_list in scenarios.items():
        eps = configs[config_name]["intersection_scenarios"]["eps"][0]
        scenario_hash = hashlib.md5("".join(scenario_list).encode()).hexdigest()[:8]
        try:
            intersections[config_name] = pd.read_csv(
                (
                    f"../../{results_dir}/{config_name}/intersection/"
                    f"intersection_e{eps}_{scenario_hash}.csv"
                ),
                index_col=0,
            )
        except FileNotFoundError:
            print(f"Intersection not found for {config_name}")

    # We find slice volumes in two different setting: inside near-opt
    # space volumes and outside near-opt space volumes.
    for side, slice_dims, other_dims in [
        (
            "outside",
            ["solar-in", "onwind-in", "offwind-in", "hydrogen-in"],
            ["solar-out", "onwind-out", "offwind-out", "hydrogen-out"],
        ),
        (
            "inside",
            ["solar-out", "onwind-out", "offwind-out", "hydrogen-out"],
            ["solar-in", "onwind-in", "offwind-in", "hydrogen-in"],
        ),
    ]:
        # Start a multiprocessing pool with up to 40 processes to
        # speed up convex hull computations.
        with Pool(40) as p:
            if compute_hulls:
                # Prepare arguments for computing convex hulls of all
                # projection combinations.
                args = [
                    (intersections[config_name], [slice_dim] + other_dims)
                    for config_name in config_names
                    for slice_dim in slice_dims
                ]
                # Use the multiprocessing pool to compute the convex hulls in parallel.
                print("Computing convex hulls of projections...")
                hulls = p.starmap(projected_hull, args)
                print("Done.")

                # Organise the results in a dictionary by config name and
                # slice dimension.
                args_idx = [
                    (config_name, slice_dim)
                    for config_name in config_names
                    for slice_dim in slice_dims
                ]
                hulls = dict(zip(args_idx, hulls))

                # Store the hulls in the processing directory.
                for (config_name, slice_dim), hull in hulls.items():
                    hull.to_csv(
                        os.path.join(processing_dir, f"{config_name}_{slice_dim}.csv")
                    )

            else:
                # Read hulls from storage.
                print("Reading projected intersections from storage...")
                hulls = {}
                for config_name in config_names:
                    for slice_dim in slice_dims:
                        hulls[(config_name, slice_dim)] = pd.read_csv(
                            os.path.join(
                                processing_dir, f"{config_name}_{slice_dim}.csv"
                            ),
                            index_col=0,
                        )

            # Now that we have the right projections of the
            # intersection, we compute their convex hulls and then
            # slice. Note that everything from here is scaled to bn
            # EUR, down by 1e9.
            print("Reconstructing convex hulls of projections...")
            t = time.time()
            args = [
                ConvexHull(hulls[(config_name, slice_dim)] / 1e9, qhull_options="Qt")
                for config_name in config_names
                for slice_dim in slice_dims
            ]
            print(f"Done ({time.time() - t:.0f} seconds).")
            args_idx = [
                (config_name, slice_dim)
                for config_name in config_names
                for slice_dim in slice_dims
            ]
            print("Computing slices of projected intersections...")
            stats = dict(zip(args_idx, p.map(slice_stats, args)))
            print("Done.")

            # Save the results in a separate csv file for each config
            # name. We organise the results in a pandas DataFrame with
            # a column for each stat, multi-indexed by slice dimension
            # and slice value.
            for config_name in config_names:
                stats_df = pd.concat(
                    [stats[(config_name, slice_dim)] for slice_dim in slice_dims],
                    keys=slice_dims,
                )
                stats_df.to_csv(f"processing/{config_name}_{side}_slice_stats.csv")
