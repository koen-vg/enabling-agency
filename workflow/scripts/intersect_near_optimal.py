# SPDX-FileCopyrightText: 2023 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Computes intersections of near-optimal feasible spaces."""

import logging

import numpy as np
import pandas as pd
from _helpers import configure_logging
from geometry import ch_centre_from_constraints, intersection
from scipy.spatial import ConvexHull

if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # Read qhull options from config
    qhull_options = snakemake.config["near_opt_approx"]["qhull_options"]

    # Read the near-optimal spaces we are intersecting.
    points = [pd.read_csv(file, index_col=0) for file in snakemake.input.hulls]

    # Scale all hulls down to fit in the unit cube.
    scaling_factors = (
        pd.concat(points, axis="rows").max() - pd.concat(points, axis="rows").min()
    )
    points_scaled = [P / scaling_factors for P in points]
    hulls_scaled = [
        ConvexHull(P, qhull_options=qhull_options["near_opt_approx"])
        for P in points_scaled
    ]

    # For logging purposes, print the number of equations for each hull.
    logging.info("Number of equations for each hull:")
    logging.info([H.equations.shape[0] for H in hulls_scaled])

    # Intersecting all convex hulls.
    logging.info("Trying to intersect hulls.")
    intersected_points = intersection(
        hulls=hulls_scaled,
        qhull_options=qhull_options["intersection"],
        pre_cluster=snakemake.params.pre_cluster_enabled,
        pre_cluster_n_clusters=snakemake.params.pre_cluster_n_clusters,
    )
    if intersected_points is None:
        raise RuntimeError(
            "No intersection was possible. Consider working with"
            " robust solutions for single years."
        )

    # Scale the intersected points back up to the original scale.
    intersected_points_df = pd.DataFrame(intersected_points, columns=points[0].columns)
    intersected_points_df *= scaling_factors
    # Write intersected space to the given output.
    intersected_points_df.to_csv(snakemake.output.intersection)

    # In the above, we computed the intersection in a space where
    # every dimension is scaled independently. This of course might
    # change the Chebyshev centre and radius. We now need to compute
    # the actual centre point and radius of the (uniformly scaled)
    # intersection.
    scaling_factor = scaling_factors.min()
    hulls = [
        ConvexHull(P / scaling_factor, qhull_options=qhull_options["near_opt_approx"])
        for P in points
    ]
    constraints = np.concatenate([h.equations for h in hulls])
    c, radius, _ = ch_centre_from_constraints(constraints)
    centre_df = pd.DataFrame(c * scaling_factor).T
    centre_df.columns = points[0].columns
    centre_df.to_csv(snakemake.output.centre)
    with open(snakemake.output.radius, "w") as f:
        f.write(str(radius * scaling_factor))
