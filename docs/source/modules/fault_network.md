# Fault Network Reconstruction

The fault network reconstruction module is the core component of **HyFI** that performs 3D fault imaging from earthquake hypocenter data.

## Overview

This module uses a Monte Carlo simulation-based approach to reconstruct 3D rupture geometries by:
1. Perturbing hypocenter locations within their error ellipsoids
2. Computing rupture plane orientations from neighbor point distributions using principal component analysis (PCA)
3. Aggregating results across multiple Monte Carlo iterations using spherical statistics
4. Detecting and removing outliers within the sequence

## Key Features

- **Monte Carlo Simulation**: Accounts for location uncertainties by sampling within error ellipsoids
- **Adaptive Neighbor Search**: Uses spatiotemporal search (radius + time window) or k-nearest neighbors
- **Planar Surface Estimation**: Computes best-fit rupture plane orientations using PCA
- **Statistical Aggregation**: Combines results from multiple iterations using Fisher statistics
- **Outlier Detection**: Removes outlier events to ensure consistent rupture plane reconstruction multiple methods (LOF, Isolation Forest, DBSCAN)

## Main Parameters

### Search Parameters
- `search_radius_meters`: Spatial search radius in meters
- `search_time_window_hours`: Temporal search window in hours
- `n_neighbors`: Number of nearest neighbors (alternative to spatiotemporal search)

### Monte Carlo Settings
- `n_mc`: Number of Monte Carlo iterations (1 = no perturbation)

### Quality Filters
- `S1_S2_ratio`: Eigenvalue ratio threshold for planarity (e.g., 3.0)
- `min_neighbor_count`: Minimum neighbors required for valid estimation

## Outputs

The module produces a pandas DataFrame with computed fault parameters for each hypocenter:
- Mean strike, dip, and normal vector components of rupture planes
- Eigenvalues (S1, S2, S3) and eigenvectors from PCA
- Rupture plane normal vector statistics
- Quality metrics (e.g., eigenvalue ratios)

## Automatic Parameter Optimization

xxx TODO: COMPLEMENT DOCU

When enabled, the module uses Optuna to optimize `search_radius_meters` and `search_time_window_hours` automatically based on:
- Maximizing the fraction of valid fault plane estimates
- Minimizing the misfit between rupture plane orientations and focal planes
- Minimizing outlier fractions

See [Parameter Optimization](../pareto_optimization) for details.



---

Happy fault imaging! 🎉