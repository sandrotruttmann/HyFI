# Fault Network Reconstruction

The fault network reconstruction module is the core component of HyFI that performs 3D fault imaging from earthquake hypocenter data.

## Overview

This module uses a Monte Carlo simulation-based approach to reconstruct 3D fault geometries by:
1. Perturbing hypocenter locations within their error ellipsoids
2. Computing local fault plane orientations from neighbor point distributions
3. Aggregating results across multiple Monte Carlo iterations
4. Detecting and removing outliers

## Key Features

- **Monte Carlo Simulation**: Accounts for location uncertainties by sampling within error ellipsoids
- **Adaptive Neighbor Search**: Uses spatiotemporal search (radius + time window) or k-nearest neighbors
- **Planar Surface Estimation**: Computes best-fit plane orientations using eigenvalue decomposition
- **Statistical Aggregation**: Combines results from multiple iterations using Fisher statistics
- **Outlier Detection**: Supports multiple methods (LOF, Isolation Forest, angle-based, DBSCAN)

## Main Parameters

### Search Parameters
- `search_radius_meters`: Spatial search radius in meters
- `search_time_window_hours`: Temporal search window in hours
- `n_neighbors`: Number of nearest neighbors (alternative to spatiotemporal search)

### Monte Carlo Settings
- `n_mc`: Number of Monte Carlo iterations (1 = no perturbation)
- `multiprocessing_bool`: Enable parallel processing

### Quality Filters
- `S1_S2_ratio`: Eigenvalue ratio threshold for planarity (e.g., 3.0)
- `min_neighbor_count`: Minimum neighbors required for valid estimation

## Outputs

The module produces a pandas DataFrame with computed fault parameters for each hypocenter:
- Mean strike, dip, and normal vector components
- Eigenvalues (S1, S2, S3) and eigenvectors
- Fault plane normal vector statistics
- Quality metrics (neighbor counts, eigenvalue ratios)

## Automatic Parameter Optimization

When enabled, the module uses Optuna to optimize `search_radius_meters` and `search_time_window_hours` based on:
- Maximizing the fraction of valid fault plane estimates
- Achieving target neighbor counts
- Minimizing outlier fractions

See [Parameter Optimization](../pareto_optimization) for details.
