# Automatic Classification

The automatic classification module groups rupture planes into clusters representing distinct fault sets.

## Overview

This module performs unsupervised clustering of rupture plane orientations and locations to identify:
- Active faults with similar orientations
- Structural domains with coherent deformation patterns
- Spatial variations in fault system geometry

## Clustering Approaches

### Orientation-Based Clustering
Clusters fault planes based on their normal vector orientations using:
- **von Mises-Fisher (vMF) clustering**: Spherical clustering for directional data
- **K-Means**: Standard clustering on unit vectors
- **DBSCAN**: Density-based clustering for detecting outliers

### Automatic Cluster Detection
The module can automatically determine optimal number of clusters by:
- Testing multiple cluster numbers (up to `max_clusters`)
- Evaluating cluster quality using silhouette scores or BIC
- Selecting configuration that best separates fault orientations

## Key Parameters

- `autoclass_bool`: Enable/disable automatic classification
- `algorithm`: Clustering algorithm ('vmf_soft', 'kmeans', 'dbscan')
- `auto_determine_clusters`: Automatically find optimal cluster count
- `n_clusters`: Fixed number of clusters (if auto-determination disabled)
- `max_clusters`: Maximum clusters to test in automatic mode
- `rotation`: Apply coordinate rotation for subvertical structures

## Spatial Clustering

Optional spatial clustering can identify:
- Geographic fault domains
- Spatially separated fault segments
- Variations in fault orientation across the study area

Parameters:
- `spatial_clustering_bool`: Enable spatial clustering
- `spatial_eps`: DBSCAN distance threshold for spatial grouping

## Outputs

The classification adds cluster labels to `HyFI_results.csv`:
- `orient_cluster`: Orientation cluster ID (0, 1, 2, ...)
- `spatial_cluster`: Spatial domain ID (if enabled)

## Visualization

Results can be visualized as:
- Stereonets colored by cluster
- 3D fault planes colored by cluster
- Spatial maps showing cluster distributions

---

Happy fault imaging! 🎉