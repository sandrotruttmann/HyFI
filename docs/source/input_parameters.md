# HyFI Input Parameters

This document provides detailed explanations for all **HyFI** parameters in `config_single_TEMPLATE.json` and `config_multi_TEMPLATE.json`.

---

## Table of Contents
- [Metadata](#metadata)
- [Global Settings](#global-settings)
- [Input Data](#input-data)
- [Fault Network](#fault-network)
- [Model Validation](#model-validation)
- [Auto Classification](#auto-classification)
- [Stress Analysis](#stress-analysis)
- [Visualization](#visualization)
- [Multi-Sequence Processing](#multi-sequence-processing)

---

## Metadata

Configuration metadata for documentation and tracking purposes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `workflow_name` | string | Name for this analysis workflow (e.g., "St. Leonard Sequence") |
| `workflow_version` | string | Version number for tracking configuration changes (e.g., "1.0.0") |
| `created_date` | string | ISO 8601 formatted date when config was created (e.g., "2025-11-11T00:00:00") |
| `description` | string | Optional description of the analysis |

---

## Global Settings

Settings that apply to the entire workflow.

### Common Settings (Single & Multi-Sequence)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_directory` | string | `"./hyfi_output"` | Directory where all output files will be saved. For single-sequence: all outputs go to this directory. For multi-sequence: individual sequence results go to subdirectories within this path |
| `log_level` | string | `"INFO"` | Logging verbosity. Options: `"DEBUG"` (verbose), `"INFO"` (standard), `"WARNING"` (errors only), `"ERROR"`, `"CRITICAL"` |

### Multi-Sequence Only Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parallel_processing` | boolean | `false` | Enable parallel processing of segmentation steps and per-sequence analysis. **Note**: Currently single-threaded (future feature). Set to `true` for future compatibility, but sequential execution will be used |
| `max_workers` | integer | `4` | Maximum number of parallel worker processes for multi-sequence processing. Used when `parallel_processing: true` is implemented. Range: 1-16 depending on system CPU cores. Recommended: 4-8 for typical systems |
| `save_individual_results` | boolean | `true` | Save intermediate and individual sequence analysis results to disk. When `true`, creates subdirectories for each sequence with full analysis outputs; when `false`, only final merged results are kept (saves disk space but loses per-sequence details) |

---

## Input Data

Configuration for input earthquake catalog files.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `hypocenter_file` | string | **Yes** | Path to hypocenter catalog file (e.g., "data_examples/A0_data.csv") |
| `hypocenter_separator` | string | Yes | Column separator for hypocenter file. Options: `","` (CSV), `"\t"` (TSV), `";"` (semicolon) |
| `focal_mechanism_file` | string or null | No | Path to focal mechanism catalog. Set to `null` if not available |
| `focal_mechanism_separator` | string | No | Column separator for focal mechanism file. Options: `","`, `"\t"`, `";"` |

### Required Hypocenter Columns
- `YR`, `MO`, `DY`, `HR`, `MI`, `SC` (Date/time)
- `LAT`, `LON`, `Z` (Location)
- `X`, `Y` (Local coordinates)
- `EX`, `EY`, `EZ` (Location uncertainties)
- `ID` (Event identifier)
- `MAG` or `ML` or `Mw` (Magnitude)

### Required Focal Mechanism Columns (if used)
- `ID` (Event identifier matching hypocenter catalog)
- `Strike1`, `Dip1`, `Rake1` (First nodal plane)
- `Strike2`, `Dip2`, `Rake2` (Second nodal plane)
- `A` (Active plane indicator: `1` or `2`, optional but recommended)

---

## Fault Network

Parameters for 3D fault network reconstruction using NN search and PCA.

### Core Network Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `monte_carlo_simulations` | integer | `1000` | Number of Monte Carlo iterations for uncertainty quantification. Range: 1-inf. Higher values = more robust statistics but slower computation. Typical: 1000 |
| `search_radius_meters` | float or string | `100.0` | Spatial search radius for neighbor detection in meters. Use numeric value or `"auto"` for automatic optimization. Typical range: 50-1000 m |
| `search_time_window_hours` | float or string | `9999999` | Temporal search window for neighbor detection in hours. Use numeric value or `"auto"` for automatic optimization. Large value (e.g. 9999999h) includes all events |
| `magnitude_type` | string | `"ML"` | Magnitude type to use for rupture radius calculation. Options: `"ML"` (local magnitude), `"Mw"` (moment magnitude) |

### Outlier Detection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `remove_outliers` | boolean | `false` | Enable outlier detection and removal before fault network reconstruction (clean-up of hypocenters) |
| `outlier_method` | string | `"DBSCAN"` | Outlier detection algorithm. Options: `"DBSCAN"` (density-based, good for distinct clusters, **default**), `"LOF"` (local outlier factor, good for varying density), `"IForest"` (isolation forest, robust and fast) |
| `lof_n_neighbors` | integer or null | `null` | **LOF-specific**: Number of neighbors to consider. `null` = auto-tuned based on dataset size, or integer 10-50 |
| `lof_contamination` | string or float | `"auto"` | **LOF-specific**: Expected outlier proportion. `"auto"` or float 0.01-0.5 (1%-50%) |
| `if_n_estimators` | integer | `100` | **IForest-specific**: Number of trees in the forest. Range: 50-200. Higher = more robust but slower |
| `if_max_samples` | string or integer | `"auto"` | **IForest-specific**: Number of samples per tree. `"auto"` = auto-tuned, or integer. Lower = faster but less accurate |
| `if_contamination` | float | `0.05` | **IForest-specific**: Expected outlier proportion. Float 0.01-0.5 (5% = 0.05, conservative default) |
| `if_random_state` | integer | `42` | **IForest-specific**: Random seed for Isolation Forest reproducibility |

**Note**: Events with valid focal mechanism data (A=1 or A=2) are protected from outlier removal.

### Focal Mechanism Constraints

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_focal_constraints` | boolean | `false` | Use focal mechanism data to constrain fault plane selection. Requires `focal_mechanism_file` to be specified. When enabled, the algorithm selects between the two nodal planes based on consistency with neighboring events |

### Automatic Parameter Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_optimize_parameters` | boolean | `false` | Enable automatic optimization of `search_radius_meters` and `search_time_window_hours`. When enabled, ignores manual values for these parameters |
| `optimization_method` | string | `"optuna"` | Optimization algorithm. Options: `"optuna"` (TPE sampler, **recommended**, default), `"grid_search"` (thorough grid search), `"pareto"` (multi-objective), `"heuristic"` (fast heuristic) |
| `optimization_use_adaptive_weights` | boolean | `true` | When `true`, automatically adjusts weights based on number of focal mechanisms (reduces focal weight for datasets with few focals, increases recovery importance) and dataset density (adjusts recovery expectations for sparse datasets). When `false`, uses fixed weights (original behavior). |
| `optimization_random_state` | integer | `42` | Random seed for optimization reproducibility |
| `optimization_plot_results` | boolean | `false` | Generate visualization plots of optimization results |
| `optimization_r_nn_range` | array or null | `[50, 1000]` | Search radius range for optimization `[min_meters, max_meters]`. Use `null` for automatic range determination |
| `optimization_dt_nn_range` | array or null | `[100, 50000]` | Time window range for optimization `[min_hours, max_hours]`. Use `null` for automatic range determination |

#### Grid Search Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimization_grid_points` | integer | `25` | Grid resolution for grid_search method. Total evaluations = `grid_points²`. Range: 10-50. Higher = more thorough but slower (25 = 625 evaluations) |

#### Optuna Optimization Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimization_n_trials` | integer | `50` | Total number of trials for Optuna optimization. Range: 30-500. Higher = more thorough |
| `optimization_sampler` | string | `"tpe"` | Optuna sampling algorithm. Options: `"tpe"` (Tree-structured Parzen Estimator, **recommended**), `"cmaes"` (CMA-ES), `"random"` (Random sampling) |
| `optimization_n_startup_trials` | integer | `10` | Number of random trials before sampler-specific optimization starts. Range: 5-20 |
| `optimization_early_stopping_rounds` | integer or null | `null` | **Early stopping**: Stop if no improvement for N consecutive trials. `null` = disabled. Recommended: 10-20 for n_trials=50, 20-30 for n_trials=200. Saves computation time |
| `optimization_early_stopping_threshold` | float | `0.0001` | Minimum improvement to be considered significant for early stopping. Range: 1e-5 to 1e-3. Lower = more conservative |

#### Pareto Multi-Objective Optimization Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimization_pareto_sampler` | string | `"nsga2"` | Pareto optimization sampler. Options: `"nsga2"` (NSGA-II, **recommended**), `"nsga3"` (NSGA-III), `"random"` |
| `optimization_pareto_population` | integer | `50` | Population size for evolutionary algorithms. Range: 30-100 |

---

## Model Validation

Validation of reconstructed fault planes using focal mechanism data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable model validation module. Requires `focal_mechanism_file` |

### Validation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `check_magnitude_consistency` | boolean | `true` | Verify magnitude consistency between hypocenter and focal mechanism catalogs |
| `check_location_consistency` | boolean | `true` | Verify spatial location consistency between catalogs |
| `maximum_distance_km` | float | `1.0` | Maximum distance (km) between hypocenter and focal mechanism location for matching |
| `maximum_magnitude_difference` | float | `0.2` | Maximum magnitude difference for matching hypocenters with focal mechanisms |

---

## Auto Classification

Automatic classification of fault structures based on orientation and spatial clustering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable automatic classification module |

### Orientation Clustering Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_determine_clusters` | boolean | `true` | Automatically determine optimal number of orientation clusters using silhouette analysis |
| `max_clusters` | integer | `8` | Maximum number of orientation clusters to consider when auto-determining. Range: 2-15 |
| `number_of_clusters` | integer | `2` | Fixed number of orientation clusters. Used only if `auto_determine_clusters=false` |
| `clustering_algorithm` | string | `"vmf_soft"` | Clustering algorithm. Options: `"vmf_soft"` (von Mises-Fisher soft, **recommended**), `"vmf_hard"` (von Mises-Fisher hard), `"skm"` (spherical k-means) |
| `rotate_poles_before_analysis` | boolean | `true` | Rotate fault normal vectors to same hemisphere before clustering. **Recommended for sub-vertical faults**. Ensures all poles point to similar direction |
| `convergence_tolerance` | float | `1e-6` | Convergence tolerance for clustering algorithms (1e-4 for SphericalKMeans, 1e-6 for VonMisesFisher) |
| `maximum_iterations` | integer | `300` | Maximum iterations for clustering algorithms |

### Spatial Sub-Clustering Parameters

All spatial sub-clustering parameters are organized under the `spatial_sub_clustering` object:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_spatial_clustering` | boolean | `true` | Enable spatial sub-clustering within orientation clusters to identify separate fault structures |
| `spatial_clustering_method` | string | `"dbscan"` | Spatial clustering method. Options: `"dbscan"` (density-based, **recommended**), `"kmeans"` (k-means), `"hierarchical"` (agglomerative) |
| `min_events_per_cluster` | integer | `10` | Minimum number of events required to form a valid spatial cluster |
| `use_fault_plane_points_for_clustering` | boolean | `false` | Use enhanced point cloud (fault plane surface points) instead of hypocenters for spatial clustering. Improves spatial resolution when enabled. When `true`, generates synthetic points on fault plane surfaces |
| `fault_plane_point_density_meters` | float | `10.0` | Target spacing in meters between points on fault plane circumference. Range: 5-50m. **Lower values = denser point cloud, more computation**. Typical: 10-25m |
| `fault_plane_radius_interval_meters` | float | `10.0` | Spacing in meters between concentric circles when generating fault plane points. Range: 5-50m. **Lower values = more circles and higher resolution**. Typical: 10-25m |
| `use_anisotropic_eps` | boolean | `false` | Use anisotropic (direction-dependent) distance metrics for DBSCAN. When `true`, different distance thresholds are applied in-plane vs. out-of-plane, improving detection of elongated fault structures |
| `in_plane_eps_meters` | float | `500.0` | DBSCAN `eps` parameter in meters for **in-plane** distances (when using anisotropic metric). Range: 100-1000m. Lower = tighter clusters |
| `out_of_plane_eps_meters` | float | `50.0` | DBSCAN `eps` parameter in meters for **out-of-plane** distances (when using anisotropic metric). Range: 10-100m. Typically much smaller than `in_plane_eps_meters` |
| `anisotropic_min_samples` | integer | `5` | DBSCAN `min_samples` parameter when using anisotropic metric. Minimum number of neighbors required. Range: 3-20. Higher = more conservative clustering |

---

## Stress Analysis

Fault stress analysis and failure assessment using regional stress field.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable fault stress analysis module |

### Regional Stress Field Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_shapefile` | boolean | `false` | Enable spatially-varying stress field from shapefile. When `true`, stress parameters are read from shapefile instead of using fixed values below |
| `shapefile_path` | string/null | `null` | Path to shapefile (.shp) with spatially-varying stress field polygons. Only used if `use_shapefile` is `true`. Shapefile must have columns: `s1_trend`, `s1_plunge`, `s3_trend`, `s3_plunge`, `R`. Example: `"data_examples/Stressfield/CH_stressfield_Kastrup.shp"` |
| `sigma1_trend_degrees` | float/null | `null` | σ₁ (maximum principal stress) azimuth/trend in degrees. Range: 0-360, measured clockwise from North. Used as fixed value (if `use_shapefile=false`) or fallback value. **Required if `use_shapefile=false`** |
| `sigma1_plunge_degrees` | float/null | `null` | σ₁ plunge in degrees. Range: 0-90 (0=horizontal, 90=vertical). Used as fixed value (if `use_shapefile=false`) or fallback value. **Required if `use_shapefile=false`** |
| `sigma3_trend_degrees` | float/null | `null` | σ₃ (minimum principal stress) azimuth/trend in degrees. Range: 0-360. Used as fixed value (if `use_shapefile=false`) or fallback value. **Required if `use_shapefile=false`** |
| `sigma3_plunge_degrees` | float/null | `null` | σ₃ plunge in degrees. Range: 0-90. Used as fixed value (if `use_shapefile=false`) or fallback value. **Required if `use_shapefile=false`** |
| `stress_shape_ratio` | float/null | `null` | Stress shape ratio R = (σ₂-σ₃)/(σ₁-σ₃). Range: 0-1 (0=uniaxial extension, 0.5=σ₂ midway, 1=isotropic/uniaxial compression). Used as fixed value (if `use_shapefile=false`) or fallback value. **Required if `use_shapefile=false`** |

**Note**: σ₂ (intermediate principal stress) is automatically calculated from σ₁, σ₃, and the stress shape ratio.

**Spatially-Varying Stress Field**: When `use_shapefile` is `true` and `shapefile_path` is provided, the algorithm calculates the center coordinate (mean X, Y) of all hypocenters and queries the stress field values from the polygon containing this point.

### Mechanical Properties Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pore_pressure_mpa` | float | `0.0` | Pore fluid pressure in MPa. Range: 0-50 MPa typical. Reduces effective normal stress on faults |
| `friction_coefficient` | float | `0.75` | Coulomb friction coefficient μ. Range: 0.6-0.85 typical (Byerlee's law: ~0.6-1.0). Controls fault reactivation potential |

**Calculated Outputs**:
- Effective normal stress (Sn_eff)
- Shear stress (Tau)
- Rake (slip direction)
- Instability index (I)
- Slip tendency
- Dilation tendency

---

## Visualization

Visualization and export settings for analysis results.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable visualization module |

### Basic Visualization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generate_3d_model` | boolean | `true` | Generate interactive 3D Plotly HTML visualization of complete fault network |
| `generate_stereonet` | boolean | `true` | Generate stereonet (lower-hemisphere equal-area projection) showing fault plane orientations |

### Fault Surface Interpolation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_plane_interpolation` | boolean | `true` | Enable Poisson surface reconstruction to interpolate continuous fault surfaces from point clouds |
| `enable_mesh_stress` | boolean | `true` | Calculate stress parameters (Sn_eff, Tau, rake, sliptend, dilatend) for each mesh face. Requires stress_analysis enabled |
| `mesh_subdivisions` | integer | `2` | Number of mesh subdivision iterations (0-3). Each iteration quadruples face count. Loop subdivision maintains smoothness. Creates denser meshes than increasing poisson_depth |
| `poisson_depth` | integer | `2` | Poisson reconstruction octree depth. Range: 4-12. Higher = more detail but can introduce noise. **Recommended: 2-3 for smooth base, then use mesh_subdivisions for density** |
| `density_threshold` | float | `0.01` | Minimum density threshold for surface reconstruction. Range: 0.01-0.9. Lower = includes sparse regions, higher = only dense regions |
| `max_distance_factor` | float | `2.5` | Maximum distance factor for point-to-surface association. Range: 1.0-5.0. Higher = more permissive association |
| `min_fault_planes_for_interpolation` | integer | `10` | Minimum number of fault planes required in a cluster to attempt Poisson surface reconstruction. Clusters with fewer fault planes are skipped |
| `fault_plane_radius_interval_meters` | float | `10.0` | Spacing in meters between concentric circles when generating synthetic fault plane points. |
| `fault_plane_point_density_meters` | float | `10.0` | Target spacing in meters between points on fault plane circles. Increase to reduce computation time. |

### 3D Export Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `export_vtp` | boolean | `true` | Export all results to VTP (VTK PolyData) format for visualization in ParaView/Blender |
| `export_obj` | boolean | `false` | Export meshes as Wavefront OBJ files for 3D modeling software (MOVE, Blender, MeshLab, etc.) |

**Exported VTP Files**:
- `hypocenters.vtp` - All hypocenter points with attributes
- `enhanced_pointcloud.vtp` - Enhanced fault plane point cloud
- `rupture_planes_combined.vtp` - All rupture plane meshes
- `focal_planes_combined.vtp` - All focal mechanism planes (if focal constraints enabled)
- `interpolated_surfaces_*.vtp` - Interpolated fault surfaces (if interpolation enabled)


---

## Multi-Sequence Processing

Parameters for catalog segmentation and merging procedures that are required specifically for the multi-sequence workflow in addition to the single-sequence parameters.

See [Workflows Guide](workflows.md) for complete multi-sequence workflow documentation.

**Note**: Multi-sequence global settings (`parallel_processing`, `max_workers`, `save_individual_results`) are documented in [Global Settings](#global-settings) → Multi-Sequence Only Settings.

### Step 1: Load Input Data

Data loading in the multi-sequence workflow is configured in the `step_1_load_data` workflow step. This step loads and prepares the earthquake catalog for segmentation:

```json
"step_1_load_data": {
  "hypocenter_file": "data_examples/SECOS_20250305_HyFI.csv",
  "hypocenter_separator": ",",
  "focal_mechanism_file": "data_examples/SECOS_20250305_FM_HyFI.csv",
  "focal_mechanism_separator": ","
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `hypocenter_file` | string | **Yes** | Path to hypocenter catalog file (e.g., "data_examples/SECOS_20250305_HyFI.csv"). This is the full earthquake catalog to be segmented into sequences |
| `hypocenter_separator` | string | Yes | Column separator for hypocenter file. Options: `","` (CSV), `"\t"` (TSV), `";"` (semicolon). Must match the format of your input file |
| `focal_mechanism_file` | string or null | No | Path to focal mechanism catalog. Set to `null` if not available. Optional but recommended for improved fault plane selection via focal constraints |
| `focal_mechanism_separator` | string | No | Column separator for focal mechanism file. Options: `","`, `"\t"`, `";"`. Only needed if `focal_mechanism_file` is specified |

**Required Columns** (see [Input Data](#input-data) section for complete column descriptions):
- Hypocenter: `YR`, `MO`, `DY`, `HR`, `MI`, `SC`, `LAT`, `LON`, `Z`, `X`, `Y`, `EX`, `EY`, `EZ`, `ID`, `MAG` (or `ML`/`Mw`)
- Focal Mechanism (if used): `ID`, `Strike1`, `Dip1`, `Rake1`, `Strike2`, `Dip2`, `Rake2`, `A` (optional)

### Step 2: Segmentation Configuration

Multi-sequence segmentation is configured in the `step_2_catalog_segmentation` workflow step:

```json
"step_2_catalog_segmentation": {
  "enabled": true,
  "segmentation_steps": [
    {
      "step_name": "Class_A",
      "method": "dbscan",
      "features": ["spatial"],
      "cluster_dimension": "3d",
      "dbscan_eps": 350.0,
      "dbscan_min_samples": 10,
      "min_cluster_size": 20,
      "outlier_handling": "next_step"
    }
  ],
  "final_outlier_handling": "keep"
}
```

To achieve hierarchical multi-scale clustering, multiple segmentation steps can be defined.

#### Clustering Method

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `method` | string | `"dbscan"`, `"hdbscan"` | Clustering algorithm to use for segmentation |

**Method Comparison**:
- **DBSCAN**: Density-based spatial clustering with fixed distance threshold.
- **HDBSCAN**: Hierarchical DBSCAN that adapts to varying densities.

#### Feature Selection

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `features` | array | `["spatial"]`, `["spatial", "temporal"]` | Features to use for clustering. Options: `["spatial"]`  (Spatial clustering only using X, Y, Z coordinates. Groups events by location regardless of time), `["spatial", "temporal"]` (Spatiotemporal clustering using X, Y, Z, and time. Groups events that are close in both space and time) |

#### Cluster Dimension

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `cluster_dimension` | string | `"3d"`, `"2d"` | Spatial dimensionality for clustering. Options: `"3d"` (Full 3D spatial clustering using X, Y, Z coordinates. Standard for most fault analysis), `"2d"` (Horizontal clustering using only X, Y coordinates (ignores depth). Useful for analyzing lateral distribution or when depth uncertainty is high) |

#### DBSCAN Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `step_name` | string | Required | Label for this segmentation step (e.g., "Class_A", "Fine_Scale", "Primary"). Used in output directory names |
| `dbscan_eps` | float | Required | Maximum distance between events in the same cluster (meters). Smaller values = tighter clusters. Typical range: 100-1000 m |
| `dbscan_min_samples` | integer | `10` | Minimum number of events required to form a dense cluster core. Higher values = stricter clustering. Typical range: 5-20 |
| `min_cluster_size` | integer | `10` | Minimum number of events required to keep a cluster after segmentation. Clusters with fewer events are treated as outliers. Typical range: 10-50 |
| `outlier_handling` | string | `"next_step"` | How to handle events not assigned to any cluster. Options: `"next_step"` (pass to next segmentation step), `"keep"` (create outlier sequence), `"discard"` (remove from analysis) |

#### DBSCAN-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dbscan_metric` | string | `"euclidean"` | Distance metric for DBSCAN. Options: `"euclidean"` (standard Euclidean distance, **recommended**), `"manhattan"` (L1 distance), `"chebyshev"` (Chebyshev distance). Only used with `method: "dbscan"` |

#### HDBSCAN Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hdbscan_min_cluster_size` | integer | `15` | Minimum number of samples in a cluster. Range: 5-50. Lower values = more clusters but potential noise. Higher values = fewer, larger clusters. Only used with `method: "hdbscan"` |
| `hdbscan_min_samples` | integer or null | `null` | Minimum number of samples in a neighborhood for a point to be considered a core point. `null` = same as `hdbscan_min_cluster_size`. Use integer 3-20 for custom values. Only used with `method: "hdbscan"` |

#### Temporal Clustering Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temporal_window_days` | integer | `30` | Time window for grouping events (days). Range: 1-365. Events within this time window are grouped together regardless of spatial separation. Only used with `method: "temporal"` or when `"temporal"` is in `features` |

**Temporal Method**: Groups events by time windows, independent of location. Useful for identifying earthquake swarms and sequences that occur within specific time periods.

#### Spatial-Temporal Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spatial_weight` | float | `0.7` | Weight for spatial vs temporal features in combined analysis. Range: 0.0-1.0. Use 0.7 = 70% spatial/30% temporal (**recommended**), 0.5 = equal weight, 0.9 = mostly spatial. Only used with `method: "spatial_temporal"` or when both `"spatial"` and `"temporal"` are in `features` |

**Spatial-Temporal Method**: Combines spatial and temporal clustering using weighted features. Better for identifying space-time patterns in seismicity.

#### Outlier Handling (Per-Step)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `process_outliers` | boolean | `true` | Whether to process outlier events in subsequent segmentation steps. When `true`, events not assigned to clusters are passed to the next step. When `false`, outliers are discarded immediately |
| `outlier_handling` | string | `"next_step"` | Strategy for handling unassigned events in this step. Options: `"next_step"` (pass to next segmentation step, **recommended for hierarchical**), `"keep"` (create separate outlier sequence), `"discard"` (remove from analysis). Applies only to this specific step |

**Hierarchical Workflow Strategy**: 
- Use `outlier_handling: "next_step"` for all intermediate steps
- Use `outlier_handling: "keep"` or `"discard"` only for the final step
- This creates a hierarchical multi-scale segmentation where fine-scale clusters are found first, then broader clusters are identified from remaining events

#### Segmentation-Only Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `segmentation_only` | boolean | `false` | If `true`, skip fault analysis (step 3) and export VTP files directly after segmentation. Useful for testing and validating segmentation parameters before running full fault plane analysis. Exports individual VTP files per sequence and a combined VTP file with all sequences |

**Output Files**:
- `segmented_sequences_{step_name}_sequence_{ID}.vtp` - Individual VTP file per sequence
- `segmented_sequences_{step_name}_combined.vtp` - Combined VTP with all sequences (different colors per cluster)
- `segmentation_summary.json` - JSON file with statistics (cluster counts, event counts per sequence)

#### Final Outlier Handling

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `final_outlier_handling` | string | `"keep"`, `"discard"` | How to handle events not clustered after all segmentation steps are complete. `"keep"` creates a `Z_outliers` sequence directory with unclustered events, `"discard"` removes them from analysis and subsequent per-sequence processing |
| `max_outlier_ratio` | float | `0.3` | Maximum acceptable ratio of outlier events relative to total catalog. Range: 0.0-1.0 (0.3 = 30%). If exceeded, a warning is logged but processing continues. Used for validation/diagnostics |


### Step 3: Per-Sequence Analysis Configuration

After segmentation, HyFI core analysis is applied independently to each identified sequence in the `step_3_per_sequence_analysis` workflow step:

```json
"step_3_per_sequence_analysis": {
  "description": "Apply HyFI core analysis to each sequence identified in step_2",
  "fault_network": { ... },
  "model_validation": { ... },
  "auto_classification": { ... },
  "stress_analysis": { ... },
  "visualization": { ... }
}
```

This step applies all single-sequence analysis modules (Fault Network, Model Validation, Auto-Classification, Stress Analysis, and Visualization) to each sequence independently. The configuration is identical to the single-sequence workflow parameters documented in the respective sections:


### Step 4: Merge and Export Configuration

After per-sequence analysis, individual results are merged and exported in the `step_4_merge_and_export` workflow step:

```json
"step_4_merge_and_export": {
  "enabled": true,
  "description": "Merge VTP files and export combined results",
  "merge_vtp_files": true,
  "merged_output": {
    "merged_vtp_path": "./output/HyFI_Database/HyFI_Database_vtp/",
    "export_merged_csv": true,
    "export_summary_statistics": true
  }
}
```

#### VTP File Merging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `merge_vtp_files` | boolean | `true` | Merge individual VTP files from all sequences into combined VTP files. Creates one merged file for each analysis layer (hypocenters, rupture planes, focal planes, interpolated surfaces, slip vectors, etc.). All sequences are combined into single files with metadata tracking which sequence each point/geometry belongs to |

**Merged VTP Output Files**:
- `hypocenters_ALL.vtp` - All hypocenter points with cluster attribution
- `rupture_planes_ALL.vtp` - All rupture plane meshes from all sequences
- `faults_ALL.vtp` - Combined fault plane compilation
- `focal_planes_ALL.vtp` - All focal mechanism planes (if constraints enabled)
- `interpolated_surfaces_ALL.vtp` - All interpolated fault surfaces

#### Merged Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `merged_output` → `merged_vtp_path` | string | Required | Directory path for merged VTP files. Example: `"./output/HyFI_Database/HyFI_Database_vtp/"`. Merged VTP files are saved to this location for combined 3D visualization |
| `export_merged_csv` | boolean | `true` | Export enriched merged CSV catalog with all analysis results from all sequences. Output file: `HyFI_Database/merged_catalog_enriched.csv`. Combines hypocenters + fault plane parameters + auto-classification results. Useful for GIS import and further analysis |
| `export_summary_statistics` | boolean | `true` | Export summary statistics CSV with aggregate metrics per sequence. Output file: `HyFI_Database/summary_statistics.csv`. One row per sequence with event counts, magnitude range, spatial extent, and mean orientations |

#### Output Directory Structure

```
output_directory/
├── HyFI_Database/                               # Main export folder
│   ├── merged_catalog_enriched.csv              # All events + analysis results
│   ├── summary_statistics.csv                   # Per-sequence summary
│   ├── HyFI_database_metadata.csv               # Fault system metadata
│   ├── HyFI_database_focal_mechanisms.csv       # Focal mechanism compilation
│   └── HyFI_Database_vtp/                       # Merged VTP files
│       ├── hypocenters_merged.vtp
│       ├── rupture_planes_merged.vtp
│       ├── faults_merged.vtp
│       ├── focal_planes_merged.vtp
│       └── interpolated_surfaces_merged.vtp
├── A1/                                          # Individual sequence outputs
│   ├── 3D_model.html
│   ├── A1_data.csv
│   ├── vtp_export/
│   └── ...
├── A2/
├── B1/
└── Z_outliers/                                  # Unclustered events (if kept)
```

---

Happy fault imaging! 🎉
