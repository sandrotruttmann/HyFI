# Configuration Template Documentation

This document provides detailed explanations for all parameters in `config_TEMPLATE.json` for single-sequence HyFI analysis.

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_directory` | string | `"./hyfi_output"` | Directory where all output files will be saved |
| `log_level` | string | `"INFO"` | Logging verbosity. Options: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"` |

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
| `optimization_method` | string | `"grid_search"` | Optimization algorithm. Options: `"heuristic"` (fast, seconds-minutes), `"grid_search"` (thorough, minutes-hours, **recommended**), `"bayesian"` (experimental), `"pareto"` (multi-objective) |
| `optimization_random_state` | integer | `42` | Random seed for optimization reproducibility |
| `optimization_n_trials` | integer | `50` | Number of trials for Bayesian/Pareto optimization. Range: 50-500. Higher = more thorough |
| `optimization_grid_points` | integer | `25` | Grid resolution for grid_search method. Range: 10-30. Higher = more thorough but slower. Typical: 20-25 |
| `optimization_plot_results` | boolean | `false` | Generate visualization plots of optimization results |
| `optimization_r_nn_range` | array or null | `[50, 1000]` | Search radius range for optimization `[min_meters, max_meters]`. Use `null` for automatic range determination |
| `optimization_dt_nn_range` | array or null | `[100, 50000]` | Time window range for optimization `[min_hours, max_hours]`. Use `null` for automatic range determination |

**Optimization Tips**:
- Small catalogs (<500 events): Use `"heuristic"` method
- Medium catalogs (500-5000 events): Use `"grid_search"` with `optimization_grid_points: 20`
- Large catalogs (>5000 events): Use `"grid_search"` with `optimization_grid_points: 25`

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_spatial_clustering` | boolean | `true` | Enable spatial sub-clustering within orientation clusters to identify separate fault structures |
| `spatial_clustering_method` | string | `"dbscan"` | Spatial clustering method. Options: `"dbscan"` (density-based, **recommended**), `"kmeans"` (k-means), `"hierarchical"` (agglomerative) |
| `min_events_per_cluster` | integer | `10` | Minimum number of events required to form a valid spatial cluster |

### Enhanced Point Cloud Clustering Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_fault_plane_points_for_clustering` | boolean | `true` | Use enhanced point cloud (fault plane surface points) instead of hypocenters for spatial clustering. Improves spatial resolution when enabled |
| `fault_plane_clustering_eps_factor` | float | `0.3` | DBSCAN `eps` parameter factor relative to median nearest neighbor distance. Range: 0.05-0.5. Lower = tighter clusters |
| `fault_plane_clustering_min_samples_factor` | float | `0.3` | DBSCAN `min_samples` as fraction of mean points per fault plane. Range: 0.05-0.3. Higher = more conservative clustering |

---

## Stress Analysis

Fault stress analysis and failure assessment using regional stress field.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable fault stress analysis module |

### Regional Stress Field Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sigma1_trend_degrees` | float | `301` | σ₁ (maximum principal stress) azimuth/trend in degrees. Range: 0-360, measured clockwise from North |
| `sigma1_plunge_degrees` | float | `23` | σ₁ plunge in degrees. Range: 0-90 (0=horizontal, 90=vertical) |
| `sigma3_trend_degrees` | float | `43` | σ₃ (minimum principal stress) azimuth/trend in degrees. Range: 0-360 |
| `sigma3_plunge_degrees` | float | `26` | σ₃ plunge in degrees. Range: 0-90 |
| `stress_shape_ratio` | float | `0.35` | Stress shape ratio R = (σ₂-σ₃)/(σ₁-σ₃). Range: 0-1 (0=uniaxial extension, 0.5=σ₂ midway, 1=isotropic/uniaxial compression) |

**Note**: σ₂ (intermediate principal stress) is automatically calculated from σ₁, σ₃, and the stress shape ratio.

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
| `poisson_depth` | integer | `3` | Poisson reconstruction octree depth. Range: 4-12. Higher = more detail but slower. **Recommended: 6-8** |
| `density_threshold` | float | `0.4` | Minimum density threshold for surface reconstruction. Range: 0.01-0.9. Lower = includes sparse regions, higher = only dense regions |
| `max_distance_factor` | float | `1.5` | Maximum distance factor for point-to-surface association. Range: 1.0-5.0. Higher = more permissive association |
| `spatial_clustering_method` | string | `"adaptive"` | Clustering method for interpolation. Options: `"kmeans"`, `"dbscan"`, `"adaptive"` (automatically chooses based on data), `"none"` (no clustering) |
| `min_events_per_cluster` | integer | `10` | Minimum events required per cluster for interpolation |

### ParaView/VTK Export Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `export_vtp` | boolean | `true` | Export all results to VTP (VTK PolyData) format for visualization in ParaView/Blender |

**Exported VTP Files**:
- `hypocenters.vtp` - All hypocenter points with attributes
- `enhanced_pointcloud.vtp` - Enhanced fault plane point cloud
- `rupture_planes_combined.vtp` - All rupture plane meshes
- `focal_planes_combined.vtp` - All focal mechanism planes (if focal constraints enabled)
- `interpolated_surfaces_*.vtp` - Interpolated fault surfaces (if interpolation enabled)

### Time Series Animation Parameters (!!! NOT WORKING CURRENTLY!)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `export_time_series` | boolean | `false` | Export time-resolved sequence of hypocenter clouds for temporal animation in ParaView |
| `time_step_hours` | integer | `24` | Time step for temporal binning in hours. Range: 1-720. Typical values: `24` (daily), `168` (weekly), `720` (monthly) |

**Output**: Creates `vtp_export/time_series/` directory with individual VTP files for each time step and a PVD collection file for animation.

### Geographic Export Parameters (!!! NOT WORKING CURRENTLY!)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `export_kml` | boolean | `false` | Export results to KML format for Google Earth visualization. **Experimental feature** |

---

## Usage Examples

### Minimal Configuration
```json
{
  "workflow_dag": {
    "input_data": {
      "hypocenter_file": "my_catalog.csv",
      "hypocenter_separator": ","
    },
    "fault_network": {
      "parameters": {
        "search_radius_meters": 100,
        "search_time_window_hours": 8760
      }
    }
  }
}
```

### Automatic Parameter Optimization
```json
{
  "workflow_dag": {
    "fault_network": {
      "parameters": {
        "search_radius_meters": "auto",
        "search_time_window_hours": "auto",
        "auto_optimize_parameters": true,
        "optimization_method": "grid_search"
      }
    }
  }
}
```

### With Focal Mechanism Constraints
```json
{
  "workflow_dag": {
    "input_data": {
      "hypocenter_file": "hypocenters.csv",
      "focal_mechanism_file": "focal_mechanisms.csv",
      "focal_mechanism_separator": ","
    },
    "fault_network": {
      "parameters": {
        "use_focal_constraints": true
      }
    }
  }
}
```

### Advanced Spatial Clustering
```json
{
  "workflow_dag": {
    "auto_classification": {
      "enabled": true,
      "parameters": {
        "enable_spatial_clustering": true,
        "use_fault_plane_points_for_clustering": true,
        "spatial_clustering_method": "dbscan",
        "fault_plane_clustering_eps_factor": 0.1
      }
    }
  }
}
```

