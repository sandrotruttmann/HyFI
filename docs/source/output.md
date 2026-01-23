# HyFI Output

After running **HyFI**, the output directory contains various files and subdirectories with analysis results, visualizations, and data exports. This guide explains each output component in detail.

## Single-Sequence Output Structure

xxx TODO: clean-up output
- remove "A2_data.csv" saving
- remove "active_plane_statistics.txt" saving
- rename "parameter_optimization_optuna.png" to "parameter_optimization.png"
- check if csv_export is valid (works??)


```
output_directory/
├── HyFI_results.csv                              # Main results table, incorporating both hypocenter and focal data, complemented with HyFI processing results
├── 3D_model.html                                 # Interactive 3D visualization (can be opened in any web browser)
├── Stereoplot.pdf                                # Stereonet plot of rupture planes (if enabled) (poles to planes on lower hemisphere)
├── execution_summary.json                        # Run statistics and metadata
├── active_plane_determination_summary.csv        # Report of focal mechanism analysis (e.g., newly determined active planes)
├── interpolated_faults_summary.csv               # Interpolated fault surfaces
├── parameter_optimization_report.json            # Optimization results (if enabled)
├── parameter_optimization.png             # Optimization plot (if enabled)
├── csv_export/                                   # CSV data files
│   ├── hypocenters.csv
│   └── enhanced_pointcloud.csv
├── vtp_export/                                   # VTP files for ParaView
│   ├── hypocenters.vtp
│   ├── enhanced_pointcloud.vtp
│   ├── rupture_planes.vtp
│   ├── slip_vectors.vtp
│   ├── focals_compiled.vtp
│   ├── fault_F1.vtp
│   └── faults_compiled.vtp
└── obj_export/                                  # OBJ files (if enabled) (e.g. for visualization in MOVE)
```

### Multi-Sequence Output Structure

```
output_multi/
├── A1/  # First Class A sequence
│   ├── HyFI_results.csv
│   ├── 3D_model.html
│   ├── ...
│   └── obj_export/
├── A2/  # Second Class A sequence
│   └── ...
├── B1/  # First Class B sequence
│   └── ...
├── Z_outliers/  # Unassigned outlier events (if kept)
│   └── ...
└── HyFI_Database/
    ├── HyFI_database_metadata.csv              # xxx
    ├── HyFI_database_segmentation.csv          # xxx
    ├── HyFI_database_hypocenters.csv           # xxx
    ├── HyFI_database_focals.csv                # xxx
    └── HyFI_Database_vtp                       # Merged VTP files of all individual clusters
        ├── hypoceners_ALL.vtp
        ├── enhanced_pointcloud_ALL.vtp
        ├── rupture_planes_ALL.vtp
        ├── focals_ALL.vtp
        └── faults_ALL.vtp

```



## Core Output Files

### 1. HyFI_results.csv

The main results table containing all events with computed rupture plane parameters.

**Key Columns:**

#### Event Information
- `ID` - Event identifier
- `LAT`, `LON`, `DEPTH` - Geographic coordinates
- `X`, `Y`, `Z` - Local Cartesian coordinates (meters)
- `EX`, `EY`, `EZ` - Location uncertainties (meters)
- `Date` - Event timestamp
- `MAG` / `Mw` - Magnitude

#### Focal Mechanism Data (if available)
- `Strike1`, `Dip1`, `Rake1` - First nodal plane
- `Strike2`, `Dip2`, `Rake2` - Second nodal plane
- `A` - Active plane indicator (1, 2, or 0)
- `pref_foc` - Preferred focal mechanism plane (0.0, 1.0, 2.0, or -1)
- `epsilon` - Angular misfit between focal mechanism and fitted rupture plane

#### Computed Rupture Plane Parameters
- `rupt_plane_azi` - Mean rupture plane strike azimuth (degrees, 0-360°)
- `rupt_plane_dip` - Mean rupture plane dip angle (degrees, 0-90°)
- `nor_x_mean`, `nor_y_mean`, `nor_z_mean` - Rupture plane normal vector components
- `nr_fits` - Number of successfully fitted rupture planes in Monte Carlo simulations
- `kappa` - Concentration parameter (higher = more consistent orientations)
- `beta`, `lambda_2_3` - Eigenvalue ratios (geometric quality metrics)

#### Rupture Geometry
- `R` - Number of events within search radius
- `N` - Total number of neighbors considered
- `R/N` - Neighbor ratio
- `rupt_area` - Estimated rupture area (m²)
- `rupt_radius` - Estimated rupture radius (m)

#### Clustering Results
- `clust_labels` - Orientation cluster assignment
- `spatial_cluster` - Spatial sub-cluster assignment
- `final_cluster_id` - Combined cluster identifier (e.g., "F1", "F2")
- `orient_cluster` - Final classification label

#### Stress Analysis Results (if enabled)
- `Sn_eff` - Effective normal stress (MPa)
- `Tau` - Shear stress (MPa)
- `rake` - Calculated slip rake angle (degrees)
- `instab` - Instability index (Tau / Sn_eff)
- `sliptend` - Slip tendency
- `dilatend` - Dilation tendency


### 2. 3D_model.html

Interactive 3D visualization built with Plotly. Open in a web browser to:
- Rotate, zoom, and pan the fault network
- Toggle visibility of different fault clusters
- View individual event properties on hover
- Examine fault plane orientations and spatial relationships
- Visualize stress parameters with color mapping (if enabled)

### 3. execution_summary.json

Summary of the workflow execution with key statistics:

```json
{
  "workflow_execution_time": 104.92,        // Total runtime (seconds)
  "total_events": 545,                      // Input events
  "events_with_fault_planes": 338,         // Successfully processed
  "workflow_steps_completed": [...],        // Analysis modules run
  "execution_date": "2025-11-21T14:56:21", // Timestamp
  "focal_mechanisms_validated": 23,         // Validated focal mechanisms
  "stress_analysis_completed": 338,         // Events with stress calculations
  "fault_clusters": 1                       // Number of identified clusters
}
```

### 4. active_plane_determination_summary.csv

Details of focal mechanism validation and active plane selection for each event with focal mechanism data.

**Columns:**
- `ID`, `Date`, `X`, `Y`, `Z`, `MAG` - Event identifiers
- `Strike1`, `Dip1`, `Rake1` - First nodal plane
- `Strike2`, `Dip2`, `Rake2` - Second nodal plane
- `A` - Original active plane indicator
- `pref_foc` - Selected preferred plane (1.0 or 2.0)
- `epsilon` - Angular difference between focal mechanism and fitted plane (degrees)
- `plane_determination_method` - How the plane was selected:
  - `"Pre-specified (A=1 or A=2)"` - User/catalog provided active plane
  - `"Newly determined (A=0, geometric selection)"` - Algorithm selected plane
- `rupt_plane_azi`, `rupt_plane_dip` - Fitted rupture plane orientation
- `preferred_strike`, `preferred_dip`, `preferred_rake` - Selected nodal plane parameters

**Interpretation:**
- **epsilon < 30°** - Good agreement between focal mechanism and fitted plane
- **epsilon > 45°** - Poor agreement; investigate event location or focal mechanism quality


### 5. interpolated_faults_summary.csv

Summary of interpolated fault surfaces (if Poisson surface reconstruction is enabled):

**Columns:**
- `cluster_id` - Fault cluster identifier (e.g., "F1")
- `fault_idx` - Fault index within cluster
- `n_fault_planes` - Number of input rupture planes
- `n_input_points` - Number of points used for interpolation
- `mesh_vertices` - Number of vertices in interpolated mesh
- `mesh_faces` - Number of faces (triangles) in mesh
- `area_m2` - Total fault surface area (m²)
- `max_Mw` - Maximum moment magnitude for this area

**Usage:** Assess the extent and continuity of identified fault structures.

### 6. parameter_optimization_report.json

In addition to the visual optimization results, plotted in the *parameter_optimization.png* figure, detailed results from automatic parameter optimization are provided in JSON format (if enabled).

**Structure:**
```json
{
  "optimization_results": {
    "search_radius_meters": 100.5,            // Optimal spatial radius
    "search_time_window_hours": 11635.8,      // Optimal time window
    "method_used": "optuna",                  // Optimization algorithm
    "confidence_score": 0.83,                 // Solution quality (0-1)
    "calculated fault planes": 209,           // Planes computed
    "plane_recovery_rate": 0.38,              // Fraction of events with planes
    "expected_angular_difference": 2.8,       // Expected focal mismatch (degrees)
    "active_plane_accuracy": 0.84             // Correct plane selection rate
  },
  "catalog_statistics": {
    "n_events": 545,
    "spatial": {
      "nn_distance_stats": {...},            // Nearest-neighbor statistics
      "event_density_stats": {...}           // Spatial density metrics
    },
    "temporal": {...}                        // Temporal distribution
  },
  "optimization_trials": [...]               // All trial results
}
```

## Export Directories

### csv_export/

CSV files for external analysis:
- **`hypocenters.csv`** - All input hypocenter data
- **`enhanced_pointcloud.csv`** - Expanded point cloud from rupture plane surfaces (used for spatial clustering)

### vtp_export/

VTK PolyData files for visualization in ParaView:

**Core Files:**
- **`hypocenters.vtp`** - All hypocenter points with attributes
- **`enhanced_pointcloud.vtp`** - Dense point cloud from rupture planes
- **`rupture_planes.vtp`** - Individual rupture plane meshes
- **`slip_vectors.vtp`** - Slip direction vectors (if stress analysis enabled)
- **`focals_compiled.vtp`** - Focal mechanism planes (if available)

**Fault Surface Files:**
- **`fault_F*.vtp`** - Individual interpolated fault surfaces (one per cluster)
- **`faults_compiled_*.vtp`** - Combined fault meshes with attributes


### obj_export/

3D model files in various formats (if enabled in configuration):
- **OBJ** - Wavefront OBJ format (widely compatible, enabled via `export_obj: true`)

---

Happy fault imaging! 🎉