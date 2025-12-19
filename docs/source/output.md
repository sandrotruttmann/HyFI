# HyFI Output

After running HyFI, the output directory contains various files and subdirectories with analysis results, visualizations, and data exports. This guide explains each output component in detail.

## Output Directory Structure

```
output_directory/
├── HyFI_results.csv                              # Main results table
├── 3D_model.html                                 # Interactive 3D visualization
├── Stereoplot.pdf                                # Stereonet plot (if enabled)
├── execution_summary.json                        # Run statistics and metadata
├── active_plane_determination_summary.csv        # Focal mechanism analysis
├── active_plane_statistics.txt                   # Active plane statistics report
├── interpolated_faults_summary.csv               # Interpolated fault surfaces
├── parameter_optimization_report.json            # Optimization results (if enabled)
├── parameter_optimization_optuna.png             # Optimization plot (if enabled)
├── csv_export/                                   # CSV data files
│   ├── hypocenters.csv
│   └── enhanced_pointcloud.csv
├── vtp_export/                                   # VTK files for ParaView
│   ├── hypocenters.vtp
│   ├── enhanced_pointcloud.vtp
│   ├── rupture_planes.vtp
│   ├── slip_vectors.vtp
│   ├── focals_compiled.vtp
│   ├── fault_F1.vtp
│   └── faults_compiled_*.vtp
├── obj_export/                                   # OBJ 3D model files (if enabled)
└── move_export/                                  # MOVE format files (if enabled)
```

## Core Output Files

### 1. HyFI_results.csv

The main results table containing all events with computed fault plane parameters.

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
- `rupt_plane_azi` - Mean fault strike azimuth (degrees, 0-360°)
- `rupt_plane_dip` - Mean fault dip angle (degrees, 0-90°)
- `nor_x_mean`, `nor_y_mean`, `nor_z_mean` - Fault normal vector components
- `nr_fits` - Number of successfully fitted fault planes in Monte Carlo simulations
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

**Special Values:**
- `-999.0` - Parameter not computed or not applicable
- `-1` - Event excluded from analysis (e.g., outlier)
- `0` - Focal mechanism available but active plane unknown

### 2. 3D_model.html

Interactive 3D visualization built with Plotly. Open in a web browser to:
- Rotate, zoom, and pan the fault network
- Toggle visibility of different fault clusters
- View individual event properties on hover
- Examine fault plane orientations and spatial relationships
- Visualize stress parameters with color mapping (if enabled)

**Features:**
- Color-coded by fault cluster
- Fault plane patches shown as semi-transparent surfaces
- Hypocenters shown as scatter points
- Focal mechanisms displayed (if available)
- Interpolated fault surfaces (if enabled)

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

### 5. active_plane_statistics.txt

Human-readable summary report of active plane determination:

```
Total events with focal mechanisms: 27

DETERMINATION METHOD BREAKDOWN:
  Pre-specified active planes (A=1 or A=2): 18
  Newly determined preferred planes (geometric selection): 5
  Not determined: 1

Newly determined plane selection breakdown:
  - Plane 1 selected: 3 events
  - Plane 2 selected: 2 events
  - Mean angular difference: 30.2°
  - Median angular difference: 31.2°
  - Best match: 5.9°
  - Worst match: 47.6°

Pre-specified plane angular differences:
  - Mean angular difference: 20.9°
  - Median angular difference: 18.8°
```

### 6. interpolated_faults_summary.csv

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

### 7. parameter_optimization_report.json

Detailed results from automatic parameter optimization (if enabled).

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

VTK PolyData files for visualization in **ParaView** or **Blender**:

**Core Files:**
- **`hypocenters.vtp`** - All hypocenter points with attributes
- **`enhanced_pointcloud.vtp`** - Dense point cloud from rupture planes
- **`rupture_planes.vtp`** - Individual rupture plane meshes
- **`slip_vectors.vtp`** - Slip direction vectors (if stress analysis enabled)
- **`focals_compiled.vtp`** - Focal mechanism planes (if available)

**Fault Surface Files:**
- **`fault_F*.vtp`** - Individual interpolated fault surfaces (one per cluster)
- **`faults_compiled_*.vtp`** - Combined fault meshes with attributes

**VTK Attributes:**
Each VTP file contains point/cell data arrays:
- Event properties (magnitude, location, time)
- Fault orientation (strike, dip, rake)
- Stress parameters (Sn_eff, Tau, instability)
- Cluster assignments
- Quality metrics (kappa, epsilon, R/N)

**ParaView Workflow:**
1. Open VTP files in ParaView
2. Apply "Glyph" filter to visualize orientations
3. Use "Calculator" for custom expressions
4. Color by any attribute (magnitude, stress, cluster, etc.)

### obj_export/, move_export/

3D model files in various formats (if enabled in configuration):
- **OBJ** - Wavefront OBJ format (widely compatible, enabled via `export_obj: true`)
- **MOVE** - Petex MOVE format (structural geology software)

## Visualization Files

### Stereoplot.pdf

Lower-hemisphere equal-area stereonet projection showing:
- Fault pole orientations (colored by cluster)
- Density contours
- Mean orientation vectors
- Statistical distribution of fault attitudes

Useful for structural geology analysis and comparison with regional tectonics.

### parameter_optimization_optuna.png

Visualization of optimization process (if Optuna method used):
- Optimization history (objective value vs. trial)
- Parameter importance
- Parallel coordinate plot
- Slice plots showing parameter interactions
