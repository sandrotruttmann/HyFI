# HyFI Output

After running **HyFI**, the output directory contains various files and subdirectories with analysis results, visualizations, and data exports. This guide explains each output component in detail.

## Single-Sequence Output Structure

```
output_directory/
├── execution_summary.json                        # Run statistics and metadata
├── HyFI_results.csv                              # Main results table, incorporating both hypocenter and focal data, complemented with HyFI processing results
├── parameter_optimization_report.json            # Optimization results (if enabled)
├── parameter_optimization.png                    # Optimization plot (if enabled)
├── active_plane_statistics.txt                   # Summary statistics of active nodal plane determination (model validation module)
├── active_plane_determination_summary.csv        # Report of focal mechanism analysis (e.g., newly determined active planes; model validation module)
├── interpolated_faults_summary.csv               # Summary of interpolated active fault surfaces
├── 3D_model.html                                 # Interactive 3D visualization (can be opened in any web browser)
├── Stereoplot.pdf                                # Stereonet plot of rupture planes (if enabled) (poles to planes on lower hemisphere)
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

## Multi-Sequence Output Structure

```
output_directory/
├── A1/  # First Class A sequence
│   ├── execution_summary.json
│   ├── HyFI_results.csv 
│   ├── ...
│   └── obj_export/
├── A2/  # Second Class A sequence
│   └── ...
├── B1/  # First Class B sequence
│   └── ...
├── Z_outliers/  # Unassigned outlier events (if kept)
│   └── ...
└── HyFI_Database/
    ├── HyFI_database_metadata.csv              # Main database with metadata of all identified active faults
    ├── HyFI_database_segmentation.csv          # Database of all segmented sequences
    ├── HyFI_database_hypocenters.csv           # Database of all hypocenters, populated with HyFI results
    ├── HyFI_database_focals.csv                # Database of all focal mechanisms, populated with HyFI results
    └── HyFI_Database_vtp                       # Merged VTP files of all individual clusters
        ├── hypocenters_ALL.vtp
        ├── enhanced_pointcloud_ALL.vtp
        ├── rupture_planes_ALL.vtp
        ├── focals_ALL.vtp
        └── faults_ALL.vtp

```

---

Happy fault imaging! 🎉