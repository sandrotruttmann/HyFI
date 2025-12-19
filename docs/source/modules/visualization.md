# Visualization and Export

HyFI provides comprehensive visualization and export capabilities for fault network results.

## 3D Visualization

### Interactive HTML Model
The main output `3D_model.html` is an interactive Plotly visualization showing:
- Fault planes as 3D surfaces colored by attributes
- Earthquake hypocenters as point clouds
- Interpolated fault surfaces (if enabled)
- Interactive controls for rotating, zooming, filtering

Color schemes can represent:
- Fault orientation (strike/dip)
- Classification clusters
- Stress analysis results (slip tendency)
- Quality metrics (eigenvalue ratios, neighbor counts)

## Stereonet Projections

Lower-hemisphere stereonets visualize:
- Fault plane poles (normal vectors)
- Principal stress axes
- Orientation distributions by cluster
- Focal mechanism solutions (if available)

## Export Formats

### VTP Export (ParaView/VTK)
Directory: `vtp_export/`

VTP (VTK PolyData) files for advanced visualization in ParaView:
- `fault_planes.vtp`: Individual fault planes with attributes
- `hypocenters.vtp`: Point cloud with event metadata
- `interpolated_faults.vtp`: Gridded fault surfaces
- Supports all computed attributes and custom coloring

### OBJ Export (3D Modeling)
Directory: `obj_export/`

Standard OBJ format for 3D modeling software:
- `fault_planes.obj`: Triangulated fault surfaces
- Can be imported into Blender, MeshLab, CloudCompare
- Enabled via `export_obj: true` in configuration

### CSV Export
Directory: `csv_export/`

Machine-readable tabular data:
- Same content as main `HyFI_results.csv`
- Additional export options for specific subsets

### MOVE Export (Structural Geology)
Directory: `move_export/`

ASCII format for Petex MOVE software:
- Fault planes as oriented discs
- Ready for structural interpretation workflows

## Visualization Parameters

Key configuration options:
- `plot_bool`: Enable/disable 3D visualization
- `stereonet_bool`: Generate stereonets
- `export_vtp`: Export VTP files
- `export_obj`: Export OBJ files
- `export_csv`: Export CSV files
- `color_by`: Attribute for coloring fault planes

## Interpolated Fault Surfaces

Optional fault surface interpolation creates continuous surfaces:
- Triangulated mesh connecting nearby fault planes
- Honors orientation constraints from HyFI results
- Useful for visualizing large-scale fault geometry
- Parameters: `interpolate_faults`, `interpolation_distance_threshold`
