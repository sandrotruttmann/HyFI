# Interpolation, Visualization and Export

**HyFI** provides comprehensive 3D interpolation, visualization and export capabilities for the processing results.


## Fault Surface Interpolation (Optional)

Optional interpolation creates continuous fault surfaces:
- Triangulated mesh connecting nearby rupture planes to reconstruct geometries of active faults
- Useful for visualizing large-scale datasets


## Visualization

### 3D Interactive HTML Model
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

### 3D Export (VTP)
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

---

Happy fault imaging! 🎉