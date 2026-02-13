# Visualization

**HyFI** provides comprehensive 3D interpolation, visualization and export capabilities. The visualization module processes clustered and classified fault data to generate interactive 3D models, stereonet projections, and exportable mesh formats suitable for advanced visualization software such as ParaView.

---

## Core Concepts

### Poisson Surface Reconstruction
To generate continuous meshes representing active faults, the system utilizes Poisson Surface Reconstruction (PSR) (Kazhdan et al., 2006). **HyFI** processes the discrete rupture segments of each `final_cluster_id` — identified during the auto-classification stage — and interpolates them into a coherent geometric surface using PSR.

Unlike local interpolation methods (such as Delaunay triangulation), PSR treats surface reconstruction as a global optimization problem. It expresses the reconstruction as a spatial Poisson problem, which effectively "wraps" a surface around the input data points based on their orientation.

- Input: Oriented point clouds or segments derived from fault clusters.
- Indicator Function: The algorithm solves for a scalar function $\chi$ (the indicator function) that is $1$ inside the fault volume and $0$ outside.
- The Poisson Equation: The surface is found by aligning the gradient of the indicator function with the vector field $\vec{V}$ created by the input normals:

    $\nabla \cdot \nabla \chi = \nabla \cdot \vec{V} \Longleftrightarrow \nabla^2 \chi = \nabla \cdot \vec{V}$

- Extraction: The final fault mesh is extracted as an isosurface (level set) using an adaptive octree.

---

## Computational Workflow

The visualization module executes the following sequential steps:

### Step 1: 3D Interactive Plotly Model Generation

Creates an interactive 3D model in html format that can be visualized in any web browser. Note that this is not very efficient and may be laggy for larger datasets. For visualization of large datasets, use the VTP-files in an external 3D software (e.g. ParaView).

### Step 2: Poisson Surface Reconstruction (Optional)

In the next step, the (enhanced) point clouds of each cluster (`final_cluster_id`) are interpolated into smooth triangulated meshes using PSR with the according octree depth (`poisson_depth`). With the density threshold (`density_threshold`), the meshes are trimmed to valid regions.

### Step 3: Stereonet Visualization

The poles of the rupture planes (normal vectors) are projected onto a lower-hemispher stereonet and returned as image.

### Step 4: Visualization Export

**HyFI** support the export of different formats to be visualized in third-party software (e.g. ParaView):

- VTP Export (VTK PolyData) → vtp_export/ directory
  - Ideal for visualization/post-processing in ParaView with full attributes attached.
- OBJ Export → obj_export/ directory
  - Ideal for visualization/post-processing in MOVE. Note that OBJ files do not have any attributes attached.

---

## Module Outputs

### Generated Files

```
{output_dir}/
├── 3D_model.html                    # Interactive Plotly 3D visualization
├── stereonet_by_cluster.png         # Equal-area stereonet projection
│
├── vtp_export/
│   ├── fault_1.vtp                  # Individual cluster mesh
│   ├── fault_2.vtp
│   ├── ...
│   ├── faults_compiled.vtp          # All meshes combined
│   └── hypocenters.vtp              # Point cloud of original events
│
├── obj_export/  (optional)
│   ├── fault_planes.obj             # Wavefront OBJ format
│   └── fault_planes.mtl             # Material definition
│
└── interpolated_faults_summary.csv   # Per-mesh metadata table
```

---

## References

- Kazhdan, M., Bolitho, M., & Hoppe, H. (2006, June). Poisson surface reconstruction. In Proceedings of the fourth Eurographics symposium on Geometry processing (Vol. 7, No. 4). 

---

Happy fault imaging! 🎉