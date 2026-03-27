# Automatic Classification

The automatic classification module groups rupture planes into clusters representing distinct fault sets. It identifies active faults with similar orientations and spatial proximity using both orientation-based clustering and optional spatial sub-clustering. For details, see the original **HyFI** publication (Truttmann et al., 2023).

---

## Core Concepts

### Fisher Distribution

The von Mises-Fisher distribution models directional data and is reflected by the concentration parameter **κ** that measures how clustered the vectors of the rupture planes are:
  - κ = 0: Uniform distribution (no preferred direction)
  - κ = 10: Moderate clustering
  - κ > 100: Very tight clustering

---

## Computational Workflow

The auto classification module follows a structured multi-stage pipeline:

### Step 1: Data Validation & Preparation

Check if the required fault plane data is available:

1. **Check required columns**: rupt_plane_azi, rupt_plane_dip
2. **Filter valid data**: Remove events with NaN fault plane parameters
3. **Output**: Subset DataFrame with only valid fault planes for clustering

**Quality gate**: Skip classification if fewer than 3 valid events

### Step 2: Enhanced Point Cloud Generation (Optional, RECOMMENDED)

If enabled (`use_fault_plane_points_for_clustering; true`), generate additional synthetic points ("hypocenters") along the calculated rupture planes to enhance the relatively sparse hypocenter catalog For each valid rupture plane:
   - Generate circular rupture plane using magnitude-based radius (Leonard 2014)
   - Create concentric rings at fixed intervals (`fault_plane_radius_interval_meters`: Spacing between concentric circles (typically 10-25 m))
   - Fill circles with points at specified density (`fault_plane_point_density_meters`: Point spacing along circles (typically 10-25 m))
   - Maintain mapping to source fault index

This generates a new hypocenter point cloud called "enhanced point cloud" with ~100-1000× more points than the original one while preserving full 3D rupture plane geometry. This can improve both spatial separation of nearby faults, as well as later interpolation.

### Step 3: Hemispherical Consistency

Ensures that all fault plane normal vectors point to same hemisphere to prevent ambiguous clustering.

1. **Reference vector**: Take first fault plane normal as reference
2. **Flip if needed**: For each other fault plane, check if angular distance > 90°
3. **Correction**: If flipped, negate all components (x, y, z) → (-x, -y, -z)
4. **Result**: All vectors on same hemisphere, consistent for clustering

### Step 4: Automatic Cluster Number Determination (Optional)

If enabled, this step determines the optimal number of orientation clusters automatically:

1. **Test range**: Evaluate k = 2 to `max_clusters`
2. **Scoring for each k**:
   - Apply clustering algorithm
   - Calculate Fisher concentration (κ) for each cluster
   - Compute the silhouette score or dispersion metric

3. **Selection strategy**:
   - Choose k with highest score
   - Penalize k > 3 to prefer simpler solutions
   - Only accept complex solution if >40% better than simple alternatives

### Step 5: Orientation Clustering

Group fault planes by similar orientations using one of the following directional clustering algorithms:

**A. Spherical K-Means (SKM)**
- Optimizes cluster centers on unit sphere
- Minimizes within-cluster angular distances
- Fast, deterministic (seeded)
- Good for roughly equal-sized clusters
- Convergence: max_iter=300, tol=1e-4 (default)

**B. Von Mises-Fisher Soft (VMF Soft)**
- Probabilistic mixture model on sphere
- Assigns soft membership probabilities
- Better for overlapping clusters
- Fisher concentration (κ) parameter auto-tuned
- Convergence: max_iter=300, tol=1e-6 (default)

**C. Von Mises-Fisher Hard (VMF Hard)**
- Deterministic variant of VMF
- Hard cluster assignments (one cluster per event)
- Better for distinct clusters
- Convergence: max_iter=300, tol=1e-6 (default)

This generates orientation cluster labels (0, 1, 2, ..., n_clusters-1) in the DataFrame column `orient_cluster`.

### Step 6: Spatial Sub-Clustering (Optional)

The rupture planes were grouped into sets of similar orientations in step 5. In this step, the algorithm now allows to separate nearby groups of rupture planes from the same orientation set into distinct fault segments. This can either be done on the original hypocenter locations alone, or including the enhanced point cloud if generated in step 2 using `use_fault_plane_points_for_clustering: true`

This generates spatial sub-cluster IDs within each orientation cluster in the column `spatial_cluster`, and the `final_cluster_id` that is the temporary global fault identifier (e.g., "0.1", "0.2", "1.0").

### Step 7: Post-Clustering Quality Control

This step checks the minimum cluster size to remove spurious small clusters that contain less events than `min_events_per_cluster` to prevent over-segmentation.

### Step 8: Results Mapping & Output

The clustering results are finally mapped backed to the full DataFrame `df_hyfi`.

---

## Main Outputs

The following columns are added to the `HyFI_results.csv` output file:

**Orientation Clustering**:
- **`orient_cluster`**: Fault system ID based on orientation

**Spatial Clustering** (if enabled):
- **`spatial_cluster`**: Sub-cluster within each orientation group
  
**Final Fault Identifiers**:
- **`final_cluster_id`**: Global fault system identifier
- **`final_cluster_id_local`**: Sequence-specific fault identifier

**Metadata**:
- **`sequence_label`**: Which sequence this event belongs to (multi-sequence context)
- **`segmentation_level`**: Hierarchical level (A, B, C, etc.)

---

## References

-  Truttmann, S., Diehl, T., & Herwegh, M. (2023). Hypocenter-based 3D imaging of active faults: Method and applications in the Southwestern Swiss Alps. Journal of Geophysical Research: Solid Earth, 128, e2023JB026352. https://doi.org/10.1029/2023JB026352 


---

Happy fault imaging! 🎉