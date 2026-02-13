# Fault Network Reconstruction

The fault network reconstruction module is the core computational engine of **HyFI** that performs 3D fault imaging from earthquake hypocenter data. It combines nearest neighbor search, machine learning-based outlier detection, and principal component analysis (PCA) to reconstruct 3D rupture plane geometries from distributed seismic events. For details, see the original **HyFI** publication (Truttmann et al., 2023).

---

## Core Concepts

### Relocation Uncertainty Propagation

The module implements Monte Carlo simulation to propagate hypocenter location uncertainties through the fault plane estimation process. For each event:

- **Input**: Hypocenter location (X, Y, Z) with associated errors (EX, EY, EZ)
- **Process**: Random perturbations drawn from normal distributions with standard deviation equal to the reported location error
- **Iterations**: Multiple independent iterations (n_mc) generate ensemble of possible hypocenters
- **Result**: Probabilistic fault plane estimates accounting for full uncertainty

When `n_mc=1`, the original coordinates are used without perturbation, effectively disabling Monte Carlo simulation.

### Nearest Neighbor Search

For each master event, the algorithm finds neighboring earthquakes using scikit-learn (Pedregosa et al, 2011). Two criteria need be specified, which are the key parameters of **HyFI**:

- **Spatial**: All events within `search_radius_meters` (`r_nn`) (typically 100-1000 m)
- **Temporal**: All events within `search_time_window_hours` (`dt_nn`) of the master event (typically ±24 hours)

This forms the basis to fit local planes around each earthquake event using PCA.

### Principal Component Analysis (PCA) for Plane Fitting

The core fault plane estimation uses PCA on the point cloud of neighboring hypocenters (Shakarji, 1998):

**Mathematical basis**:
- Input: 3D coordinates of neighboring events (N × 3 matrix)
- Compute: Covariance matrix of the point cloud
- Decompose: Eigendecomposition yields eigenvalues (λ₁ ≥ λ₂ ≥ λ₃) and eigenvectors
- Plane normal: The eigenvector corresponding to the smallest eigenvalue (λ₃) gives the best-fit plane normal
- Quality metrics: Ratios of eigenvalues quantify how well-fit the plane is

**Physical interpretation**:
- Points strongly distributed around a plane → λ₃ much smaller than λ₁, λ₂ (good planarity)
- Points distributed along a line → λ₂ similar to λ₁, much larger than λ₃ (poor planarity)
- Points scattered isotropically → λ₁ ≈ λ₂ ≈ λ₃ (no clear structure)

### Quality Control Through Eigenvalue Ratios

The quality of each plane estimate is evaluated using two key metrics (Jones et al., 2016):

1. **Planarity (λ₂/λ₃ ratio)**: 
   - Compares intermediate to smallest eigenvalue
   - Threshold: typically λ₂/λ₃ > 3-5
   - High ratio → points tightly distributed around the plane
   - Low ratio → points scattered, no clear plane

2. **Collinearity (λ₂ threshold)**:
   - Absolute threshold based on mean location error squared
   - Prevents fitting planes when all points are nearly collinear
   - Rejects poor-quality point clouds with insufficient spatial extent

### Focal Mechanism Constraints (Optional)

When focal mechanism data is available, the module can optionally enhance the point cloud before PCA:

- **Synthetic Point Generation**: For each event with known active focal plane, generates synthetic points distributed on the expected nodal plane
- **Magnitude-Based Scaling**: Uses Leonard (2014) scaling relationships to estimate rupture radius from magnitude
- **Plane Parameterization**: Creates circular fault planes in 3D space with multiple concentric rings of points

---

## Computational Workflow

The module executes a well-defined pipeline of steps. Understanding this workflow is essential for appropriate parameterization, interpreting results, and troubleshooting issues:

### Step 1: Data Loading & Input Validation
In this step, data is loaded and the structure of the input dataset is validated.
- **Load hypocenter data**: Read catalog file as pandas DataFrame
- **Validate hypocenter catalog format**: Check for required columns (ID, YR, MO, DY, HR, MI, SC, X, Y, Z, EX, EY, EZ, MAG)
- **Parse datetime**: Construct datetime from year/month/day/hour/minute/second columns
- **Handle missing relocation uncertainties**: Replace missing location errors with default of 0 m (no error)
- **Output**: Single DataFrame `df_hyfi` that will be augmented through the pipeline

**Example columns at this stage**: ID, Date, X, Y, Z, EX, EY, EZ, MAG, YR, MO, DY, HR, MI, SC

### Step 2: Automatic Parameter Optimization (Optional)
Next, automatic parameter optimization can be used to define optimal nearest-neighbor parameters `r_nn` and `dt_nn`. For this, `auto_optimize_parameters` needs to be set to `true`:

1. **Initialize optimizer**: Create `ParameterOptimizer` instance with catalog and optional focal mechanisms
2. **Analyze catalog characteristics**:
   - Calculate spatial density (nearest neighbor distances, pairwise distances)
   - Calculate temporal distribution (inter-event times, sequence duration, event rates)
   - Analyze magnitude distribution (if available)
   - Estimate initial parameter ranges

3. **Select optimization method** (configurable):

   **Option A: Grid Search** (`optimization_method: 'grid_search'`)
   - Evaluate parameter combinations on regular grid
   - Default: 5×5 = 25 evaluations
   - Advantages: Comprehensive, interpretable, parallelizable
   - Execution time: ~5-60 seconds (depending on grid size and catalog)
   
   **Option B: Heuristic** (`optimization_method: 'heuristic'`)
   - Fast rule-based parameter estimation from catalog statistics
   - Execution time: ~1 second
   - Advantages: Minimal computation, good initial guess
   - Output: Single parameter pair (not a full scan)
   
   **Option C: Optuna (RECOMMENDED)** (`optimization_method: 'optuna'`)
   - State-of-the-art hyperparameter optimization framework
   - Intelligent sampling: Tree-structured Parzen Estimator (TPE) by default
   - Default: 50 trials with 10 startup random trials
   - Optional: Early stopping if no improvement for N trials
   - Advantages: Efficient exploration, handles complex objectives, produces beautiful visualizations
   - Execution time: ~30-120 seconds (depending on n_trials and catalog)
   
   **Option D: Pareto** (`optimization_method: 'pareto'`)
   - Multi-objective optimization (plane recovery vs. focal mechanism fit)
   - Default: 100 trials using NSGA-II algorithm
   - Advantages: Balances multiple criteria, generates Pareto frontier
   - Execution time: ~60-180 seconds
   - Output: Pareto-optimal solutions + single "best balanced" recommendation

4. **Objective function evaluation** (same for all methods):
   - For each parameter pair (`r_nn`, `dt_nn`):
     1. Run fault network reconstruction with those parameters.
     2. Count the number of recovered fault planes.
     3. If focal mechanisms are available: calculate angular differences with focal solutions.
     4. If focal mechanisms are available: compute fit-quality metrics (λ₂/λ₃ ratio).
     5. Combine the individual metrics into a single objective score.
   - Score composition:
     - **Plane recovery rate**: weight typically in the 0.5–0.9 range.
     - **Focal mechanism fit**: weight typically in the 0.1–0.5 range (used only when focal data exist).
   - Weights adapt automatically based on the number of matched focal mechanisms (adaptive weighting).

5. **Results processing**:
   - Extract best parameters (minimum objective score)
   - Generate parameter importance plots (if `plot_results: true`)
   - Generate summary CSV with all trial results

6. **Parameter application**:
   - Use optimized `r_nn` and `dt_nn` for subsequent fault plane fitting
   - Log optimization results and selected parameters
   - Store results in `optimization_summary.csv` (location: output directory)

### Step 3: Focal Mechanism Integration (Optional)
In this step, focal mechanism data is loaded and merged with the hypocenters to include additional knowledge from known nodal planes already in the fault network module (only if `use_focal_constraints: true`).

- **Load focal mechanism catalog**: If provided, read strike/dip and focal plane data
- **Merge by event ID**: Align focal mechanism data with hypocenters using ID matching
- **Add columns**: Strike1, Dip1, Rake1, Strike2, Dip2, Rake2, A (active plane indicator)
- **Validation**: Check data consistency and completeness

**Output columns added**: Strike1, Dip1, Rake1, Strike2, Dip2, Rake2, A (filled with NaN if no focal data)

### Step 4: Outlier Detection
Identify outliers in the hypocenter catalog (individual sequence) to clean up the catalog. This ensures that outliers in the sequence don't influence/disturbt the PCA fitting. Three different methods are available:

#### Option A: DBSCAN (Density-Based Spatial Clustering; RECOMMENDED)
- **Method**: Groups points by density, marks isolated points as outliers
- **Advantages**: Finds clusters of arbitrary shape, automatic cluster count
- **Parameters**: Auto-tuned based on k-distance graph (75th percentile distance)
- **Use Case**: Best for identifying spatially separated event clusters or regional anomalies

1. **Calculate k-distance graph**: Find distance to k-th nearest neighbor for each event
2. **Auto-tune eps parameter**: Use 75th percentile of k-distances × 1.5 for search radius
3. **Apply DBSCAN clustering**: Group events by spatial density (min_samples=5)
4. **Identify isolated events**: Mark events with cluster label -1 as outliers
5. **Protect focal mechanisms**: Events with valid focal mechanisms (A=1 or A=2) reassigned to largest cluster

**Output**: `clust_labels` column (-1 for outliers, 0+ for clusters)

#### Option B: Local Outlier Factor (LOF)
- **Method**: Compares local density of each point vs. its neighbors
- **Advantages**: Sensitive to local density variations, detects contextual outliers
- **Parameters**: n_neighbors (auto-tuned based on √N), contamination rate
- **Use Case**: Best for continuous density variations, detecting events isolated from their neighborhood

1. **Auto-tune n_neighbors**: Set to √(N) if not specified, capped at 10-50
2. **Calculate local density**: Compute LOF scores based on neighbor density ratios
3. **Set contamination threshold**: Expected outlier fraction (default='auto')
4. **Mark outliers**: Events with LOF score below threshold flagged as -1
5. **Protect focal mechanisms**: Reassign constrained events to inliers (cluster 0)

**Output**: `clust_labels` column (-1 for outliers, 0 for inliers), `lof_score` column

#### Option C: Isolation Forest
- **Method**: Recursively isolates points using random feature selection
- **Advantages**: Fast, scalable, works in high dimensions, no distance metrics
- **Parameters**: Conservative default contamination (5%), configurable estimator count
- **Use Case**: Best for high-dimensional problems and very large datasets

1. **Build ensemble**: Train 100 isolation trees (configurable)
2. **Set contamination**: Default 0.05 (5%, conservative) or user-specified
3. **Isolate anomalies**: Events requiring fewer splits to isolate → more anomalous
4. **Score events**: Negative scores indicate outliers
5. **Protect focal mechanisms**: Reassign constrained events to inliers (cluster 0)

**Output**: `clust_labels` column (-1 for outliers, 0 for inliers), `isolation_score` column

**If no outlier detection**: All events assigned `clust_labels=0`
**Focal Mechanism Protection**: Events with valid focal mechanisms (A=1 or A=2) are protected from outlier removal and reassigned to the largest cluster.


### Step 5: Hypocenter Perturbation (Monte Carlo Iterations)
This step creates a dataset of shifted hypocenter that are perturbed within their relocation errors (EX, EY, EZ). For each iteration k = 1 to n_mc:

**For n_mc = 1**: Use original coordinates without perturbation (fast, uncertainty ignored)

**For n_mc > 1**: Create perturbed copies
1. **Random sampling**: For each coordinate (X, Y, Z)
2. **Normal distribution**: Draw from N(location, error_std)
3. **Standard deviation**: Set to reported location error
4. **Repeat**: Generate n_mc independent realizations
5. **Output arrays**: Shape = (n_events, n_mc)

**Example**: Event with X=2683500±10m generates n_mc different X values around 2683500

**Mathematical basis**: 
- Error interpretation: Reported error ≈ 1-sigma uncertainty (standard deviation)
- Normal assumption: Location uncertainty follows normal distribution

### Step 6: Neighbor Search & Point Cloud Creation
In this step, for each iteration and each earthquake event i, the nearest neighbors are extracted:

1. **Spatial radius search**:
   - Find all events j within `search_radius_meters` of event i
   - Use k-d tree for efficient nearest-neighbor lookup
   - Return indices and distances

2. **Temporal window filtering**:
   - Calculate time difference between event i and each neighbor j
   - Remove neighbors with |date_i - date_j| > `search_time_window_hours`
   - Keep only temporally coherent neighbors

3. **Point cloud assembly**:
   - Collect 3D coordinates of remaining neighbors
   - Optional: Add synthetic points from focal mechanism constraints
   - Result: N×3 matrix of neighbor coordinates

4. **Quality check**:
   - Require minimum N ≥ `min_neighbor_count` (typically 5)
   - Skip plane fitting if threshold not met (return NaN)

**Output**: List of point clouds, one per event

### Step 7: Principal Component Analysis (PCA) - Plane Fitting
For each event i and the extracted neighbors from step 6, a rupture plane is calculated using PCA:

1. **Covariance matrix**:
   - Input: N×3 matrix of neighbor coordinates
   - Center data to origin
   - Compute: Cov = (X^T X) / (N-1)

2. **Eigendecomposition**:
   - Calculate eigenvalues: λ₁ ≥ λ₂ ≥ λ₃ ≥ 0
   - Calculate eigenvectors: e₁, e₂, e₃ (orthonormal)

3. **Plane extraction**:
   - **Normal vector**: n = e₃ (eigenvector of smallest eigenvalue)
   - **In-plane vector 1**: v₁ = e₁ (direction of maximum variance)
   - **In-plane vector 2**: v₂ = e₂ (direction of secondary variance)

4. **Orientation convention**:
   - Enforce lower-hemisphere convention: z-component of n < 0
   - Flip if needed: n → -n

5. **Quality metrics**:
   - **Planarity ratio**: λ₂/λ₃ (higher = more planar)
   - **Collinearity check**: λ₂ > threshold (based on location error)
   - **Eigenvalue spread**: λ₁, λ₂, λ₃ (characterize point distribution)

**Output per event**: [n_x, n_y, n_z, v1_x, v1_y, v1_z, v2_x, v2_y, v2_z, λ₁, λ₂, λ₃, λ₂/λ₃, total_variance]

### Step 8: Repeat for All Monte Carlo Iterations
Steps 5-7 repeat n_mc times, producing:
- `plane_fit_list`: List of n_mc arrays, each with shape (n_events, 14)
- Ensemble of plane estimates representing uncertainty

### Step 9: Spherical Statistics & Aggregation
For each event i, all n_mc rupture plane estimates are aggregated and evaluated using directional statistics:

1. **Extract vectors**:
   - Collect all n normal vectors from n_mc iterations
   - Remove NaN values (failed fits)
   - Check success rate: require >80% valid fits for statistics

2. **Hemispherical registration**:
   - Reference first valid vector
   - Flip other vectors if angular distance > 90° (ensure same hemisphere)
   - Ensures consistent orientation across iterations

3. **Vector statistics**:
   - **Resultant vector**: R = Σn_i (sum of all normal vectors)
   - **Resultant length**: |R|
   - **Resultant norm**: R/N (length normalized by count)
   
4. **Kent distribution fitting** (if n_mc > 1):
   - Fit Kent (spherical bivariate normal) distribution
   - Extract concentration parameter: κ (how clustered are the vectors)
   - Extract ovalness parameter: β (how elliptical is the distribution)
   - Physical meaning: κ > 10 indicates well-constrained orientations

5. **Mean orientation**:
   - Calculate mean vector: m = R / |R|
   - Convert to azimuth and dip angles
   - These are the final fault plane parameters

6. **Quality summary**:
   - **nr_fits**: Fraction of successful plane fits (0-1)
   - **R, N, R/N**: Resultant statistics
   - **kappa, beta**: Kent distribution parameters
   - **mean_λ₂/λ₃**: Average planarity across iterations

**Output per event**:
- nor_x_mean, nor_y_mean, nor_z_mean: Mean normal vector components
- nr_fits: Fraction of successful fits
- R, N, R/N: Resultant statistics
- kappa, beta: Directional distribution parameters
- lambda_2_3: Mean planarity ratio

### Step 10: Normal Vector to Fault Parameters
This converts the mean normal vector (n_x, n_y, n_z) to geological azimuth/dip angles:

**Output columns**:
- `rupt_plane_azi`: Dip direction azimuth (0-360°)
- `rupt_plane_dip`: Dip angle (0-90°)

### Step 11: Magnitude-Based Rupture Scaling
In this step, the earthquake magnitude is translated into expected rupture size, assuming cirulcar ruptures. The module includes multiple empirical rupture scaling relationships:
- **Leonard (2014)**: 
- **Wells & Coppersmith (1994)**
- **Thingbaijam (2017)**

These convert moment magnitude (Mw) to rupture area and radius, enabling synthetic fault plane generation.

1. **Input magnitude**: MAG column (ML or Mw)
2. **Convert ML → Mw**: If needed (Goertz-Allmann et al. 2011)
3. **Scaling relationship**: Apply Leonard (2014) for stable continental regions
   - Mw = 4.18 + log(A)  where A = rupture area in km²
4. **Calculate rupture area**: A = 10^(Mw - 4.18) km²
5. **Calculate rupture radius**: r = √(A/π) meters
   - Assumes circular rupture plane

**Output columns**:
- `Mw`: Moment magnitude (converted if input was ML)
- `rupt_area`: Rupture area in km²
- `rupt_radius`: Rupture radius in meters

### Step 12: Final Output
This returns the DataFrame `df_hyfi` populated with all computed parameters from the fault network module:

**Original columns**: ID, Date, X, Y, Z, EX, EY, EZ, MAG, YR, MO, DY, HR, MI, SC

**Added columns from outlier detection**:
- clust_labels: Cluster assignment (-1 for outliers)

**Added columns from focal mechanism merge** (if available):
- Strike1, Dip1, Rake1, Strike2, Dip2, Rake2, A

**Added columns from plane fitting**:
- nor_x_mean, nor_y_mean, nor_z_mean: Mean normal vector
- nr_fits: Fraction of successful MC fits
- R, N, R/N: Resultant statistics
- kappa, beta: Kent distribution parameters
- lambda_2_3: Mean planarity ratio
- rupt_plane_azi, rupt_plane_dip: Fault plane orientation

**Added columns from magnitude scaling**:
- Mw: Moment magnitude
- rupt_area: Rupture area (km²)
- rupt_radius: Rupture radius (m)

---

## References

- Goertz-Allmann, B. P., Edwards, B., Bethmann, F., Deichmann, N., Clinton, J., Fäh, D., & Giardini, D. (2011). A new empirical magnitude scaling relation for Switzerland. Bulletin of the Seismological Society of America, 101(6), 3088–3095. https://doi.org/10.1785/0120100291 

- Jones, R. R., Pearce, M. A., Jacquemyn, C., & Watson, F. E. (2016). Robust best-fit planes from geospatial data. Geosphere, 12(1), 196–202. https://doi.org/10.1130/GES01247.1 

- Leonard, M. (2014). Self-consistent earthquake fault-scaling relations: Update and extension to stable continental strike-slip faults. Bulletin of the Seismological Society of America, 104(6), 2953-2965.

- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., & Thirion, B. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://doi.org/10.1289/EHP4713 

- Shakarji, C. M. (1998). Least-squares fitting algorithms of the NIST algorithm testing system. Journal of Research of the National Institute of Standards and Technology, 103(6), 633–641. https://doi.org/10.6028/jres.103.043 

- Thingbaijam, K, Martin Mai, P., Goda, K. (2017). New Empirical Earthquake Source‐Scaling Laws. Bulletin of the Seismological Society of America 2017;; 107 (5): 2225–2246. doi: https://doi.org/10.1785/0120170017

- Wells, D. L., & Coppersmith, K. J. (1994). New empical relationship between magnitude, rupture length, rupture width, rupture area, and surface displacement. Bulletin of the Seismological Society of America, 84(4), 974–1002. 

-  Truttmann, S., Diehl, T., & Herwegh, M. (2023). Hypocenter-based 3D imaging of active faults: Method and applications in the Southwestern Swiss Alps. Journal of Geophysical Research: Solid Earth, 128, e2023JB026352. https://doi.org/10.1029/2023JB026352 


---

Happy fault imaging! 🎉