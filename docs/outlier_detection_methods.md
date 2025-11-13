# Outlier Detection Methods: DBSCAN vs LOF vs Isolation Forest

This document explains the three outlier detection algorithms available in HyFI and provides guidance on when to use each method.

## Overview

HyFI supports three outlier detection algorithms:
1. **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise)
2. **LOF** (Local Outlier Factor)
3. **Isolation Forest** (IForest)

All methods protect events with valid focal mechanism data (A=1 or A=2) from being marked as outliers.

---

## DBSCAN: Density-Based Spatial Clustering

### How it works
- Groups events into clusters based on density
- Points in low-density regions are marked as outliers
- Requires two parameters: `eps` (maximum distance) and `min_samples` (minimum cluster size)
- Parameters are **automatically calculated** based on data distribution

### Strengths
- ✓ Finds clusters of arbitrary shape
- ✓ Handles varying densities reasonably well
- ✓ Intuitive concept (density-based clustering)
- ✓ Automatically determines number of clusters
- ✓ Well-tested in seismological applications

### Weaknesses
- ✗ Struggles with varying density clusters
- ✗ Sensitive to parameter selection
- ✗ May not detect outliers within clusters
- ✗ Binary decision (cluster or outlier)

### When to use DBSCAN
- When you expect distinct spatial clusters in your data
- When outliers are clearly separated from main clusters
- When you want to identify multiple fault structures
- For datasets with relatively uniform density

### Configuration example
```json
{
  "remove_outliers": true,
  "outlier_method": "DBSCAN"
}
```

---

## LOF: Local Outlier Factor

### How it works
- Measures local density deviation of each point relative to its neighbors
- Assigns an outlier score based on density comparison
- Points with substantially lower density than neighbors are outliers
- Uses `n_neighbors` to define locality and `contamination` for threshold

### Strengths
- ✓ Detects outliers in varying density regions
- ✓ Provides outlier scores (not just binary classification)
- ✓ Better for local anomalies
- ✓ Does not assume global density
- ✓ Excellent for isolated spurious events

### Weaknesses
- ✗ More computationally expensive
- ✗ Does not create meaningful clusters
- ✗ Sensitive to n_neighbors parameter
- ✗ May miss outliers in very sparse regions

### When to use LOF
- When you have varying density in your hypocenter distribution
- When outliers may be embedded within denser regions
- When you want continuous outlier scores (not just binary)
- For detecting isolated spurious events
- When DBSCAN identifies too many/too few outliers

### Configuration example
```json
{
  "remove_outliers": true,
  "outlier_method": "LOF",
  "lof_n_neighbors": 20,
  "lof_contamination": "auto"
}
```

### Parameters
- **lof_n_neighbors**: Number of neighbors to consider (default: auto-tuned based on dataset size)
  - Smaller values: more sensitive to local density variations
  - Larger values: smoother, considers broader neighborhood
  - Auto-tuning: `sqrt(N)` capped between 10 and 50
  
- **lof_contamination**: Expected proportion of outliers
  - `"auto"`: automatically determine threshold (recommended)
  - `0.05`: expect 5% outliers
  - `0.1`: expect 10% outliers

---

## Isolation Forest

### How it works
- Builds an ensemble of isolation trees (decision trees)
- Randomly selects features and split values to isolate observations
- Outliers are easier to isolate (require fewer splits/partitions)
- Assigns anomaly scores based on path length in trees
- Does not rely on distance or density metrics

### Strengths
- ✓ Fast and highly scalable
- ✓ Works well with high-dimensional data
- ✓ Does not assume specific data distribution
- ✓ Less sensitive to parameter tuning
- ✓ Effective for global outliers
- ✓ Handles large datasets efficiently
- ✓ Provides interpretable anomaly scores

### Weaknesses
- ✗ May miss local outliers in varying density regions
- ✗ Random nature requires setting random seed for reproducibility
- ✗ Less effective for clustered outliers
- ✗ Does not create meaningful clusters

### When to use Isolation Forest
- When you have large datasets (>1000 events)
- When computational efficiency is important
- When you want reproducible, stable results
- For detecting globally anomalous events
- When other methods are too slow or parameter-sensitive
- When you don't need clustering, just outlier detection

### Configuration example
```json
{
  "remove_outliers": true,
  "outlier_method": "IsolationForest",
  "if_n_estimators": 100,
  "if_contamination": "auto",
  "if_random_state": 42
}
```

### Parameters
- **if_n_estimators**: Number of isolation trees (default: 100)
  - Higher values: more accurate but slower
  - Typically 100-200 is sufficient
  - Increase for very large datasets or higher accuracy needs
  
- **if_contamination**: Expected proportion of outliers (default: 0.01 = 1%)
  - **0.01 (default)**: Very conservative, expects only ~1% outliers (recommended for seismic data)
  - **0.02**: Conservative, expects ~2% outliers
  - **0.05**: Moderate, expects ~5% outliers
  - **0.1**: More aggressive, expects ~10% outliers
  - **'auto'**: Automatic threshold (WARNING: can be too aggressive, may over-detect outliers)
  - For seismic data, explicit values (0.01-0.1) are recommended over 'auto'
  
- **if_max_samples**: Samples per tree (default: "auto")
  - `"auto"`: uses min(256, n_samples)
  - Smaller values: faster but less accurate
  - Larger values: more accurate but slower
  
- **if_random_state**: Random seed (default: 42)
  - Set for reproducible results
  - Different seeds may give slightly different results

---

## Comparison Table

| Aspect | DBSCAN | LOF | Isolation Forest |
|--------|--------|-----|------------------|
| **Primary Goal** | Clustering + outliers | Pure outlier detection | Pure outlier detection |
| **Output** | Cluster labels + outliers | Inliers/outliers + scores | Inliers/outliers + scores |
| **Method** | Density-based | Local density comparison | Tree-based isolation |
| **Computation** | Medium | Slow | Fast |
| **Scalability** | Medium | Poor | Excellent |
| **Parameters** | Auto-tuned (eps, min_samples) | n_neighbors, contamination | n_estimators, contamination |
| **Best for** | Distinct spatial clusters | Local anomalies, varying density | Large datasets, global outliers |
| **Handles varying density** | Moderate | Excellent | Good |
| **Reproducibility** | Deterministic | Deterministic | Needs random_state |

---

## Testing and Comparison

### Running the comparison script
```bash
python examples_archive/compare_outlier_methods.py data_examples/A18_data.csv
```

This will:
1. Apply DBSCAN, LOF, and Isolation Forest to your dataset
2. Show outlier counts and agreement statistics
3. Create visualizations comparing the results (6 plots including overlap analysis)
4. Save detailed comparison to CSV with outlier scores from all methods

### Command-line options
```bash
# Specify all parameters
python examples_archive/compare_outlier_methods.py data_examples/A18_data.csv \
    --lof-neighbors 25 \
    --lof-contamination 0.05 \
    --if-estimators 150 \
    --if-contamination 0.05

# Different separator
python examples_archive/compare_outlier_methods.py data_examples/custom.csv --sep ";"

# Just test Isolation Forest parameters
python examples_archive/compare_outlier_methods.py data_examples/A18_data.csv \
    --if-estimators 200 \
    --if-random-state 123
```

---

## Recommendations

### Start with Isolation Forest if:
- You have a large dataset (>500 events)
- You want fast, reproducible results
- You're looking for globally anomalous events
- You want a robust default method
- Computational efficiency matters

### Use DBSCAN if:
- You need to identify spatial clusters
- Your data has clear cluster structure
- You want backward compatibility with existing workflows
- You need cluster membership information

### Switch to LOF if:
- DBSCAN or IForest mark too many legitimate events as outliers
- You have varying density in your hypocenter distribution
- You need to detect local anomalies
- You have isolated spurious events within dense regions
- Dataset is small to medium (<1000 events)

### Best Practice
1. **Start with Isolation Forest** (fastest, most robust for most cases)
2. Review results and check if outliers make geological sense
3. If unsatisfied, run comparison script to test all three methods
4. Compare results visually and statistically
5. Choose method based on:
   - Dataset size and characteristics
   - Type of outliers you expect (global vs local)
   - Need for clustering vs pure outlier detection
   - Computational constraints
6. Manually inspect events with focal mechanisms that were protected

---

## Examples

### Example 1: Simple Isolation Forest (Recommended Default)
```json
{
  "hypo_file": "data_examples/A18_data.csv",
  "remove_outliers": true,
  "outlier_method": "IsolationForest"
}
```
This uses the default `if_contamination=0.01` (1%, very conservative).

### Example 2: Moderate Isolation Forest
```json
{
  "hypo_file": "data_examples/bedretto.csv",
  "remove_outliers": true,
  "outlier_method": "IsolationForest",
  "if_contamination": 0.05
}
```
This expects ~5% outliers (moderate setting).

### Example 3: More aggressive Isolation Forest
```json
{
  "hypo_file": "data_examples/noisy_data.csv",
  "remove_outliers": true,
  "outlier_method": "IsolationForest",
  "if_contamination": 0.1
}
```

### Example 4: LOF for varying density
```json
### Example 4: LOF for varying density
```json
{
  "hypo_file": "data_examples/complex_density.csv",
  "remove_outliers": true,
  "outlier_method": "LOF",
  "lof_n_neighbors": 15,
  "lof_contamination": 0.08
}
```

### Example 5: DBSCAN for clustering
```json
{
  "hypo_file": "data_examples/multi_fault.csv",
  "remove_outliers": true,
  "outlier_method": "DBSCAN"
}
```

### Example 6: Ultra-conservative outlier removal
```json
{
  "hypo_file": "data_examples/SECOS.csv",
  "remove_outliers": true,
  "outlier_method": "IsolationForest"
}
```
Default (0.01) is already very conservative, but you could use 0.005 for even less removal.
```

### Example 4: DBSCAN for clustering
```json
{
  "hypo_file": "data_examples/multi_fault.csv",
  "remove_outliers": true,
  "outlier_method": "DBSCAN"
}
```

### Example 5: Conservative outlier removal (all methods)
```json
{
  "hypo_file": "data_examples/SECOS.csv",
  "remove_outliers": true,
  "outlier_method": "IsolationForest",
  "if_contamination": 0.02
}
```

---

## Technical Details

### Isolation Forest Score Interpretation
- Anomaly score close to 1: Strong outlier (easy to isolate)
- Anomaly score around 0.5: Borderline (normal behavior)
- Anomaly score << 0.5: Clear inlier (hard to isolate)

In the implementation, scores are stored as **isolation_score**:
- More negative values = stronger outliers (easier to isolate)
- Values close to 0 or positive = normal inliers

**Path length interpretation:**
- Short path length (few splits needed) → Outlier
- Long path length (many splits needed) → Inlier

### LOF Score Interpretation
- LOF score ≈ 1: Similar density to neighbors (inlier)
- LOF score >> 1: Much lower density than neighbors (outlier)
- LOF score < 1: Higher density than neighbors (definitely inlier)

In the implementation, scores are stored as **negative outlier factor**:
- More negative values = stronger outliers
- Values close to -1 = normal inliers

### Algorithm Comparison by Dataset Characteristics

| Dataset Characteristic | DBSCAN | LOF | Isolation Forest |
|----------------------|--------|-----|------------------|
| **Small (<100 events)** | Good | Good | Good |
| **Medium (100-1000)** | Good | Moderate | Excellent |
| **Large (>1000)** | Moderate | Poor | Excellent |
| **Uniform density** | Excellent | Good | Good |
| **Varying density** | Moderate | Excellent | Good |
| **Clear clusters** | Excellent | Good | Moderate |
| **Dispersed/scattered** | Poor | Good | Excellent |
| **High-dimensional** | Moderate | Moderate | Excellent |

### Focal Mechanism Protection
All three methods automatically protect events with valid focal mechanism data (A=1 or A=2):
1. Outliers are first identified using the algorithm
2. Events with focal mechanisms that were marked as outliers are identified
3. These events are reassigned to the inlier group (cluster 0)
4. A log message reports which events were protected

This ensures that valuable focal mechanism constraints are not lost during outlier removal.

---

## Performance Tips

### For Large Datasets (>1000 events)
- **Recommended:** Isolation Forest (fastest, scales well)
- Increase `if_n_estimators` to 150-200 for better accuracy
- Consider setting `if_max_samples` to smaller value (e.g., 256) for speed

### For Small Datasets (<100 events)
- **Recommended:** DBSCAN or LOF (more appropriate for small samples)
- LOF: Use lower `n_neighbors` (10-15) for small datasets
- Be conservative with outlier removal (use lower contamination)

### For Datasets with Known Outlier Percentage
- **Always set `contamination` parameter explicitly** instead of "auto" for Isolation Forest
- "auto" can be too aggressive for seismic data
- Recommended values:
  - **0.01 (1%)**: Very conservative default, only most obvious outliers (RECOMMENDED)
  - **0.02 (2%)**: Conservative, good for high-quality data
  - **0.05 (5%)**: Moderate, balanced approach
  - **0.10 (10%)**: More aggressive, use if you know data is noisy
- Example: If you expect 1% outliers, use `if_contamination=0.01` (default)

### For Reproducible Results
- Isolation Forest: Always set `if_random_state` to a fixed value (e.g., 42)
- DBSCAN and LOF are deterministic (always give same results)

---

## References

- Ester, M., et al. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise." KDD-96 Proceedings.
- Breunig, M. M., et al. (2000). "LOF: identifying density-based local outliers." ACM SIGMOD Record, 29(2), 93-104.
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation forest." ICDM'08, 413-422.
- Scikit-learn documentation: https://scikit-learn.org/stable/modules/outlier_detection.html
