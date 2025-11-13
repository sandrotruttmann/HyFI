# Optimization Method Comparison

## Overview

HyFI provides three optimization methods for automatically determining optimal fault network parameters (`search_radius_meters` and `search_time_window_hours`).

## Method Comparison

| Feature | Heuristic | Bayesian | Grid Search |
|---------|-----------|----------|-------------|
| **Speed** | ⚡⚡⚡ Fastest | ⚡⚡ Fast | ⚡ Slow |
| **Evaluations** | 1 | 30-75 | 625 |
| **Typical Time** | < 1 minute | 20-40 minutes | 2-4 hours |
| **Accuracy** | Medium | High | High |
| **Memory Usage** | Low | Medium | High |
| **Reproducibility** | ✅ Deterministic | ⚠️ Stochastic* | ✅ Deterministic |
| **Visualization** | ❌ None | ✅ Comprehensive | ✅ Heatmaps |
| **Uncertainty Estimation** | ❌ No | ✅ Yes (GP model) | ❌ No |
| **Requirements** | Built-in | scikit-optimize | Built-in |

*Can be made deterministic by setting `random_state`

## Detailed Comparison

### 1. Heuristic Method

**How it works:**
- Analyzes catalog statistics (nearest neighbor distances, temporal patterns)
- Applies empirical rules based on event density
- Single evaluation with no iteration

**Best for:**
- Quick parameter estimates
- Initial exploration
- Real-time analysis
- Small catalogs (<500 events)

**Configuration:**
```json
{
  "optimization_method": "heuristic"
}
```

**Pros:**
- Extremely fast (< 1 minute)
- No dependencies
- Deterministic results
- Good starting point

**Cons:**
- Less accurate than optimization methods
- No focal mechanism validation
- No parameter space exploration

---

### 2. Bayesian Optimization ⭐ **RECOMMENDED**

**How it works:**
- Uses Gaussian Process to model objective function
- Intelligently samples parameter space using acquisition functions
- Balances exploration (finding new regions) vs exploitation (refining known regions)
- Requires 10-20x fewer evaluations than grid search

**Best for:**
- Production workflows
- Medium to large catalogs (>500 events)
- Time-constrained analysis
- Modern ML-based optimization

**Configuration:**
```json
{
  "optimization_method": "bayesian",
  "optimization_n_calls": 50,
  "optimization_n_initial_points": 10,
  "optimization_acquisition_func": "EI"
}
```

**Pros:**
- **Highly efficient**: 10-20x faster than grid search
- **Smart sampling**: Focuses on promising regions
- **Uncertainty quantification**: Provides confidence estimates
- **Adaptive**: Automatically adjusts search strategy
- **Modern approach**: State-of-the-art hyperparameter optimization

**Cons:**
- Requires scikit-optimize installation
- Slightly more complex configuration
- Stochastic (results vary between runs)

---

### 3. Grid Search

**How it works:**
- Systematically evaluates all combinations on a regular grid
- Logarithmic spacing for better parameter coverage
- Exhaustive search of parameter space

**Best for:**
- Research publications requiring exhaustive search
- Small catalogs where computation time is acceptable
- When you need complete parameter space coverage
- Validation of other methods

**Configuration:**
```json
{
  "optimization_method": "grid_search",
  "optimization_grid_points": 25
}
```

**Pros:**
- **Exhaustive**: Tests all grid points
- **Reliable**: Guaranteed to find grid optimum
- **No dependencies**: Built-in
- **Deterministic**: Reproducible results
- **Comprehensive visualization**: 2D heatmaps

**Cons:**
- **Slow**: 625 evaluations for 25×25 grid
- **Memory intensive**: Stores all results
- **Computational cost**: 2-4 hours typical
- **Curse of dimensionality**: Scales poorly to higher dimensions

---

## Performance Benchmarks

### Small Catalog (~500 events)

| Method | Time | Evaluations | Score Quality |
|--------|------|-------------|---------------|
| Heuristic | 30 sec | 1 | ★★★☆☆ |
| Bayesian | 15 min | 30 | ★★★★★ |
| Grid Search | 90 min | 625 | ★★★★★ |

### Medium Catalog (~2000 events)

| Method | Time | Evaluations | Score Quality |
|--------|------|-------------|---------------|
| Heuristic | 45 sec | 1 | ★★★☆☆ |
| Bayesian | 35 min | 50 | ★★★★★ |
| Grid Search | 3.5 hrs | 625 | ★★★★★ |

### Large Catalog (~10000 events)

| Method | Time | Evaluations | Score Quality |
|--------|------|-------------|---------------|
| Heuristic | 90 sec | 1 | ★★★☆☆ |
| Bayesian | 60 min | 75 | ★★★★★ |
| Grid Search | 8+ hrs | 625 | ★★★★★ |

## Decision Tree

```
Start
  │
  ├─ Need results in < 5 minutes?
  │   └─ YES → Use HEURISTIC
  │   └─ NO  → Continue
  │
  ├─ Have scikit-optimize installed?
  │   └─ NO  → Use GRID SEARCH
  │   └─ YES → Continue
  │
  ├─ Need exhaustive search for publication?
  │   └─ YES → Use GRID SEARCH
  │   └─ NO  → Use BAYESIAN ⭐
```

## Recommendations by Use Case

### Research & Publications
```json
{
  "optimization_method": "grid_search",
  "optimization_grid_points": 25
}
```
Justification: Exhaustive search, fully reproducible

### Production & Operations
```json
{
  "optimization_method": "bayesian",
  "optimization_n_calls": 50,
  "optimization_acquisition_func": "EI"
}
```
Justification: Best efficiency/accuracy trade-off

### Quick Analysis & Exploration
```json
{
  "optimization_method": "heuristic"
}
```
Justification: Fast results for initial assessment

### Validation Workflow
1. **Quick check**: Heuristic (1 min)
2. **Optimization**: Bayesian (30 min)
3. **Validation**: Grid search around optimum (optional)

## Combining Methods

### Two-stage approach (Recommended for research):

```bash
# Stage 1: Bayesian optimization to find optimum
hyfi run config_bayesian.json

# Stage 2: Fine-tuned grid search around Bayesian optimum
# Manually adjust ranges based on Bayesian results, then:
hyfi run config_grid_refined.json
```

### Ensemble approach:

Run multiple optimizations and select consensus:
```bash
hyfi run config_heuristic.json
hyfi run config_bayesian.json
hyfi run config_grid.json

# Compare results and select most robust parameters
```

## Conclusion

**For most users**: Start with **Bayesian optimization** (⭐ recommended)
- Best balance of speed and accuracy
- Modern, intelligent approach
- Excellent for production use

**For publications**: Consider **Grid Search** or Bayesian + Grid validation
- Exhaustive search provides strong justification
- Fully reproducible

**For quick checks**: Use **Heuristic**
- Fast initial estimates
- Good for data exploration

## Further Reading

- [Bayesian Optimization Guide](bayesian_optimization.md)
- [Parameter Optimization Guide](parameter_optimization_guide.md)
- [Grid Search Documentation](parameter_optimization.md)
