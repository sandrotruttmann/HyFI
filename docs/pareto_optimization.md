# Pareto Multi-Objective Optimization for Fault Network Parameters

## Overview

**Pareto multi-objective optimization** treats fault network parameter optimization as a true multi-objective problem, finding the **Pareto front** - the set of solutions where improving one objective requires degrading another. This approach is more rigorous than weighted single-objective optimization and reveals trade-offs explicitly.

## Why Pareto Optimization?

### Problems with Traditional Weighted Approach

The traditional approach combines objectives with fixed weights:
```
f = 0.6 × focal_fit + 0.25 × recovery_loss + 0.15 × quality_loss
```

**Limitations:**
1. **Arbitrary weights**: How do you know 0.6 is better than 0.5 or 0.7?
2. **Hidden trade-offs**: You get one solution, but don't see alternatives
3. **Context-dependent**: Optimal weights vary by catalog characteristics
4. **Not comparable**: Can't compare results across studies with different weights

### Pareto Approach Advantages

Instead of one weighted objective, Pareto optimization:
1. **Finds all optimal trade-offs**: Provides multiple solutions on the Pareto front
2. **No weight selection needed**: Weights are chosen *after* optimization based on priorities
3. **Reveals structure**: Shows relationships between objectives explicitly
4. **Publication-ready**: More rigorous for scientific publications
5. **Flexible**: Select solution based on your specific needs

## How It Works

### Objectives

#### With Focal Mechanism Data (3 objectives)
1. **Focal Mechanism Fit**: Angular difference between reconstructed and observed planes
2. **Plane Recovery Rate**: Fraction of events successfully assigned to planes
3. **Statistical Quality**: Lambda2/3 ratio from PCA (planarity measure)

#### Without Focal Mechanism Data (2 objectives)
1. **Plane Recovery Rate**: Maximize completeness
2. **Statistical Quality**: Maximize lambda2/3 ratio

### Pareto Dominance

A solution **A** dominates solution **B** if:
- A is better than or equal to B in *all* objectives, AND
- A is strictly better than B in *at least one* objective

The **Pareto front** contains all non-dominated solutions.

### Example

Consider three solutions:

| Solution | Focal Fit (°) | Recovery (%) | λ2/3 Ratio | Dominated? |
|----------|---------------|--------------|------------|------------|
| A        | 15            | 85           | 25         | No (Pareto optimal) |
| B        | 20            | 90           | 20         | No (Pareto optimal) |
| C        | 25            | 80           | 22         | Yes (dominated by A) |
| D        | 18            | 87           | 30         | No (Pareto optimal) |

- **A**: Best focal fit
- **B**: Best recovery (trades some focal fit for higher recovery)
- **D**: Best quality (excellent λ2/3, balanced others)
- **C**: Dominated (A is better in all respects)

**Pareto Front**: {A, B, D}

You can then choose:
- **A** if focal mechanism agreement is most important (e.g., for publications)
- **B** if you need maximum data coverage (e.g., exploratory analysis)
- **D** if you want the most statistically reliable subset

## Configuration

### Basic Setup

```json
{
  "auto_optimize_parameters": true,
  "optimization_method": "pareto",
  "optimization_n_trials": 100,
  "optimization_pareto_sampler": "nsga2",
  "optimization_pareto_population": 50,
  "optimization_n_startup_trials": 20,
  "optimization_plot_results": true
}
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimization_n_trials` | 100 | Total number of parameter combinations to evaluate |
| `optimization_pareto_sampler` | `"nsga2"` | Evolutionary algorithm: `"nsga2"`, `"nsga3"`, or `"random"` |
| `optimization_pareto_population` | 50 | Population size for evolutionary algorithm |
| `optimization_n_startup_trials` | 20 | Random trials before evolutionary optimization starts |
| `optimization_plot_results` | `true` | Generate Pareto front visualization |

### Sampler Selection

#### NSGA-II (Recommended)
- **Use for**: 2-3 objectives (typical case)
- **Advantages**: Well-established, robust, efficient
- **Algorithm**: Non-dominated Sorting Genetic Algorithm II
```json
"optimization_pareto_sampler": "nsga2"
```

#### NSGA-III
- **Use for**: >3 objectives (if you add custom objectives)
- **Advantages**: Better for many-objective problems
- **Note**: Usually not needed for standard fault network optimization
```json
"optimization_pareto_sampler": "nsga3"
```

#### Random
- **Use for**: Baseline comparison only
- **Advantages**: None (use for testing)
```json
"optimization_pareto_sampler": "random"
```

## Usage

### Python API

```python
from hyfi.utils.parameter_optimization import ParameterOptimizer
import pandas as pd

# Load data
data = pd.read_csv('earthquake_catalog.csv')

# Initialize optimizer
optimizer = ParameterOptimizer(
    data_input=data,
    focal_mechanisms=focal_data,  # Optional
    method='pareto'
)

# Run Pareto optimization
results = optimizer.optimize_pareto(
    n_trials=100,
    sampler='nsga2',
    population_size=50,
    plot_results=True,
    save_plot_path='pareto_results.png'
)

# Access Pareto front
pareto_front = results['pareto_front']
print(f"Found {len(pareto_front)} Pareto-optimal solutions")

# Get representative solutions
best_balanced = results['best_balanced']  # Best overall compromise
best_focal = results['best_focal']        # Best focal fit (if available)
best_recovery = results['best_recovery']  # Highest recovery rate
best_quality = results['best_quality']    # Best statistical quality

# Use the balanced solution (recommended starting point)
print(f"Recommended parameters:")
print(f"  Search radius: {best_balanced['r_nn']:.1f} m")
print(f"  Time window: {best_balanced['dt_nn']:.1f} hours")
print(f"  Recovery rate: {best_balanced['plane_recovery_rate']*100:.1f}%")
print(f"  Lambda2/3: {best_balanced['mean_lambda23_ratio']:.1f}")
```

### Command Line (via Config File)

```bash
# Run with Pareto optimization
hyfi config_pareto_example.json

# Results will include:
# - output_dir/parameter_optimization_pareto.png  (Pareto front visualization)
# - output_dir/parameter_optimization_report.json (detailed results)
```

## Interpreting Results

### Pareto Front Visualization

The generated plot shows:

1. **3D Pareto Front** (with focal data): Shows all three objectives simultaneously
   - X-axis: Focal fit (°) - lower is better
   - Y-axis: Recovery (%) - higher is better
   - Z-axis: λ2/3 ratio - higher is better

2. **2D Trade-off Plots**: Pairwise objective relationships
   - Focal vs Recovery: Shows if high recovery costs focal fit quality
   - Focal vs Quality: Shows relationship between fit and statistical quality
   - Recovery vs Quality: Always available, shows completeness vs. reliability

3. **Parameter Space**: Where Pareto solutions lie in (r_nn, dt_nn) space

4. **Representative Solutions**: Highlighted with different markers
   - ⭐ Red star: Best focal fit
   - 🔶 Orange square: Best balanced
   - 🔷 Green diamond: Best recovery
   - 🔺 Purple triangle: Best quality

### Selecting Your Solution

#### For Publications
→ Use **best_focal**: Maximizes agreement with focal mechanisms
- Most defensible scientifically
- May sacrifice some data coverage

#### For Exploratory Analysis
→ Use **best_recovery**: Maximizes number of fault planes
- Most complete picture of fault network
- May include some lower-quality planes

#### For High-Quality Subset
→ Use **best_quality**: Maximizes statistical confidence
- Most reliable planes (high λ2/3 ratios)
- May miss some valid but noisy planes

#### Recommended (Default)
→ Use **best_balanced**: Optimal compromise
- Good performance across all criteria
- Minimizes distance to ideal point in objective space
- Safe choice when priorities are unclear

### Advanced: Custom Selection

You can select from the full Pareto front based on specific criteria:

```python
# Example: Find solution with recovery > 80% AND focal fit < 20°
from hyfi.utils.parameter_optimization import ParameterOptimizer

pareto_front = results['pareto_front']

suitable_solutions = [
    sol for sol in pareto_front
    if sol['plane_recovery_rate'] > 0.80 and
       sol.get('mean_angular_diff', 100) < 20
]

if suitable_solutions:
    # Among suitable, select best quality
    best = max(suitable_solutions, key=lambda x: x['mean_lambda23_ratio'])
    print(f"Selected: r_nn={best['r_nn']:.1f}m, dt_nn={best['dt_nn']:.1f}h")
else:
    print("No solution meets your constraints - relax criteria")
```

## Computational Cost

### Timing Estimates

| Catalog Size | n_trials | Expected Time |
|-------------|----------|---------------|
| Small (<500 events) | 100 | 30-45 minutes |
| Medium (500-2000) | 100 | 45-75 minutes |
| Large (>2000) | 100 | 75-120 minutes |

**Note**: Pareto requires ~2x trials of single-objective for good coverage
- Single-objective: 50 trials typical
- Pareto: 100 trials recommended

### Efficiency Tips

1. **Start small**: Use `n_trials=50` for initial exploration
2. **Increase for final**: Use `n_trials=150-200` for publication-quality results
3. **Use focal data**: Narrows parameter space with 3rd objective
4. **Parallel evaluation**: Future feature (currently sequential)

## Comparison with Other Methods

| Method | Trials | Time | Solutions | Weights | Use Case |
|--------|--------|------|-----------|---------|----------|
| Grid Search | 625 | 2-4h | 1 (best) | Pre-defined | Exhaustive exploration |
| Optuna | 50 | 20-40m | 1 (best) | Pre-defined | Modern, good UI |
| **Pareto** | **100** | **45-90m** | **Many** | **Post-hoc** | **Publications, exploration** |

## Scientific Justification

### For Publications

Pareto optimization is more rigorous for scientific publications because:

1. **Reproducible**: Different studies can compare Pareto fronts directly
2. **Transparent**: Trade-offs are explicit, not hidden in weight choices
3. **Comprehensive**: Shows full solution space, not just one point
4. **Flexible**: Readers can see alternative solutions for different priorities

### Example Publication Language

> "We employed Pareto multi-objective optimization using NSGA-II to identify 
> optimal fault network parameters. The optimization balanced three objectives: 
> (1) focal mechanism agreement (mean angular difference), (2) plane recovery 
> rate, and (3) statistical quality (λ2/3 eigenvalue ratio). We identified 
> 23 Pareto-optimal solutions from 100 trials. For our analysis, we selected 
> the solution with best focal mechanism agreement (r_nn=127m, dt_nn=8200h), 
> achieving 18.3° mean angular difference with 82% plane recovery and λ2/3=31."

## Troubleshooting

### No Focal Mechanism Data

If you don't have focal mechanisms:
- Pareto still works with 2 objectives (recovery + quality)
- Results show recovery vs. quality trade-off
- Select based on your priority: coverage vs. reliability

### Pareto Front Too Large

If you get >50 solutions in Pareto front:
- This indicates objectives don't conflict strongly
- Most parameters give similar performance
- Safe to use best_balanced solution
- Consider tightening parameter ranges

### Pareto Front Too Small

If you get <5 solutions:
- Increase `n_trials` to explore more thoroughly
- Check if one objective dominates (e.g., focal fit perfect everywhere)
- Verify data quality and availability

### Long Runtime

To speed up:
- Reduce `n_trials` for testing (minimum ~50)
- Reduce `n_mc` in fault network (e.g., 500 instead of 1000)
- Use smaller catalog subset for initial optimization

## Advanced Topics

### Adding Custom Objectives

You can extend the code to add custom objectives:

```python
# In _calculate_combined_objective, add your metric
custom_metric = calculate_my_metric(data_output)

# In optimize_pareto, return it as additional objective
return focal_objective, recovery_loss, quality_loss, custom_metric_loss
```

Then use NSGA-III for >3 objectives:
```json
"optimization_pareto_sampler": "nsga3"
```

### Hypervolume Indicator

To quantitatively compare Pareto fronts:

```python
import optuna

# Get hypervolume (requires reference point)
study = results['optuna_study']
hypervolume = study.best_trials[0].system_attrs.get('hypervolume')
```

Larger hypervolume = better Pareto front coverage

### Knee Point Detection

To automatically find "best balanced" solution:

```python
from scipy.spatial.distance import cdist

def find_knee_point(pareto_front):
    """Find elbow/knee in Pareto front using distance metric."""
    # Normalize objectives to [0, 1]
    objectives = np.array([[s['focal_objective'], s['recovery_loss'], 
                           s['quality_loss']] for s in pareto_front])
    
    norm_obj = (objectives - objectives.min(axis=0)) / objectives.ptp(axis=0)
    
    # Find point closest to ideal (0, 0, 0)
    distances = np.sqrt((norm_obj**2).sum(axis=1))
    knee_idx = distances.argmin()
    
    return pareto_front[knee_idx]
```

This is essentially what `best_balanced` does automatically.

## References

1. **NSGA-II**: Deb et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II". IEEE Transactions on Evolutionary Computation.

2. **Optuna**: Akiba et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework". KDD.

3. **Pareto Optimization**: Miettinen (1998). "Nonlinear Multiobjective Optimization". Springer.

4. **Application**: Truttmann et al. (2023). "Hypocenter-based 3D Imaging of Active Faults". JGR: Solid Earth.

## Summary

**Key Takeaways:**
- Pareto finds all optimal trade-off solutions, not just one
- No need to guess weights beforehand
- More rigorous for publications
- Costs ~2x trials but reveals solution structure
- Use `best_balanced` as default, others for specific priorities
- NSGA-II sampler recommended for 2-3 objectives

**Quick Start:**
```json
{
  "optimization_method": "pareto",
  "optimization_n_trials": 100,
  "optimization_pareto_sampler": "nsga2",
  "optimization_plot_results": true
}
```

**Getting Help:**
- Check visualization to understand trade-offs
- Try different representative solutions
- Increase trials if Pareto front looks incomplete
- Use single-objective methods first for quick estimates
