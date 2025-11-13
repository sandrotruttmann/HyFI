# Bayesian Optimization for Fault Network Parameters

## Overview

Bayesian optimization is now available as an advanced parameter optimization method for the HyFI fault imaging workflow. This method uses Gaussian Process regression to efficiently explore the parameter space and find optimal values for `search_radius_meters` and `search_time_window_hours`.

## Why Bayesian Optimization?

### Advantages over Grid Search

1. **Sample Efficiency**: Requires 10-20x fewer evaluations than grid search
2. **Intelligent Exploration**: Uses probabilistic models to focus on promising regions
3. **Uncertainty Quantification**: Provides confidence estimates for parameter selection
4. **Adaptive Strategy**: Balances exploration vs exploitation automatically

### Performance Comparison

| Method | Evaluations | Time (typical) | Accuracy |
|--------|-------------|----------------|----------|
| Grid Search (25²) | 625 | 2-4 hours | High |
| **Bayesian** | **50** | **20-40 min** | **High** |
| Heuristic | 1 | < 1 minute | Medium |

## Installation

Bayesian optimization requires the `scikit-optimize` package:

```bash
# Install with Bayesian optimization support
pip install -e ".[bayesian]"

# Or install directly
pip install scikit-optimize
```

## Usage

### JSON Configuration

Add Bayesian optimization to your workflow configuration:

```json
{
  "workflow_dag": {
    "fault_network": {
      "parameters": {
        "search_radius_meters": "auto",
        "search_time_window_hours": "auto",
        "auto_optimize_parameters": true,
        "optimization_method": "bayesian",
        "optimization_n_calls": 50,
        "optimization_n_initial_points": 10,
        "optimization_acquisition_func": "EI",
        "optimization_plot_results": true
      }
    }
  }
}
```

### Python API

```python
from hyfi.utils.parameter_optimization import optimize_fault_network_parameters
import pandas as pd

# Load your data
data_input = pd.read_csv('your_catalog.csv')
focal_mechanisms = pd.read_csv('your_focal_mechanisms.csv')  # Optional

# Run Bayesian optimization
recommended_params = optimize_fault_network_parameters(
    data_input, 
    focal_mechanisms, 
    method='bayesian',
    n_calls=50,
    n_initial_points=10,
    acquisition_func='EI',
    plot_results=True
)

print(f"Optimal search radius: {recommended_params['search_radius_meters']:.1f} m")
print(f"Optimal time window: {recommended_params['search_time_window_hours']:.1f} h")
```

## Configuration Parameters

### Core Parameters

- **`optimization_method`**: Set to `"bayesian"` to enable Bayesian optimization
- **`optimization_n_calls`** (default: 50): Total number of objective function evaluations
  - Includes both random initialization and Bayesian iterations
  - Recommended: 40-100 for most catalogs
  
- **`optimization_n_initial_points`** (default: 10): Number of random evaluations before starting GP-based optimization
  - Should be 10-25% of `n_calls`
  - More initial points = better initial coverage but slower convergence
  
- **`optimization_acquisition_func`** (default: "EI"): Acquisition function for selecting next evaluation point
  - `"EI"` (Expected Improvement): Best for most cases, balances exploration/exploitation
  - `"PI"` (Probability of Improvement): More conservative, focuses on improvement
  - `"LCB"` (Lower Confidence Bound): More explorative, good for noisy objectives
  - `"gp_hedge"`: Adaptive, automatically selects best acquisition strategy

### Optional Parameters

- **`optimization_plot_results`** (default: false): Generate visualization plots
- **`optimization_r_nn_range`**: Custom search radius range `[min, max]` in meters
- **`optimization_dt_nn_range`**: Custom time window range `[min, max]` in hours

## How It Works

### Algorithm Steps

1. **Initialization Phase** (Random Sampling)
   - Evaluates `n_initial_points` random parameter combinations
   - Builds initial dataset for Gaussian Process model

2. **Optimization Phase** (Bayesian Iterations)
   - Fits GP model to current observations
   - Uses acquisition function to select most promising next point
   - Evaluates objective at selected point
   - Updates GP model with new observation
   - Repeats until `n_calls` evaluations completed

3. **Result Selection**
   - Returns parameter combination with best objective score
   - Provides convergence diagnostics and confidence metrics

### Objective Function

The optimization minimizes a composite objective that considers:
- Fault plane recovery rate
- Statistical quality metrics (kappa, R/N ratio)
- Focal mechanism validation (angular differences)
- Active plane accuracy (when focal mechanisms available)

## Visualization

When `optimization_plot_results: true`, a comprehensive visualization is generated showing:

1. **Convergence Plot**: Objective value evolution over iterations
2. **Parameter Space Exploration**: 2D scatter of evaluated points
3. **Parameter Evolution**: Individual parameter trajectories
4. **Score Distribution**: Comparison between random and Bayesian phases
5. **Quality Metrics**: Fault plane detection and recovery rates
6. **Summary Statistics**: Best parameters and efficiency gains

## Best Practices

### When to Use Bayesian Optimization

✅ **Use Bayesian when:**
- You need optimal parameters quickly (< 1 hour)
- Working with medium to large catalogs (>500 events)
- Computational budget is limited
- You want probabilistic confidence estimates

❌ **Use Grid Search when:**
- You need exhaustive parameter space coverage
- Publication requires systematic exploration
- Computational resources are abundant
- Catalog is very small (<200 events)

### Recommended Settings

#### Small Catalogs (<500 events)
```json
{
  "optimization_n_calls": 30,
  "optimization_n_initial_points": 10,
  "optimization_acquisition_func": "EI"
}
```

#### Medium Catalogs (500-5000 events)
```json
{
  "optimization_n_calls": 50,
  "optimization_n_initial_points": 10,
  "optimization_acquisition_func": "EI"
}
```

#### Large Catalogs (>5000 events)
```json
{
  "optimization_n_calls": 75,
  "optimization_n_initial_points": 15,
  "optimization_acquisition_func": "gp_hedge"
}
```

## Example Workflow

```bash
# 1. Quick parameter estimate with heuristic
python -c "
from hyfi.utils.parameter_optimization import optimize_fault_network_parameters
import pandas as pd
data = pd.read_csv('catalog.csv')
params = optimize_fault_network_parameters(data, method='heuristic')
print(f'Quick estimate: {params[\"search_radius_meters\"]:.0f}m, {params[\"search_time_window_hours\"]:.0f}h')
"

# 2. Bayesian optimization for optimal parameters
hyfi run config_bayesian_example.json

# 3. Optional: Validate with grid search around optimum
# (manually adjust ranges based on Bayesian results)
```

## Troubleshooting

### "ImportError: Bayesian optimization requires scikit-optimize"

Install the package:
```bash
pip install scikit-optimize
```

### Optimization Not Converging

Try:
- Increase `optimization_n_calls` (e.g., to 75-100)
- Increase `optimization_n_initial_points` for better initialization
- Use `"gp_hedge"` acquisition function for adaptive strategy
- Check data quality and ensure sufficient event density

### Results Different from Grid Search

This is expected! Bayesian optimization:
- May find different local optima
- Has inherent randomness (can set `random_state` for reproducibility)
- Trades exhaustive coverage for efficiency

Both methods should find similarly good solutions. If results differ significantly, consider:
- Running optimization multiple times with different random seeds
- Increasing `n_calls` for more thorough exploration
- Validating results with focal mechanism metrics

## References

- Brochu, E., Cora, V. M., & De Freitas, N. (2010). A tutorial on Bayesian optimization of expensive cost functions. arXiv preprint arXiv:1012.2599.
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. NeurIPS.

## Citation

If you use Bayesian optimization in your research, please cite:

```bibtex
@software{hyfi_bayesian,
  author = {Truttmann, Sandro},
  title = {HyFI: Bayesian Optimization for Fault Network Parameters},
  year = {2025},
  url = {https://github.com/sandrotruttmann/hypo_fault_imaging}
}
```
