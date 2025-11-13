# Pareto Multi-Objective Optimization - Quick Start Guide

## What is Pareto Optimization?

Instead of combining objectives with fixed weights (e.g., "60% focal fit + 25% recovery + 15% quality"), Pareto optimization finds **all optimal trade-offs**. You get multiple solutions showing different balances, then choose based on your priorities.

**Result**: The Pareto front - a set of solutions where improving one objective requires sacrificing another.

## When to Use Pareto vs. Other Methods?

| Method | Best For | Time | Output |
|--------|----------|------|--------|
| Grid Search | Exhaustive exploration, publications | 2-4 hours | 1 solution (+ full parameter map) |
| Bayesian | Fast optimization, limited time | 20-40 min | 1 solution (best found) |
| Optuna | Modern workflow, good visualization | 20-40 min | 1 solution (best found) |
| **Pareto** | **Publications, exploring trade-offs** | **45-90 min** | **Multiple solutions (Pareto front)** |

**Use Pareto when:**
- Writing a paper (more rigorous than arbitrary weights)
- You want to see trade-offs explicitly
- Different stakeholders have different priorities
- You're unsure which objective matters most

**Don't use Pareto when:**
- You need a quick answer (use Bayesian/Optuna instead)
- You have very clear priorities (use weighted single-objective)
- Computational time is critical (use heuristic)

## Minimal Example

### 1. Config File Approach

Create `config_pareto.json`:
```json
{
  "project_title": "Pareto Optimization Example",
  "hypo_file": "your_earthquake_catalog.csv",
  "hypo_sep": ",",
  "out_dir": "output_pareto",
  
  "r_nn": "auto",
  "dt_nn": "auto",
  "auto_optimize_parameters": true,
  "optimization_method": "pareto",
  "optimization_n_trials": 100,
  "optimization_plot_results": true
}
```

Run:
```bash
hyfi config_pareto.json
```

Results will be in `output_pareto/`:
- `parameter_optimization_pareto.png` - Visualization
- `parameter_optimization_report.json` - Detailed results
- `HyFI_results.csv` - Fault network using recommended parameters

### 2. Python API Approach

```python
from hyfi.utils.parameter_optimization import ParameterOptimizer
import pandas as pd

# Load data
data = pd.read_csv('earthquake_catalog.csv')

# Optional: Load focal mechanisms for 3-objective optimization
focal_data = pd.read_csv('focal_mechanisms.csv', sep=';')

# Run Pareto optimization
optimizer = ParameterOptimizer(data, focal_mechanisms=focal_data)
results = optimizer.optimize_pareto(
    n_trials=100,
    sampler='nsga2',
    plot_results=True,
    save_plot_path='pareto_front.png'
)

# Get recommended solution (best balanced)
best = results['best_balanced']
print(f"Search radius: {best['r_nn']:.1f} m")
print(f"Time window: {best['dt_nn']:.1f} hours")
print(f"Recovery: {best['plane_recovery_rate']*100:.1f}%")

# See all Pareto-optimal solutions
pareto_front = results['pareto_front']
print(f"\nFound {len(pareto_front)} Pareto-optimal solutions")
```

## Understanding the Output

### Representative Solutions

Pareto optimization provides 4 representative solutions:

1. **best_balanced** ⭐ *[RECOMMENDED DEFAULT]*
   - Best overall compromise
   - Minimizes distance from ideal point
   - Use when priorities are unclear

2. **best_focal** (if focal mechanisms available)
   - Best agreement with focal mechanisms
   - Use for publications emphasizing validation

3. **best_recovery**
   - Maximum number of fault planes
   - Use for comprehensive fault mapping

4. **best_quality**
   - Highest statistical confidence (λ2/3 ratios)
   - Use when reliability matters most

### Example Output

```
Representative Pareto Solutions:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best Balanced:
  r_nn=127.3m, dt_nn=8234.5h
  Focal fit: 18.3°, Recovery: 82.1%, λ2/3: 31.2

Best Focal Fit:
  r_nn=98.4m, dt_nn=6521.3h
  Focal fit: 15.7°, Recovery: 76.5%, λ2/3: 28.9

Best Recovery:
  r_nn=156.2m, dt_nn=12453.8h
  Focal fit: 22.1°, Recovery: 89.3%, λ2/3: 25.4

Best Quality:
  r_nn=115.6m, dt_nn=7845.2h
  Focal fit: 19.8°, Recovery: 79.7%, λ2/3: 45.6
```

**Interpretation:**
- **Best Focal**: Sacrifices ~6% recovery for 2.6° better focal fit
- **Best Recovery**: Gets 7% more planes but focal fit degrades by 3.8°
- **Best Quality**: λ2/3 is 45% higher, indicating very reliable planes
- **Best Balanced**: Middle ground - good performance everywhere

## Configuration Parameters

### Essential Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimization_method` | - | Set to `"pareto"` |
| `optimization_n_trials` | 100 | Number of parameter combinations to test |
| `optimization_plot_results` | false | Generate Pareto front visualization |

### Advanced Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimization_pareto_sampler` | `"nsga2"` | Algorithm: `"nsga2"` (recommended), `"nsga3"`, `"random"` |
| `optimization_pareto_population` | 50 | Population size for evolutionary algorithm |
| `optimization_n_startup_trials` | 20 | Random exploration before optimization |

### Typical Configurations

**Quick Exploration** (testing):
```json
{
  "optimization_n_trials": 50,
  "optimization_pareto_population": 30
}
```

**Standard** (default):
```json
{
  "optimization_n_trials": 100,
  "optimization_pareto_sampler": "nsga2",
  "optimization_pareto_population": 50
}
```

**Publication Quality** (thorough):
```json
{
  "optimization_n_trials": 200,
  "optimization_pareto_sampler": "nsga2",
  "optimization_pareto_population": 100,
  "optimization_n_startup_trials": 30
}
```

## Visualization Guide

The Pareto plot shows:

### With Focal Mechanisms (3D)

1. **3D Pareto Front** (main plot)
   - Blue points: All Pareto-optimal solutions
   - Gray points: Dominated solutions
   - Colored markers: Representative solutions
   
2. **2D Projections**
   - Focal vs Recovery: Does better focal fit cost coverage?
   - Focal vs Quality: Relationship between fit and statistical quality
   - Recovery vs Quality: Coverage vs. reliability trade-off

3. **Parameter Space**
   - Where Pareto solutions are in (r_nn, dt_nn) space
   - Often clustered (good sign - consistent optimal region)

### Without Focal Mechanisms (2D)

1. **Pareto Front** (Recovery vs Quality)
   - Shows trade-off between coverage and reliability
   - Curve shape indicates conflict strength
   
2. **Parameter Space**
   - Distribution of optimal (r_nn, dt_nn) combinations

## Selecting Your Solution

### Decision Tree

```
Do you have focal mechanisms?
├─ YES
│  ├─ Publishing? → Use best_focal (most defensible)
│  ├─ Exploring? → Use best_recovery (most complete)
│  └─ Unsure? → Use best_balanced (safe choice)
│
└─ NO
   ├─ Need coverage? → Use best_recovery
   ├─ Need reliability? → Use best_quality
   └─ Balanced approach? → Use best_balanced
```

### Custom Selection

If none of the representatives fit your needs, select manually:

```python
# Example: Recovery > 85% AND λ2/3 > 30
pareto_front = results['pareto_front']

candidates = [
    sol for sol in pareto_front
    if sol['plane_recovery_rate'] > 0.85 and 
       sol['mean_lambda23_ratio'] > 30
]

if candidates:
    # Among candidates, pick best focal fit
    best = min(candidates, key=lambda x: x.get('mean_angular_diff', 999))
    print(f"Custom selection: r_nn={best['r_nn']:.1f}m, dt_nn={best['dt_nn']:.1f}h")
```

## Time Estimates

| Catalog Size | n_trials=50 | n_trials=100 | n_trials=200 |
|--------------|-------------|--------------|--------------|
| <500 events  | 15-25 min   | 30-45 min    | 60-90 min    |
| 500-2000     | 25-40 min   | 45-75 min    | 90-150 min   |
| >2000        | 40-60 min   | 75-120 min   | 150-240 min  |

**Recommendation**: Start with `n_trials=50` for testing, use `n_trials=100` for final results.

## Troubleshooting

### "Import Error: No module named optuna.samplers"

Install Optuna with Pareto support:
```bash
pip install optuna
```

Or install HyFI with all optimization features:
```bash
pip install hyfi[optuna]
# or
pip install hyfi[all]
```

### Pareto Front is Too Large (>50 solutions)

**Cause**: Objectives don't conflict strongly; many solutions are equally good.

**Solution**: This is actually good! It means parameter choice is robust. Just use `best_balanced`.

### Pareto Front is Too Small (<5 solutions)

**Cause**: One objective dominates, or insufficient trials.

**Solutions:**
1. Increase `optimization_n_trials` to 150-200
2. Check if focal mechanisms are very consistent (less variation to optimize)
3. Verify data quality

### Long Runtime

**Speed up:**
1. Reduce `n_trials` for testing (minimum 50)
2. Reduce `n_mc` in fault network (e.g., 500 instead of 1000)
3. Use subset of catalog for parameter optimization

### "All solutions have same parameters"

**Cause**: Parameter ranges too narrow, or data doesn't support discrimination.

**Solutions:**
1. Check parameter ranges in code (should be reasonable defaults)
2. Verify earthquake catalog has sufficient events (>100 recommended)
3. Check if focal mechanisms are available and valid

## Comparison with Single-Objective

### Single-Objective (Bayesian/Optuna)
```json
{
  "optimization_method": "optuna",
  "optimization_n_trials": 50
}
```
→ **Output**: One solution with score 0.187
→ **Question**: "Is 0.187 good? What if we used different weights?"

### Pareto Multi-Objective
```json
{
  "optimization_method": "pareto",
  "optimization_n_trials": 100
}
```
→ **Output**: 23 Pareto-optimal solutions
→ **Insight**: "Focal fit ranges 15-25°, recovery 75-90%, quality 25-45"
→ **Choice**: Pick based on your needs, with full context

## Next Steps

1. **Run Initial Optimization**
   ```bash
   hyfi config_pareto_example.json
   ```

2. **Examine Visualization**
   - Look at `parameter_optimization_pareto.png`
   - Understand trade-offs in your data

3. **Select Solution**
   - Default: Use `best_balanced`
   - Publication: Use `best_focal`
   - Exploration: Use `best_recovery`

4. **Run Full Analysis**
   - Update config with selected parameters
   - Run complete fault network workflow

5. **Document Choice**
   - Report all representative solutions in paper/report
   - Explain why you selected your preferred solution
   - Show Pareto front plot in supplementary materials

## Example for Paper

> "We employed Pareto multi-objective optimization using the NSGA-II algorithm 
> (Deb et al., 2002) to determine optimal fault network parameters. The 
> optimization balanced three competing objectives: (1) focal mechanism agreement 
> (mean angular difference), (2) plane recovery rate (completeness), and (3) 
> statistical quality (λ₂/λ₃ eigenvalue ratio). 
>
> From 100 trials, we identified 23 Pareto-optimal parameter combinations 
> (Figure S1). Among these, we selected the solution maximizing focal mechanism 
> agreement (r_nn = 127 m, Δt_nn = 8,235 hours), achieving 18.3° mean angular 
> difference while maintaining 82% plane recovery and λ₂/λ₃ = 31. Alternative 
> selections prioritizing recovery (89%, 22.1° angular difference) or quality 
> (λ₂/λ₃ = 45, 19.8° angular difference) are available in the Pareto front 
> (Table S1)."

## Resources

- **Full Documentation**: `docs/pareto_optimization.md`
- **Advanced Objective Functions**: `docs/advanced_objective_functions.md`
- **Example Config**: `example_projects/config_pareto_example.json`
- **Method Comparison**: `docs/optimization_methods_comparison.md`

## Summary

**Quick Recipe:**
1. Set `"optimization_method": "pareto"`
2. Use `n_trials=100` (or 50 for testing)
3. Generate plot to see trade-offs
4. Use `best_balanced` by default
5. Choose others if you have specific priorities

**Key Advantage**: See all optimal trade-offs, not just one weighted solution.

**Trade-off**: Takes ~2x longer than single-objective, but provides much richer information.
