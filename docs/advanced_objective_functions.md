# Advanced Objective Function Approaches

## Current Approach Analysis

The current objective function uses a **weighted linear combination** approach:

```
f = w1·S_angular + w2·S_active + w3·S_recovery + w4·S_lambda23
```

**Strengths:**
- Simple and interpretable
- Works well in practice
- Fast to compute
- Easy to tune weights

**Limitations:**
- Fixed weights don't adapt to problem characteristics
- Linear combination may not capture trade-offs optimally
- Doesn't account for uncertainty in metrics
- Treats all objectives as commensurable (comparable)

## Proposed Advanced Approaches

### 1. **Pareto Multi-Objective Optimization** ⭐ RECOMMENDED

Instead of a single weighted objective, treat this as a **true multi-objective problem** and find the Pareto front.

#### Concept
Each parameter setting produces a vector of objectives:
```
f(r_nn, dt_nn) → (S_angular, S_recovery, S_lambda23)
```

A solution is **Pareto optimal** if improving one objective requires degrading another.

#### Advantages
- No need to choose weights beforehand
- Reveals trade-offs between objectives (e.g., recovery vs. quality)
- User can select from Pareto front based on priorities
- More scientifically rigorous for publications

#### Implementation Options

**Option A: NSGA-II (Non-dominated Sorting Genetic Algorithm)**
```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# Define multi-objective problem
class FaultNetworkProblem(Problem):
    def _evaluate(self, X, out, *args, **kwargs):
        # X: array of (r_nn, dt_nn) pairs
        # out["F"]: array of objective vectors
        
# Run optimization
algorithm = NSGA2(pop_size=50)
res = minimize(problem, algorithm, termination=('n_gen', 100))
pareto_front = res.F  # All Pareto-optimal solutions
```

**Option B: Optuna Multi-Objective**
```python
def objective(trial):
    r_nn = trial.suggest_float('r_nn', 50, 1000, log=True)
    dt_nn = trial.suggest_float('dt_nn', 100, 50000, log=True)
    
    # Return tuple of objectives (all minimization)
    return (S_angular, S_recovery, S_lambda23)

study = optuna.create_study(directions=['minimize', 'minimize', 'minimize'])
study.optimize(objective, n_trials=100)
```

**Visualization**: 3D scatter plot showing Pareto front with trade-offs

#### When to Use
- Publications requiring rigorous analysis
- Exploring parameter space comprehensively
- When you're unsure about relative importance of objectives

---

### 2. **Adaptive Weight Learning** ⭐ RECOMMENDED FOR PRACTICE

Learn optimal weights automatically based on problem characteristics.

#### Approach A: Data-Driven Weight Adaptation

```python
def adaptive_weights(quality_metrics, focal_metrics):
    """
    Adapt weights based on data quality and availability.
    """
    weights = {}
    
    # Base weights
    if focal_metrics:
        # More focal mechanisms → higher weight on angular fit
        n_focal = focal_metrics.get('n_focal_comparisons', 0)
        focal_confidence = min(n_focal / 100, 1.0)  # Saturate at 100 focals
        
        weights['angular'] = 0.4 + 0.3 * focal_confidence  # 40-70%
        weights['recovery'] = 0.3 - 0.1 * focal_confidence  # 20-30%
        weights['quality'] = 0.3 - 0.2 * focal_confidence   # 10-30%
    else:
        weights['recovery'] = 0.7
        weights['quality'] = 0.3
    
    # Adapt based on current performance
    if quality_metrics.get('mean_lambda23_ratio', 0) > 50:
        # Already excellent quality, reduce its importance
        transfer = weights['quality'] * 0.3
        weights['quality'] -= transfer
        weights['recovery'] += transfer
    
    return weights
```

#### Approach B: Bayesian Weight Optimization

Pre-run small grid search on subset, learn which weights correlate with good results:

```python
# Meta-optimization: find weights that work best for this dataset
def meta_objective(weight_vector):
    w_angular, w_recovery, w_quality = weight_vector / weight_vector.sum()
    
    # Quick grid search with these weights
    best_result = quick_grid_search(weights=(w_angular, w_recovery, w_quality))
    
    # Return quality of best result (e.g., focal fit if available)
    return best_result['focal_angular_diff']

# Optimize weights themselves
best_weights = bayesian_optimize(meta_objective, n_calls=20)
```

---

### 3. **Hierarchical Objective Function**

Use objectives in **priority order** rather than weighted sum.

#### Concept
```python
def hierarchical_objective(params):
    """
    Lexicographic optimization: optimize in strict priority order.
    """
    results = evaluate_params(params)
    
    # Primary: Focal mechanism fit (if available)
    if has_focal_data:
        primary = results['angular_diff']
        
        # Secondary: Among solutions with angular_diff < 20°, maximize recovery
        if primary < 20:
            secondary = -results['recovery_rate']  # Negative for maximization
            
            # Tertiary: Among good recovery, maximize quality
            if results['recovery_rate'] > 0.8:
                tertiary = -results['lambda23_ratio']
                return (primary, secondary, tertiary)
            return (primary, secondary, 0)
        return (primary, 0, 0)
    else:
        # Without focals: maximize recovery first, then quality
        primary = -results['recovery_rate']
        secondary = -results['lambda23_ratio'] if results['recovery_rate'] > 0.7 else 0
        return (primary, secondary, 0)
```

#### Advantages
- Clear priority structure
- Avoids bad trade-offs (e.g., won't sacrifice focal fit for slightly better recovery)
- Easy to explain and justify

#### Disadvantages
- Can be overly rigid
- Might miss good solutions just outside thresholds

---

### 4. **Uncertainty-Weighted Objective**

Account for **confidence** in each metric.

#### Concept

```python
def uncertainty_weighted_objective(params):
    """
    Weight objectives by inverse of their uncertainty.
    
    More uncertain metrics get lower weights.
    """
    results = evaluate_params(params)
    
    # Calculate metric uncertainties
    angular_std = results['angular_diff_std']  # Std dev across events
    angular_weight = 1 / (1 + angular_std / 10)  # Lower std → higher weight
    
    n_events = results['n_events']
    recovery_uncertainty = 1 / sqrt(n_events)  # Bootstrap-style
    recovery_weight = 1 / (1 + recovery_uncertainty)
    
    lambda23_std = results['lambda23_std']
    quality_weight = 1 / (1 + lambda23_std / 20)
    
    # Normalize weights
    total = angular_weight + recovery_weight + quality_weight
    
    # Weighted objective
    objective = (
        (angular_weight / total) * results['angular_diff'] +
        (recovery_weight / total) * (1 - results['recovery_rate']) +
        (quality_weight / total) * (1 - normalize_lambda23(results['lambda23_ratio']))
    )
    
    return objective
```

#### Advantages
- Statistically principled
- Automatically down-weights noisy metrics
- Works well with small catalogs

---

### 5. **Non-Linear Transformation Functions**

Instead of linear scaling, use **domain-aware transformations**.

#### Current Lambda23 Scaling
```python
# Current: Piecewise linear
if lambda23 < 5:
    score = 0.0
elif lambda23 < 20:
    score = 0.7 + (lambda23 - 5) / 50
else:
    score = 1.0
```

#### Improved: Sigmoid Transformations

```python
def sigmoid_transform(x, inflection=10, steepness=0.2):
    """
    Smooth sigmoid instead of piecewise linear.
    
    - inflection: value at which score = 0.5
    - steepness: how quickly it transitions
    """
    return 1 / (1 + np.exp(-steepness * (x - inflection)))

def lambda23_score(lambda23):
    """
    Quality score using sigmoid.
    """
    if lambda23 < 5:
        return 0.0  # Hard cutoff for unacceptable
    
    # Sigmoid between 5 and 100
    normalized = sigmoid_transform(lambda23, inflection=25, steepness=0.1)
    return 0.7 + 0.3 * normalized  # Scale to [0.7, 1.0]
```

**For Angular Difference:**
```python
def angular_score(theta):
    """
    Non-linear penalty for angular differences.
    
    Heavily penalize >45°, gently penalize <15°.
    """
    if theta < 15:
        return theta / 90  # Linear for good fits
    else:
        # Exponential penalty for poor fits
        return (theta / 90) ** 2
```

---

### 6. **Constraint-Based Optimization**

Turn some objectives into **hard constraints** rather than objectives.

#### Concept

```python
def objective_with_constraints(params):
    """
    Minimize focal misfit subject to constraints.
    """
    results = evaluate_params(params)
    
    # Hard constraints
    if results['recovery_rate'] < 0.5:
        return float('inf')  # Reject: too few planes
    
    if results['mean_lambda23_ratio'] < 5:
        return float('inf')  # Reject: poor quality
    
    if has_focal_data and results['angular_diff'] > 45:
        return float('inf')  # Reject: unacceptable focal fit
    
    # Primary objective: minimize angular difference
    # (among solutions that satisfy constraints)
    return results['angular_diff']
```

#### With Soft Constraints (Penalty Method)

```python
def objective_with_penalties(params):
    results = evaluate_params(params)
    
    # Base objective
    objective = results['angular_diff']
    
    # Add penalties for constraint violations
    if results['recovery_rate'] < 0.5:
        penalty = 100 * (0.5 - results['recovery_rate'])
        objective += penalty
    
    if results['mean_lambda23_ratio'] < 5:
        penalty = 50 * (5 - results['mean_lambda23_ratio'])
        objective += penalty
    
    return objective
```

---

### 7. **Machine Learning-Based Surrogate**

For expensive objective functions, build a **surrogate model**.

#### Concept

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class SurrogateOptimizer:
    def __init__(self, true_objective):
        self.true_objective = true_objective
        self.X_observed = []
        self.y_observed = []
        self.gp = None
    
    def fit_surrogate(self):
        """Fit GP to observed evaluations."""
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        self.gp.fit(self.X_observed, self.y_observed)
    
    def acquisition_function(self, X):
        """Expected Improvement."""
        mu, sigma = self.gp.predict(X, return_std=True)
        best_y = min(self.y_observed)
        
        Z = (best_y - mu) / (sigma + 1e-9)
        ei = (best_y - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei
    
    def optimize_with_surrogate(self, n_initial=10, n_iterations=40):
        # Initial random sampling
        for _ in range(n_initial):
            params = random_params()
            y = self.true_objective(params)
            self.X_observed.append(params)
            self.y_observed.append(y)
        
        # Iterative optimization
        for _ in range(n_iterations):
            self.fit_surrogate()
            
            # Find point with highest expected improvement
            next_params = optimize_acquisition(self.acquisition_function)
            
            # Evaluate true objective
            y = self.true_objective(next_params)
            self.X_observed.append(next_params)
            self.y_observed.append(y)
```

This is essentially what Bayesian optimization does, but you could extend it with:
- **Multi-fidelity**: Use quick approximate evaluations (fewer Monte Carlo runs)
- **Transfer learning**: Use results from similar catalogs as prior

---

## Practical Recommendations

### For Most Use Cases: **Adaptive Weights** (Approach #2)
```python
# Easy to implement, significant improvement over fixed weights
# Preserves simplicity while adapting to data characteristics
```

### For Publications: **Pareto Multi-Objective** (Approach #1)
```python
# More rigorous, shows trade-offs explicitly
# Provides multiple solutions for different priorities
```

### For Large Catalogs: **Hierarchical + Constraints** (Approaches #3 + #6)
```python
# Clear requirements (e.g., recovery > 70%)
# Strict priority order makes sense when you have enough data
```

### For Small/Noisy Catalogs: **Uncertainty-Weighted** (Approach #4)
```python
# Accounts for statistical uncertainty
# Prevents over-optimizing on noisy metrics
```

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. ✅ **Sigmoid transformations** for lambda23 and angular scores
2. ✅ **Adaptive weights** based on focal mechanism availability
3. ✅ Add **uncertainty metrics** (std dev of angular diff, lambda23)

### Phase 2: Medium-Term (1 week)
4. **Pareto optimization** using Optuna multi-objective
5. **Constraint-based** optimization with soft penalties
6. Comprehensive **visualization** of trade-offs

### Phase 3: Advanced (2-4 weeks)
7. **Transfer learning** from previous optimization runs
8. **Multi-fidelity** optimization (quick evaluations + detailed validation)
9. **Automated weight tuning** via meta-optimization

---

## Comparison Table

| Approach | Complexity | Computation | Interpretability | Robustness | Best For |
|----------|-----------|-------------|------------------|------------|----------|
| Current Linear | ⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ | General use |
| Pareto Multi-Objective | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Publications |
| Adaptive Weights | ⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Practice |
| Hierarchical | ⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ | Clear priorities |
| Uncertainty-Weighted | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | Small datasets |
| Non-Linear Transform | ⭐ | ⭐ | ⭐⭐ | ⭐⭐ | All cases |
| Constraint-Based | ⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ | Hard requirements |
| ML Surrogate | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ | Expensive evals |

---

## Example Implementation

Here's a drop-in replacement that combines several improvements:

```python
def _calculate_combined_objective_v2(self, plane_recovery_rate, quality_metrics, 
                                     focal_metrics, adaptive=True):
    """
    Enhanced objective function with adaptive weights and non-linear transformations.
    """
    
    # ============= ADAPTIVE WEIGHT CALCULATION =============
    if adaptive:
        weights = self._calculate_adaptive_weights(quality_metrics, focal_metrics)
    else:
        # Fallback to original weights
        has_focal_data = (focal_metrics and 'mean_angular_diff' in focal_metrics)
        if has_focal_data:
            weights = {'angular': 0.60, 'active': 0.05, 'recovery': 0.25, 'quality': 0.15}
        else:
            weights = {'recovery': 0.70, 'quality': 0.30, 'angular': 0.0, 'active': 0.0}
    
    # ============= NON-LINEAR SCORE TRANSFORMATIONS =============
    score = 0.0
    
    # 1. Angular difference (if available)
    if weights.get('angular', 0) > 0:
        angular_diff = focal_metrics['mean_angular_diff']
        
        # Non-linear penalty: gentle for good fits, harsh for poor fits
        if angular_diff < 15:
            angular_score = angular_diff / 90
        else:
            angular_score = (angular_diff / 90) ** 1.5  # Superlinear penalty
        
        score += weights['angular'] * angular_score
    
    # 2. Active plane bonus (if available)
    if weights.get('active', 0) > 0:
        if 'active_plane_match_rate' in focal_metrics:
            active_score = 1 - focal_metrics['active_plane_match_rate']
            score += weights['active'] * active_score
        else:
            score += weights['active'] * 1.0
    
    # 3. Recovery rate
    recovery_score = 1 - plane_recovery_rate
    
    # Add soft constraint penalty for very low recovery
    if plane_recovery_rate < 0.3:
        penalty = 0.5 * (0.3 - plane_recovery_rate) / 0.3
        recovery_score += penalty
    
    score += weights['recovery'] * recovery_score
    
    # 4. Quality (lambda23) with sigmoid transformation
    if 'mean_lambda23_ratio' in quality_metrics:
        lambda23 = quality_metrics['mean_lambda23_ratio']
        
        if lambda23 < 5.0:
            quality_normalized = 0.0
        else:
            # Sigmoid transformation: smooth transition from 5 to 50
            # Center at 15, steepness 0.1
            x = (lambda23 - 15) / 10
            sigmoid = 1 / (1 + np.exp(-x))
            quality_normalized = 0.7 + 0.3 * sigmoid
        
        quality_score = 1 - quality_normalized
        score += weights['quality'] * quality_score
    else:
        score += weights['quality'] * 1.0
    
    return score

def _calculate_adaptive_weights(self, quality_metrics, focal_metrics):
    """
    Calculate weights that adapt to data characteristics.
    """
    weights = {}
    
    has_focal_data = (focal_metrics and 'mean_angular_diff' in focal_metrics)
    
    if has_focal_data:
        # Adapt based on number of focal mechanisms
        n_focal = focal_metrics.get('n_focal_comparisons', 0)
        focal_confidence = min(n_focal / 100, 1.0)
        
        # More focals → trust angular fit more
        weights['angular'] = 0.40 + 0.25 * focal_confidence  # 40-65%
        weights['active'] = 0.05
        weights['recovery'] = 0.30 - 0.05 * focal_confidence  # 25-30%
        weights['quality'] = 0.25 - 0.20 * focal_confidence   # 5-25%
        
        # If quality is already excellent, reduce its weight further
        if quality_metrics.get('mean_lambda23_ratio', 0) > 50:
            transfer = weights['quality'] * 0.4
            weights['quality'] -= transfer
            weights['recovery'] += transfer * 0.5
            weights['angular'] += transfer * 0.5
    else:
        weights['angular'] = 0.0
        weights['active'] = 0.0
        weights['recovery'] = 0.70
        weights['quality'] = 0.30
    
    return weights
```

Would you like me to implement any of these approaches in your codebase?

