# Automatic Parameter Optimization for Fault Network Reconstruction

This document describes the automatic parameter optimization functionality for the fault network reconstruction module in the HyFI package.

## Overview

The fault network reconstruction algorithm requires two critical parameters:
- **Search radius (`r_nn`)**: The spatial search radius in meters for nearest neighbor search
- **Search time window (`dt_nn`)**: The temporal search window in hours for event clustering

These parameters significantly affect the quality and quantity of recovered fault planes. The automatic parameter optimization functionality helps determine optimal values based on catalog characteristics and focal mechanism validation (when available).

## Features

### Optimization Methods

1. **Heuristic Method** (`'heuristic'`)
   - Fast parameter estimation based on catalog statistics
   - Uses empirical rules
   - Suitable for quick estimates and initial parameter selection

2. **Grid Search Method** (`'grid_search'`)
   - Systematic exploration of parameter space
   - Evaluates multiple parameter combinations
   - More computationally intensive but provides optimal results

### Validation Metrics

The optimization uses multiple criteria to evaluate parameter quality:

1. **Fault Plane Recovery**: Number and percentage of events with reliable fault plane fits
2. **Statistical Quality**: Kappa and R/N ratios indicating fit reliability
3. **Focal Mechanism Validation**: Angular differences with observed focal mechanisms
4. **Active Plane Accuracy**: Preference for known active fault planes (when specified)

## Usage

### JSON Configuration

Add automatic parameter optimization to your workflow configuration:

```json
{
  "workflow_dag": {
    "fault_network": {
      "parameters": {
        "monte_carlo_simulations": 1000,
        "search_radius_meters": "auto",
        "search_time_window_hours": "auto",
        "magnitude_type": "ML",
        "auto_optimize_parameters": true,
        "optimization_method": "grid_search",
        "optimization_grid_points": 25
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

# Optimize parameters
recommended_params = optimize_fault_network_parameters(
    data_input, 
    focal_mechanisms, 
    method='grid_search'
)

print(f"Optimal search radius: {recommended_params['search_radius_meters']:.1f} m")
print(f"Optimal time window: {recommended_params['search_time_window_hours']:.1f} h")
```

## Configuration Parameters

### Required Parameters

- `search_radius_meters`: Set to `"auto"` to enable automatic optimization
- `search_time_window_hours`: Set to `"auto"` to enable automatic optimization

### Optional Parameters

- `auto_optimize_parameters`: Boolean, explicitly enable optimization (default: false)
- `optimization_method`: String, method to use (`"grid_search"` or `"heuristic"`)
- `optimization_grid_points`: Integer, number of grid points per dimension for grid search (default: 25)

## Input Data Requirements

### Hypocenter Catalog

Required columns:
- `ID`: Event identifier
- `YR`, `MO`, `DY`, `HR`, `MI`, `SC`: Date and time components
- `X`, `Y`, `Z`: Spatial coordinates
- `EX`, `EY`, `EZ`: Location uncertainties (optional)

### Focal Mechanism Data (Optional)

Required columns for validation:
- `ID`: Event identifier (must match hypocenter catalog)
- `Strike1`, `Dip1`: First fault plane orientation
- `Strike2`, `Dip2`: Second fault plane orientation

Optional columns:
- `A`: Active plane indicator (1 or 2) for preferred plane selection

## Output

### Recommended Parameters

The optimization returns a dictionary with:
- `search_radius_meters`: Optimal search radius
- `search_time_window_hours`: Optimal time window
- `expected_planes`: Number of fault planes expected with these parameters
- `plane_recovery_rate`: Fraction of events with reliable plane fits
- `confidence_score`: Confidence in parameter selection (0-1)

### Validation Metrics (if focal mechanisms available)

- `expected_angular_difference`: Mean angular difference with focal mechanisms
- `active_plane_accuracy`: Accuracy of active plane selection

### Optimization Report

When run through the workflow, an optimization report is saved as `parameter_optimization_report.json` containing:
- Optimization results and parameters
- Catalog statistics used for optimization
- Method-specific details

## Performance Considerations

### Heuristic Method
- **Runtime**: Seconds to minutes
- **Use case**: Quick parameter estimation, initial values
- **Accuracy**: Good for most catalogs

### Grid Search Method
- **Runtime**: Minutes to hours (depends on grid size)
- **Use case**: Optimal parameter selection for final analysis
- **Accuracy**: Highest accuracy, systematic exploration

### Grid Size Recommendations
- Small catalogs (<1000 events): 15-20 points per dimension
- Medium catalogs (1000-10000 events): 20-25 points per dimension  
- Large catalogs (>10000 events): 25-30 points per dimension

## Examples

### Example 1: Basic Optimization

```json
{
  "fault_network": {
    "parameters": {
      "search_radius_meters": "auto",
      "search_time_window_hours": "auto"
    }
  }
}
```

### Example 2: Grid Search with Custom Settings

```json
{
  "fault_network": {
    "parameters": {
      "auto_optimize_parameters": true,
      "optimization_method": "grid_search", 
      "optimization_grid_points": 30,
      "search_radius_meters": 100.0,
      "search_time_window_hours": 8760.0
    }
  }
}
```

### Example 3: Heuristic Method for Quick Analysis

```json
{
  "fault_network": {
    "parameters": {
      "optimization_method": "heuristic",
      "search_radius_meters": "auto",
      "search_time_window_hours": "auto"
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **Optimization fails**: Check input data format and completeness
2. **Poor parameter quality**: Try different optimization method or check data quality
3. **Long runtime**: Reduce grid size or use heuristic method
4. **No focal mechanism validation**: Ensure focal mechanism file exists and has correct format

### Logging

Enable detailed logging to monitor optimization progress:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Best Practices

1. **Use heuristic method first** for quick parameter estimates
2. **Use grid search for final analysis** when accuracy is critical
3. **Include focal mechanisms** when available for better validation
4. **Specify active planes** in focal mechanism data for optimal results
5. **Check optimization report** for quality assessment

