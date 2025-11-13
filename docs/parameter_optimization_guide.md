# 🎯 Automatic Parameter Optimization - User Guide

The HyFI package now includes **automatic parameter optimization** for fault network reconstruction! This feature automatically determines optimal search radius (`r_nn`) and search time window (`dt_nn`) parameters based on your earthquake catalog characteristics.

## ✨ Key Features

- **🧠 Intelligent Analysis**: Automatically analyzes spatial and temporal patterns in your earthquake catalog
- **🎯 Focal Mechanism Validation**: Uses focal mechanisms (including active plane information) to validate parameters
- **⚡ Multiple Methods**: Choose between fast heuristic or thorough grid search optimization
- **📊 Quality Metrics**: Provides confidence scores and expected performance metrics
- **🔧 Easy Integration**: Works seamlessly with existing workflows and configurations

## 🚀 Quick Start

### JSON Configuration (Recommended)

Simply set parameters to `"auto"` in your configuration file:

```json
{
  "workflow_dag": {
    "fault_network": {
      "parameters": {
        "search_radius_meters": "auto",
        "search_time_window_hours": "auto",
        "optimization_method": "grid_search"
      }
    }
  }
}
```

### Python API

```python
from hyfi.utils.parameter_optimization import optimize_fault_network_parameters

# Load your data
data_input = pd.read_csv('your_catalog.csv')
focal_mechanisms = pd.read_csv('focal_mechanisms.csv')  # Optional

# Get optimal parameters
params = optimize_fault_network_parameters(
    data_input, 
    focal_mechanisms, 
    method='grid_search'
)

print(f"Optimal radius: {params['search_radius_meters']:.1f} m")
print(f"Optimal time window: {params['search_time_window_hours']:.1f} h")
```

## 📋 Configuration Options

| Parameter | Description | Options | Default |
|-----------|-------------|---------|---------|
| `search_radius_meters` | Search radius | Numeric value or `"auto"` | `100.0` |
| `search_time_window_hours` | Time window | Numeric value or `"auto"` | `9999999` |
| `auto_optimize_parameters` | Enable auto-optimization | `true` or `false` | `false` |
| `optimization_method` | Method to use | `"heuristic"`, `"grid_search"` | `"grid_search"` |
| `optimization_grid_points` | Grid resolution | Integer (10-30) | `25` |

## 🔬 Optimization Methods

### 🏃 Heuristic Method
- **Speed**: Seconds to minutes
- **Use case**: Quick parameter estimation, real-time analysis
- **How it works**: Uses statistical rules based on nearest neighbor distances and event rates

### 🎯 Grid Search Method  
- **Speed**: Minutes to hours
- **Use case**: Optimal parameter selection for research/publication
- **How it works**: Systematically tests parameter combinations to find optimal values

## 🎪 Focal Mechanism Integration

The optimization can use focal mechanism data for validation:

### Required Columns
- `ID`: Event identifier (matching hypocenter catalog)
- `Strike1`, `Dip1`: First fault plane
- `Strike2`, `Dip2`: Second fault plane

### Optional Active Plane Column
- `A`: Active plane indicator (`1` or `2`)
  - Helps identify the geologically preferred fault plane
  - Improves parameter optimization accuracy

### Example Focal Mechanism Data
```csv
ID,Strike1,Dip1,Strike2,Dip2,A
EVT001,45,60,225,30,1
EVT002,120,75,300,15,2
```

## 📈 Understanding Results

### Confidence Metrics
- **Confidence Score**: 0-1 scale (higher = better)
- **Plane Recovery Rate**: Fraction of events with reliable fault planes
- **Expected Angular Difference**: Mean difference with focal mechanisms (if available)
- **Active Plane Accuracy**: Accuracy of active plane selection

### Quality Indicators
- **High Quality**: Recovery rate > 70%, confidence > 0.8
- **Good Quality**: Recovery rate > 50%, confidence > 0.6  
- **Acceptable**: Recovery rate > 30%, confidence > 0.4

## 🛠️ Performance Tips

### For Small Catalogs (<500 events)
```json
{
  "optimization_method": "heuristic",
  "optimization_grid_points": 15
}
```

### For Medium Catalogs (500-5000 events)  
```json
{
  "optimization_method": "grid_search",
  "optimization_grid_points": 20
}
```

### For Large Catalogs (>5000 events)
```json
{
  "optimization_method": "grid_search", 
  "optimization_grid_points": 25
}
```

## 📁 Output Files

The optimization generates several output files:

### `parameter_optimization_report.json`
Contains detailed optimization results and catalog statistics:
```json
{
  "optimization_results": {
    "search_radius_meters": 50.0,
    "search_time_window_hours": 26280.0,
    "method_used": "heuristic",
    "confidence_score": 0.75,
    "expected_planes": 74
  },
  "catalog_statistics": {
    "spatial": {...},
    "temporal": {...}
  }
}
```

## 🔧 Troubleshooting

### Common Issues

1. **"No optimization convergence"**
   - Try different optimization method
   - Check data quality and completeness
   - Reduce grid size for faster testing

2. **"Poor parameter quality"** 
   - Verify input data format
   - Check for sufficient event density
   - Consider manual parameter adjustment

3. **"Long optimization time"**
   - Use heuristic method for quick results
   - Reduce `optimization_grid_points`
   - Check catalog size

### Best Practices

✅ **Do:**
- Start with heuristic method for quick estimates
- Use grid search for final analysis
- Include focal mechanisms when available
- Validate results against literature values

❌ **Don't:**
- Use very small grid sizes (<10 points)
- Skip validation with manual parameters
- Ignore confidence scores
- Apply results blindly without checking

## 📚 Example Workflows

### Research Publication Workflow
```bash
# 1. Quick parameter estimate
python -c "
from hyfi.utils.parameter_optimization import optimize_fault_network_parameters
import pandas as pd
data = pd.read_csv('catalog.csv')
params = optimize_fault_network_parameters(data, method='heuristic')
print(f'Quick estimate: {params[\"search_radius_meters\"]:.0f}m, {params[\"search_time_window_hours\"]:.0f}h')
"

# 2. Optimal parameter search  
hyfi run config_with_auto_optimization.json

# 3. Validation run with manual parameters
hyfi run config_with_manual_params.json
```

### Real-time Analysis Workflow
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
