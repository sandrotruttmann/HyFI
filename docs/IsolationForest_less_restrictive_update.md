# Isolation Forest - Very Conservative Update

## Changes Made

I've updated the Isolation Forest implementation to be **very conservative** by default, using only 1% contamination to minimize false positive outlier detection.

## Key Changes

### 1. Default `contamination` Parameter
**OLD:** `contamination='auto'` (sklearn automatic threshold - often too aggressive)
**CURRENT:** `contamination=0.01` (expects 1% outliers - very conservative and appropriate for seismic data)

### 2. Updated Files

#### `fault_network.py`
- Changed function signature: `def IsolationForest_outlier_detection(..., contamination=0.01, ...)`
- Added warning when user explicitly sets `contamination='auto'`
- Updated docstring with detailed guidance on contamination values
- Updated `faultnetwork3D()` to use `0.01` as default instead of `'auto'`

#### `config_IsolationForest_example.json`
- Changed: `"if_contamination": 0.01` (was `"auto"`)
- Updated comments to explain contamination options

#### `compare_outlier_methods.py`
- Changed default parameter: `if_contamination=0.01` (was `'auto'`)
- Updated command-line help text

#### `outlier_detection_methods.md`
- Updated parameter descriptions with clear guidance
- Added new examples showing different contamination levels
- Added warnings about 'auto' being too aggressive

## Contamination Value Guide

| Value | Behavior | Use Case |
|-------|----------|----------|
| **0.01** | Very conservative (NEW DEFAULT) | General seismic data, minimal removal |
| **0.02** | Conservative | High-quality data with few expected outliers |
| **0.05** | Moderate | Balanced approach, more outlier removal |
| **0.10** | Aggressive | Noisy data, substantial outlier removal |
| **'auto'** | Very aggressive (NOT RECOMMENDED) | Can over-detect outliers in seismic data |

## Impact

### Before (with 'auto'):
- Could mark 10-20%+ of events as outliers
- Too aggressive for typical seismic catalogs
- Inconsistent results across datasets

### After (with 0.01):
- Expects only ~1% outliers (very conservative)
- Minimizes false positives
- Preserves nearly all legitimate events
- Consistent, predictable behavior
- Still effective at removing true outliers

## Usage Examples

### Default (Very Conservative - Recommended)
```json
{
  "remove_outliers": true,
  "outlier_method": "IsolationForest"
}
```
This now uses `if_contamination=0.01` by default (1% outliers expected).

### Moderate Setting
```json
{
  "remove_outliers": true,
  "outlier_method": "IsolationForest",
  "if_contamination": 0.05
}
```
This expects ~5% outliers (more outlier removal).

### Aggressive Setting (For Noisy Data)
```json
{
  "remove_outliers": true,
  "outlier_method": "IsolationForest",
  "if_contamination": 0.1
}
```

### Use 'auto' (Not Recommended)
```json
{
  "remove_outliers": true,
  "outlier_method": "IsolationForest",
  "if_contamination": "auto"
}
```
⚠️ This will show a warning message.

## Testing Your Data

Run the comparison script to see how different contamination values perform:

```bash
# Default (0.01 = 1%, very conservative)
python examples_archive/compare_outlier_methods.py data_examples/A18_data.csv

# Conservative (0.02 = 2%)
python examples_archive/compare_outlier_methods.py data_examples/A18_data.csv \
    --if-contamination 0.02

# Moderate (0.05 = 5%)
python examples_archive/compare_outlier_methods.py data_examples/A18_data.csv \
    --if-contamination 0.05

# Aggressive (0.1 = 10%)
python examples_archive/compare_outlier_methods.py data_examples/A18_data.csv \
    --if-contamination 0.1
```

## Recommendations

1. **Start with default (0.01)** - Very conservative, appropriate for most seismic data
2. **Review outliers** - Check if marked outliers make geological sense
3. **Adjust if needed**:
   - Still too many outliers? → Keep at 0.01 or even try 0.005
   - Too few outliers? → Increase to 0.02, 0.05, or 0.1
4. **Avoid 'auto'** - Explicit values give more control and consistency

## Summary

The Isolation Forest method is now **very conservative by default**, using a 1% outlier expectation instead of the aggressive automatic threshold. This should provide excellent results for typical seismic hypocenter catalogs by only removing the most obvious outliers while preserving nearly all legitimate events.
