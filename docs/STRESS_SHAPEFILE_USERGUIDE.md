# Spatially-Varying Stress Field Feature - User Guide

## Overview

The HyFI stress analysis module now supports reading stress field parameters from a shapefile, allowing you to define spatially-varying stress regimes across your study area. This is particularly useful when working with regions that have heterogeneous stress fields.

## Quick Start

### 1. Prepare Your Shapefile

Your shapefile must contain polygon features with the following attributes:

| Column Name | Description | Units | Range |
|-------------|-------------|-------|-------|
| `s1_trend` | σ₁ (maximum principal stress) azimuth | degrees | 0-360 |
| `s1_plunge` | σ₁ plunge angle | degrees | 0-90 |
| `s3_trend` | σ₃ (minimum principal stress) azimuth | degrees | 0-360 |
| `s3_plunge` | σ₃ plunge angle | degrees | 0-90 |
| `R` | Stress shape ratio (σ₂-σ₃)/(σ₁-σ₃) | - | 0-1 |

**Important**: The shapefile coordinate system must match your hypocenter coordinates (e.g., CH1903+ for Swiss data).

### 2. Configure Your Analysis

In your JSON configuration file, enable `use_shapefile` and provide the shapefile path:

```json
{
  "stress_analysis": {
    "enabled": true,
    "parameters": {
      "stress_field": {
        "use_shapefile": true,
        "shapefile_path": "data_examples/Stressfield/CH_stressfield_Kastrup.shp",
        "sigma1_trend_degrees": 301,
        "sigma1_plunge_degrees": 23,
        "sigma3_trend_degrees": 43,
        "sigma3_plunge_degrees": 26,
        "stress_shape_ratio": 0.35
      },
      "mechanical_properties": {
        "pore_pressure_mpa": 0.0,
        "friction_coefficient": 0.75
      }
    }
  }
}
```

**Note**: The fixed stress field parameters serve as fallback values if the shapefile cannot be loaded or if the hypocenter center point falls outside all polygons.

### 3. Run Your Analysis

The algorithm will automatically:
1. Calculate the center coordinates (mean X, Y) of all hypocenters
2. Query the shapefile to find which polygon contains this center point
3. Extract the stress field parameters from that polygon
4. Use these parameters for all stress calculations

## How It Works

### Center Point Calculation

The algorithm computes the centroid of all hypocenters:

```
center_x = mean(all hypocenter X coordinates)
center_y = mean(all hypocenter Y coordinates)
```

This single representative point is then used to query the stress field.

### Spatial Query

A point-in-polygon test determines which stress field domain contains the center point. If multiple polygons overlap, the first match is used.

### Parameter Extraction

Once a matching polygon is found, the stress field parameters are extracted and applied to **all** events in the analysis, regardless of individual event locations.


## Example Output

When using a shapefile, the console output will show:

```
==================================================
FAULT STRESS ANALYSIS
==================================================
Using spatially-varying stress field from: data_examples/Stressfield/CH_stressfield_Kastrup.shp
Loaded stress field shapefile with 2 polygons
Hypocenter center coordinates: X=2608135.6m, Y=1136737.0m
Stress field parameters from shapefile:
  σ1: trend=301°, plunge=23°
  σ3: trend=43°, plunge=26°
  R=0.35
```

## Dependencies

This feature requires two additional Python packages:

```bash
pip install geopandas shapely
```

These are optional dependencies and only needed if you want to use the shapefile functionality.


## Traditional (Fixed) Parameters

To use fixed stress field parameters, set `use_shapefile` to `false`:

```json
"stress_field": {
  "use_shapefile": false,
  "shapefile_path": null,
  "sigma1_trend_degrees": 301,
  "sigma1_plunge_degrees": 23,
  "sigma3_trend_degrees": 43,
  "sigma3_plunge_degrees": 26,
  "stress_shape_ratio": 0.35
}
```

All existing configurations continue to work without modification.

## Future Enhancements

Possible future improvements:
- Visualization of stress field domains in 3D output
- Interpolation between stress field domains
