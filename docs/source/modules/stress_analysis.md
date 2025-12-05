# Stress Analysis

The stress analysis module analyzes fault orientations in relation to the regional stress field to determine fault slip tendencies and identify optimally oriented structures.

## Overview

This module:
1. Loads stress field information (principal stress orientations and stress ratio)
2. Assigns stress field parameters to each fault based on spatial location
3. Calculates slip tendency and dilation tendency for each fault plane
4. Identifies faults that are optimally oriented for failure

## Stress Field Input

Stress fields can be defined via:
- **Shapefile**: Spatial polygons with stress parameters (recommended for heterogeneous stress)
- **Global parameters**: Single stress field applied to entire study area

### Shapefile Format
Required columns:
- `s1_trend`, `s1_plunge`: Maximum principal stress (σ₁) orientation
- `s3_trend`, `s3_plunge`: Minimum principal stress (σ₃) orientation  
- `R`: Stress ratio R = (σ₂-σ₃)/(σ₁-σ₃), range [0,1]

Geometry: Polygons defining spatial extent of each stress domain

## Stress Analysis Metrics

### Slip Tendency (Ts)
Ratio of shear stress to normal stress on a plane:
$$T_s = \\tau / \\sigma_n$$

Higher values indicate higher likelihood of slip. Range: [0, ~1]

### Dilation Tendency (Td)  
Normalized tensile stress component:
$$T_d = (\\sigma_1 - \\sigma_n) / (\\sigma_1 - \\sigma_3)$$

Higher values indicate potential for opening/dilation. Range: [0, 1]

## Key Parameters

- `stress_bool`: Enable/disable stress analysis
- `stress_shapefile`: Path to stress field shapefile
- `s1_trend`, `s1_plunge`: Global σ₁ orientation (if no shapefile)
- `s3_trend`, `s3_plunge`: Global σ₃ orientation
- `stress_ratio`: Global R value (default: 0.5)

## Outputs

Stress analysis adds columns to `HyFI_results.csv`:
- `slip_tendency`: Calculated slip tendency (0-1)
- `dilation_tendency`: Calculated dilation tendency (0-1)
- `stress_regime`: Assigned stress domain ID
- `s1_trend`, `s1_plunge`, `s3_trend`, `s3_plunge`, `R`: Applied stress parameters

## Applications

Stress analysis helps identify:
- Faults most likely to rupture under current stress
- Preferentially oriented fault sets
- Spatial variations in fault activity potential
- Relation between fault geometry and stress field
