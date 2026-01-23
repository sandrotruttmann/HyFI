# Stress Analysis

The **HyFI** stress analysis module analyzes rupture plane orientations in relation to the regional stress field to determine fault slip tendencies and identify optimally oriented structures.

## Overview

This module:
1. Loads stress field information (principal stress orientations and stress ratio)
2. Assigns stress field parameters to each rupture plane based on spatial location
3. Calculates slip tendency and dilation tendency for each rupture plane
4. Identifies rupture planes that are optimally oriented for failure

## Stress Field Input

Stress fields can be defined via:
- **Shapefile**: Spatial polygons with stress parameters (recommended for large-scale analysis)
- **Global parameters**: Single stress field applied to entire study area (works for small-scale analysis where the assumption of a uniform stress field holds)

### Shapefile Format
The shape field needs to contain the following columns:
- `s1_trend`, `s1_plunge`: Maximum principal stress (σ₁) orientation
- `s3_trend`, `s3_plunge`: Minimum principal stress (σ₃) orientation  
- `R`: Stress ratio R = (σ₂-σ₃)/(σ₁-σ₃), range [0,1]

Geometry: Polygons defining spatial extent of each stress domain

## Stress Analysis Metrics

### Normalized Slip Tendency (Ts)
Ratio of shear stress to normal stress on a plane:
$$T_s = \\tau / \\sigma_n$$

Higher values indicate higher likelihood of slip. Range: [0, 1]

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

## Applications

Stress analysis helps to identify:
- Rupture planes most likely to reactivation and/or dilation (i.e. potential fluid transmission) under the given stresses
- Preferentially oriented fault sets

---

Happy fault imaging! 🎉