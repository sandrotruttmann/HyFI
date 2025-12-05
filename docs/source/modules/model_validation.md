# Model Validation

The model validation module compares HyFI fault plane estimates with independent focal mechanism solutions to assess reconstruction accuracy.

## Overview

This module:
1. Merges focal mechanism data with hypocenter data
2. Compares HyFI-derived fault planes with focal mechanism nodal planes
3. Determines the most likely active plane using angular differences
4. Calculates validation statistics and quality metrics

## Focal Mechanism Integration

The module can match focal mechanisms to hypocenters using:
- **Magnitude matching**: Tolerates small differences (configurable threshold)
- **Location matching**: Uses spatial proximity (configurable distance threshold)
- **Manual matching**: Direct ID-based matching for specific events

## Active Plane Determination

For each event with a focal mechanism:
1. Calculate angular differences between HyFI plane and both nodal planes
2. Select nodal plane with smallest angular difference as "active plane"
3. Compute angular misfit metrics

## Key Parameters

- `validation_bool`: Enable/disable model validation
- `foc_file`: Path to focal mechanism CSV file
- `foc_sep`: Separator character in focal mechanism file
- `foc_mag_check`: Enable magnitude-based matching
- `foc_max_mag_diff`: Maximum magnitude difference for matching (e.g., 0.2)
- `foc_loc_check`: Enable location-based matching
- `foc_max_dist_km`: Maximum distance in km for location matching (e.g., 1.0)

## Focal Mechanism File Format

Required columns in focal mechanism CSV:
- Event identifiers (ID, magnitude, location)
- Nodal plane parameters: `strike1`, `dip1`, `rake1`, `strike2`, `dip2`, `rake2`
- Alternatively: `azimuth`, `plunge`, `slip` (P/T/B axes)

## Outputs

The validation produces:
- **active_plane_determination_summary.csv**: Angular differences and active plane selections
- **active_plane_statistics.txt**: Summary statistics of validation results
- Updated `HyFI_results.csv` with focal mechanism parameters and angular misfits

## Quality Metrics

Statistics computed include:
- Mean/median angular differences between HyFI and active planes
- Percentage of events with small misfits (e.g., < 15°, < 30°)
- Distribution of angular differences
