# Model Validation

The model validation module compares **HyFI** rupture plane estimates with independent focal mechanism solutions to assess reconstruction accuracy.

## Overview

This module:
1. Merges focal mechanism data with hypocenter data based on their IDs (requires that both hypocenter and focal mechanism catalogs share the same IDs for the respective events)
2. Compares HyFI-derived rupture planes with focal mechanism nodal planes
3. Determines the most likely active plane using angular differences
4. Calculates validation statistics and quality metrics

## Active Plane Determination

For each event with a focal mechanism:
1. Calculate angular differences between HyFI rupture plane and both nodal planes
2. Select the nodal plane with smallest angular difference as "active plane"
3. Compute angular misfit ε (misfit angle in degrees) 

## Outputs

The validation produces:
- **active_plane_determination_summary.csv**: Angular differences and active plane selections
- **active_plane_statistics.txt**: Summary statistics of validation results
- Updated `HyFI_results.csv` with focal mechanism parameters and angular misfits

---

Happy fault imaging! 🎉