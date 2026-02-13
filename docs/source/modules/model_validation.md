# Model Validation

The model validation module compares **HyFI** rupture plane estimates with independent focal mechanism solutions to assess reconstruction accuracy and determine active fault planes based on hypocenter distribution constraints. For details, see the original **HyFI** publication (Truttmann et al., 2023).

---

## Core Concepts

### Angular Differences (ε) as Quality Measure
**HyFI** calculates the angle between the computed rupture plane and the active nodal plane of the focal mechanism.

**Physical Interpretation**:
- 0°: Perfect alignment (computed plane ≡ focal mechanism plane)
- < 15°: Excellent agreement (typical for well-constrained cases)
- 15-30°: Good agreement (acceptable for seismic data quality)
- 30-45°: Fair agreement (visible discrepancy)
- > 45°: Poor agreement (significant misalignment)

### Pre-specified vs. Auto-determined Planes
**HyFI** can be used to verify pre-specified nodal planes (A=1 or A=2) and automatically determines unspecified nodal planes (A=0 or A=NaN) based on the fit with the calculated rupture plane from the fault network module.

Note that the pre-specified nodal planes are always preserved!

---

## Computational Workflow

The validation module is executed according to the following structured pipeline:

### Step 1: Data Merging - Focal Mechanism Integration

Focal mechanism data is merged with hypocenter data using two strategies (tried in order):

**Strategy A: ID-Based Matching** (Primary)
- Match events by event ID across catalogs
- Requires identical ID formatting in both files
- Most reliable when ID numbers are unique and consistent
- Fastest approach - direct one-to-one correspondence

**Strategy B: Temporal Matching** (Fallback)
- If ID matching fails, use `merge_asof` on datetime
- Matches events within 60-second tolerance window
- Sorted by date before merging
- More robust for catalogs with ID formatting differences or with unconsistent/missing catalog IDs

The algorithm also performs two **Quality Control Checks** (Optional, configurable):

1. **Magnitude Consistency Check**:
   - Compare hypocenter magnitude (MAG) with focal mechanism magnitude
   - Remove focal data if |MAG_hypo - MAG_focal| > threshold (default: 0.2)
   - Reports events with significant magnitude mismatches
   - Protects against erroneous cross-catalog associations

2. **Location Consistency Check**:
   - Compare hypocenter location (X, Y, Z) with focal mechanism location
   - Depth threshold: km units (default: 1.0 km)
   - Latitude/Longitude: converted from km to degrees (~0.009°/km)
   - Reports mismatches but does NOT remove data (just warnings)
   - Identifies potential duplicate events in different catalogs

### Step 2: Normal Vector Conversion

In step 2, both nodal planes are converted from azimuth/dip to normal vectors. This results in normal vectors for both focal mechanism nodal planes

### Step 3: Angular Difference Calculation

For each event with both a computed rupture plane and a focal mechanism, the angular difference from the rupture plane to both nodal planes (1 and 2) is calculated:

1. **Plane 1 angular difference**:
   - angle1 = angle_between(nor_computed, nor_focal_plane1)
   - Ensure acute angle: angle1 = min(angle1, 180 - angle1)

2. **Plane 2 angular difference**:
   - angle2 = angle_between(nor_computed, nor_focal_plane2)
   - Ensure acute angle: angle2 = min(angle2, 180 - angle2)

**Geometric meaning**: Angular difference measures the angle between the normal vectors and thus how well the computed rupture plane aligns with each focal mechanism nodal plane

### Step 4: Preferred Plane Selection
In the next step, a two-tier selection process to define the active nodal plane is exexuted:

**Tier 1: Active Plane Indicator (if available)**
- Check column A (specifies the known 1active plane) from the focal mechanism catalog: values 1, 2 indicate pre-specified active plane
- If A = 1: Use angle1 (Strike1, Dip1, Rake1) as final angular difference (epsilon)
- If A = 2: Use angle2 (Strike2, Dip2, Rake2) as final angular difference (epsilon)
- If A = 0: Fall through to Tier 2 (unknown nodal plane, use geometry)
- If A = NaN: Fall through to Tier 2 (no pre-specification, use geometry)

**Tier 2: Geometric Selection**
- Compare: angle1 vs. angle2
- Choose plane with MINIMUM angular difference (epsilon = min(angle1, angle2))

**Rationale**: 
- Pre-specified active planes (A=1 or A=2) take priority and are always maintained
- For unknown planes (A=0 or A=NaN), the algorithm selects the nodal plane based on the best geometric fit to the computed rupture plane

**Output columns**:
- `epsilon`: Angular difference of the preferred focal plane in degrees (0-90°)
- `pref_foc`: **HyFI**-based preferred focal plane (1 or 2)
- `plane_determination_method`: Categorical description
  - "Pre-specified (A=1 or A=2)"
  - "Newly determined (A=0, geometric selection)"
  - "Newly determined (no A, geometric selection)"
  - "Not determined (A specified but no rupture plane)"
  - "Not determined (no computed rupture plane)"

### Step 5: Reporting
Finally, the validation module generates comprehensive summary files on the preferred plane determination (see below).

---

## Main Outputs

**Active Plane Statistics: `active_plane_statistics.txt`**

This file includes the statistics summary in a human-readable format for quick understanding of validation results quality.
- Total events with focal mechanisms
- Breakdown by determination method with counts
- For newly determined planes: selection distribution (plane 1 vs. 2)
- Angular difference statistics: mean, median, min, max
- For pre-specified planes: similar angular difference statistics
- For not-determined events: reasons and counts
- Interpretation guidance


**Active Plane Determination Summary: `active_plane_determination_summary.csv`**

Detailed row-by-row documentation including:
- Event ID, Date, Location (X, Y, Z), Magnitude
- Focal mechanism data: Strike1, Dip1, Rake1, Strike2, Dip2, Rake2
- Focal mechanism active plane indicator: A
- Computed fault plane: rupt_plane_azi, rupt_plane_dip
- Validation results: pref_foc, epsilon
- Preferred strike/dip/rake extracted from selected plane
- Determination method: How was pref_foc determined?

Sorted by:
1. Determination method (pre-specified → newly determined → not determined)
2. Angular difference (best matches first)

---

## References

-  Truttmann, S., Diehl, T., & Herwegh, M. (2023). Hypocenter-based 3D imaging of active faults: Method and applications in the Southwestern Swiss Alps. Journal of Geophysical Research: Solid Earth, 128, e2023JB026352. https://doi.org/10.1029/2023JB026352 

---

Happy fault imaging! 🎉