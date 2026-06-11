# Stress Analysis

The **HyFI** stress analysis module analyzes rupture plane orientations in relation to the regional stress field to determine fault slip tendencies, dilation tendencies, slip vectors, and thus allows to identify optimally oriented structures for reactivation and fluid flow. For details, see the original **HyFI** publication (Truttmann et al., 2023).

---

## Core Concepts

### Fault Instability

**Fault instability** (I) quantifies the susceptibility of a fault plane to rupture (slip) under the ambient stress field. Originally introduced and formalized by Vavryčuk et al. (2013), the instability parameter measures how favorably oriented a fault is with respect to the regional tectonic stress. It ranges from 0 (perfectly stable fault, least likely to fail) to 1 (maximally unstable, most likely to fail).

### Slip Tendency

**Slip tendency** measures the potential of a fault plane to slip under an applied stress field, quantified as the ratio of shear stress to normal stress acting on the plane. Formalized by Morris et al. (1999), slip tendency (T_s) is calculated as:

$$T_s = \frac{\tau_s}{\sigma_n}$$

where $\tau_s$ is the shear stress component resolved onto the fault plane and $\sigma_n$ is the normal stress component perpendicular to the plane. Using the reduced stress tensor, the metric ranges from 0 (no shear, friction-independent stability) to 1 (high potential for slip under Coulomb frictional reactivation theory).

### Dilation Tendency

**Dilation tendency** (T_d) quantifies the potential for volume expansion and fracture opening on a fault plane under a given stress field. Formalized by Ferrill et al. (2020), dilation tendency is calculated as:

$$T_d = \frac{\sigma_1 - \sigma_n}{\sigma_1 - \sigma_3}$$

where $\sigma_1$ is the maximum principal stress, $\sigma_n$ is the normal stress on the plane, and $\sigma_3$ is the minimum principal stress. Using the reduced stress tensor, D ranges from 0 (no dilation potential, volume loss) to 1 (maximum dilation, volume gain). High D values indicate fault planes prone to opening, fracture dilation, and enhanced fluid-flow pathways; low D values indicate planes subject to compaction and fluid sealing. Importantly, dilation tendency captures fault deformation behavior independent of slip tendency: a fault segment may simultaneously experience high slip tendency but low dilation tendency (or vice versa), producing contrasting mechanical responses (pure shear vs. volume-changing deformation modes).

### Slip Vectors (Rake)

**Rake** (also referred to as slip vector) describes the direction of expected slip motion within a fault plane, measured as the angle from the fault strike direction toward the dip direction. Calculated from the stress tensor projection onto the fault plane as:

$$\text{rake} = \arctan2(\tau_d, -\tau_s)$$

where $\tau_d$ is the dip-parallel shear stress component and $\tau_s$ is the strike-parallel shear stress component. Rake ranges from -180° to +180°, with distinct kinematic interpretations.

---

## Computational Workflow

The stress analysis module follows a structured pipeline:

### Step 1: Stress Field Definition

Define the regional stress field using one of two methods:

**Option A: Fixed Stress Field (Global Parameters)**
- Single uniform stress field applied to entire study area
- Parameters:
  - `S1_trend`: Maximum principal stress (σ₁) azimuth direction (0-360°)
  - `S1_plunge`: Maximum principal stress plunge angle (0-90°)
  - `S3_trend`: Minimum principal stress (σ₃) azimuth direction (0-360°)
  - `S3_plunge`: Minimum principal stress plunge angle (0-90°)
  - `stress_R`: Stress shape ratio R = (σ₂-σ₃)/(σ₁-σ₃) (range: 0-1)
  
Use for small (local) study areas where stress field is approximately uniform

**Option B: Spatially-Varying Stress Field (Shapefile)**
- Load stress field polygons from GIS shapefile
- Each polygon contains:
  - Geometry: Polygon boundary defining stress domain extent
  - Attributes: s1_trend, s1_plunge, s3_trend, s3_plunge, R
- Query process:
  1. Calculate center coordinates of the respective earthquake sequence
  2. Locate which polygon contains center point
  3. Extract stress parameters from that polygon
  4. Apply to all events in study area

Use for large (regional) study areas with spatially varying stress (e.g., different tectonic domains)

### Step 2: Stress Tensor Construction

In step 2, the given stress field orientations are converted into vector format. Additionally, σ₂ is derived from the cross product of σ₁ and σ₃, defining the stress tensor. Since slip and dilation tendencies are independent of absolute magnitudes, the reduced stress tensor can be normalized to:
- S1_mag = 1 (normalized)
- S2_mag = 1 - 2R (intermediate, depends on stress shape ratio R)
- S3_mag = -1 (minimum/compression, negative)

The reduced stress tensor is then transformed to geographic xyz coordinate system using a rotation matrix.

### Step 3: Stress-on-Plane Calculation

Project the stress tensor onto each rupture plane to get the resolved stresses:

1. **Project stress tensor onto plane normal**:
   - Stress vector t = Stress_tensor · normal_vector
   
2. **Calculate resolved stresses**:
   - **Normal stress (total)**: σ_n_total = t · n_normal
   - **Effective normal stress**: σ_n_eff = σ_n_total - P_pore
     - Higher values → more compression on plane
     - Lower/negative values → tension/extension
   - **Dip-parallel shear**: τ_d = t · n_dip
   - **Strike-parallel shear**: τ_s = t · n_strike
   - **Total shear stress**: τ = √(τ_d² + τ_s²)
     - Magnitude of shear traction on plane

3. **Calculate rake angle**:
   - rake = arctan(τ_d / -τ_s)
   - Range: -180° to +180°
   - -90°: Normal faulting (down-dip motion)
   - +90°: Reverse faulting (up-dip motion)
   - 0°: Pure sinistral strike-slip motion
   - -180°/+180°: Pure dextral strike-slip motion

### Step 4: Fault Instability Calculation

Calculate the fault instability I according to Vavryčuk et al. (2013):
   - I = (τ - μ(σ_n_eff - 1)) / (μ + √(1 + μ²))
   - Where μ = friction coefficient
   - Measures likelihood of fault instability/rupture

### Step 5: Slip & Dilation Tendency Calculation

Calculate normalized slip and dilation tendency metrics using the reduced stress tensor:

1. **Slip Tendency (T_s)** (Morris et al., 1996; Lisle and Srivastava, 2004):
   - T_s = τ / σ_n_eff
   - Ratio of shear stress to effective normal stress
   - Range: 0 to 1
   - Higher values → higher likelihood of slip
   - Physical meaning: Planes with high T_s are more prone to slip

2. **Dilation Tendency (T_d)** (Ferrill et al., 2020):
   - T_d = (σ₁ - σ_n) / (σ₁ - σ₃)
   - Measures potential for opening/dilation
   - Range: 0 to 1
   - Higher values → more tensile stress on plane
   - Physical meaning: Planes with high T_d can open for fluid flow

### Step 6: Output Generation

The stress analysis results are finally added to the full DataFrame `df_hyfi`.

---

## Main Outputs

### DataFrame Columns Added

#### Rupture Plane (HyFI-fitted from hypocenter distribution)

**Stress Magnitude Parameters**:
- **`Sn_eff`**: Effective normal stress on fault plane (normalized units)
- **`Tau`**: Resolved shear stress magnitude on fault plane (normalized units)

**Motion Parameters**:
- **`rake`**: Predicted rake angle of fault motion (-180° (dextral) to -90° (normal) to 0° (sinistral) to +90° (reverse) to +180° (dextral))

**Tendency Parameters** (0-1 scale):
- **`instab`**: Instability index
- **`sliptend`**: Slip tendency (T_s)
- **`dilatend`**: Dilation tendency (T_d)

#### Focal Mechanism Nodal Planes (NP1 and NP2)

When focal mechanism data is available, the same tendency metrics are calculated for both nodal planes. These are computed during the final database export step and are written to `HyFI_database_focals.csv` (not the per-sequence `HyFI_results.csv`), allowing direct comparison of both nodal planes across the full focal mechanism catalog.

**Nodal Plane 1 (NP1)** — from `Strike1`/`Dip1`:
- **`NP1_instability_index`**: Fault instability on NP1
- **`NP1_slip_tendency`**: Slip tendency on NP1 (T_s)
- **`NP1_dilation_tendency`**: Dilation tendency on NP1 (T_d)

**Nodal Plane 2 (NP2)** — from `Strike2`/`Dip2`:
- **`NP2_instability_index`**: Fault instability on NP2
- **`NP2_slip_tendency`**: Slip tendency on NP2 (T_s)
- **`NP2_dilation_tendency`**: Dilation tendency on NP2 (T_d)

All tendency parameters range from 0 to 1, with interpretation identical to the rupture plane metrics. This enables identification of which nodal plane is more favorably oriented for slip and dilation under the regional stress field.

---

## References

- Ferrill, D. A., Smart, K. J., Morris, A. P. (2020). Fault failure modes, deformation mechanisms, dilation tendency, slip tendency, and conduits v. seals, Integrated Fault Seal Analysis, S. R. Ogilvie, S. J. Dee, R. W. Wilson, W. R. Bailey.  https://doi.org/10.1144/SP496-2019-7 

- Lisle, R. J., & Srivastava, D. C. (2004). Test of the frictional reactivation theory for faults and validity of fault-slip analysis. Geology, 32(7), 569-572. https://doi.org/10.1130/G20408.1

- Morris, A., Ferrill, D. A., & Henderson, D. B. (1996). Slip-tendency analysis and fault reactivation. Geology, 24(3), 275-278. https://doi.org/10.1130/0091-7613(1996)024<0275:STAAFR>2.3.CO;2

- Vavryčuk, V., Bouchaala, F., & Fischer, T. (2013). High-resolution fault image from accurate locations and focal mechanisms of the 2008 swarm earthquakes in West Bohemia, Czech Republic. Tectonophysics, 590, 189-195. https://doi.org/10.1016/j.tecto.2013.01.025


---

Happy fault imaging! 🎉