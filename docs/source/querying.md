# HyFI Database Querying

This guide explains how to query **HyFI** CSV output files using DuckDB and SQL. DuckDB is a fast, embeddable analytical database that can query CSV files directly without loading them into memory, making it ideal for analyzing large **HyFI** datasets.

---

## Why DuckDB?

✅ **No setup required** - Just `pip install duckdb`  
✅ **Query CSVs directly** - No import or data loading needed  
✅ **Fast** - Processes gigabytes of data in seconds  
✅ **SQL** - Familiar, powerful query language  
✅ **Python native** - Works seamlessly with pandas DataFrames  
✅ **Multi-file queries** - Analyze across all projects at once  

---

## Overview

**HyFI** produces several CSV files containing earthquake analysis results:

| CSV File | Description | Location |
|----------|-------------|----------|
| **HyFI_results.csv** | Main results with fault parameters for each event | `output_dir/` |
| **active_plane_determination_summary.csv** | Focal mechanism validation details | `output_dir/` |
| **interpolated_faults_summary.csv** | Interpolated fault surface metadata | `output_dir/` |
| **hypocenters.csv** | Original hypocenter data | `output_dir/csv_export/` |
| **enhanced_pointcloud.csv** | Expanded point cloud from rupture surfaces | `output_dir/csv_export/` |

---

## Installation

```bash
pip install duckdb
```

---

## Query Methods

HyFI provides three ways to query results:

1. **HyFI Query CLI** - Command-line tool for common queries (easiest)
2. **Python Query API** - Pre-built query functions in Python
3. **Custom SQL** - Direct DuckDB SQL queries (most flexible)

---

## HyFI Query CLI

The HyFI query CLI provides easy command-line access to common database queries for multi-sequence workflows.

### Installation

The CLI is included with HyFI:

```bash
pip install -e .  # If installing from source
```

### Basic Usage

All commands require specifying the database directory:

```bash
python -m hyfi.query.cli --database-dir output_SECOS_VS/HyFI_Database <command>
```

### Available Commands

#### `overview` - Fault Systems Overview

Show overview of all identified fault systems:

```bash
python -m hyfi.query.cli --database-dir output_SECOS_VS/HyFI_Database overview

# With minimum event filter
python -m hyfi.query.cli --database-dir output_SECOS_VS/HyFI_Database overview --min-events 20
```

**Output:**
```
Fault Systems Overview:
fault_id  sequence_label  n_events  azimuth  dip  area_km2  max_mag  instability
A1        Class_A         145       85.2     65   2.34      3.5      0.456
A2        Class_A         98        112.5    58   1.89      3.2      0.423
B1        Class_B         45        95.8     72   0.85      2.8      0.512
```

#### `summary` - Database Summary

Get overall statistics:

```bash
python -m hyfi.query.cli --database-dir output_SECOS_VS/HyFI_Database summary
```

**Output:**
```
Database Summary:
total_fault_systems  total_events_in_faults  avg_events_per_fault  total_area_km2  avg_instability  max_potential_mag
15                   1245                    83.0                  25.6            0.445            4.2
```

#### `instability` - High Instability Faults

Show faults with highest instability indices:

```bash
python -m hyfi.query.cli --database-dir output_SECOS_VS/HyFI_Database instability --limit 5
```

**Output:**
```
Top 5 High Instability Fault Systems:
fault_id  n_events  instability  slip_tendency  dilation_tendency  mesh_instability
B3        52        0.612        0.645          0.234              0.598
A5        78        0.587        0.612          0.198              0.571
```

#### `spatial` - Spatial Query

Find events within radius of coordinates:

```bash
python -m hyfi.query.cli --database-dir output_SECOS_VS/HyFI_Database spatial \
    --x 592500 --y 130500 --radius 2.0 --limit 20
```

**Output:**
```
Events within 2.0km of (592500.0, 130500.0):
ID           fault_id  magnitude  x       y       z      distance_km
EV20230415   A1        2.8        592750  130250  -5200  0.35
EV20230418   A1        2.5        593100  130800  -4800  0.82
```

#### `fault-detail` - Detailed Fault Information

Get comprehensive details for a specific fault:

```bash
python -m hyfi.query.cli --database-dir output_SECOS_VS/HyFI_Database fault-detail --fault-id A1
```

**Output:**
```
Fault System A1 Details:

Metadata:
fault_id  sequence_label  n_events  rupture_mean_azimuth  rupture_mean_dip  mesh_area_m2
A1        Class_A         145       85.2                  65.3              2340000

Events (top 10 by magnitude):
ID           MAG   X       Y       Z      rupture_plane_azimuth  instability_index
EV20230415   3.5   592750  130250  -5200  87.5                   0.523
EV20230418   3.2   593100  130800  -4800  82.1                   0.489
...

Focal Mechanisms:
ID           Strike1  Dip1  Rake1  Strike2  Dip2  Rake2  A
EV20230415   85       67    -15    178      76    -157   1
```

#### `list-faults` - List All Fault IDs

List all available fault systems across tables:

```bash
python -m hyfi.query.cli --database-dir output_SECOS_VS/HyFI_Database list-faults
```

**Output:**
```
Fault System IDs by table:

Metadata table (12 faults):
fault_id  n_events
A1        145
A2        98
A3        76
...

Hypocenters table (12 faults):
fault_id  event_count
A1        145
A2        98
...
```

#### `query` - Custom SQL Query

Execute custom SQL queries:

```bash
python -m hyfi.query.cli --database-dir output_SECOS_VS/HyFI_Database query \
    --sql "SELECT COUNT(*) as total FROM hypocenters WHERE MAG > 3.0"
```

#### `tables` - Show Table Structure

List loaded tables and their structure:

```bash
python -m hyfi.query.cli --database-dir output_SECOS_VS/HyFI_Database tables
```

**Output:**
```
Loaded Tables:
✓ metadata
column_name              column_type
fault_id                 VARCHAR
sequence_label           VARCHAR
n_events                 INTEGER
...

✓ hypocenters
column_name              column_type
ID                       VARCHAR
fault_id                 VARCHAR
MAG                      DOUBLE
...
```


## Quick Start with Direct SQL Queries

For complete flexibility, query CSV files directly with DuckDB SQL:

### Basic Query

```python
import duckdb

# Query CSV file directly
result = duckdb.query("""
    SELECT 
        COUNT(*) as total_events,
        ROUND(AVG(MAG), 2) as avg_magnitude,
        ROUND(MAX(MAG), 2) as max_magnitude
    FROM read_csv_auto('output_A0/HyFI_results.csv')
""").df()

print(result)
```


--- ## CSV File Structure

Understanding the column structure helps write effective queries.

### HyFI_results.csv Columns

**Event Information:**
- `ID`, `Date` - Event identifier and timestamp
- `X`, `Y`, `Z` - Local coordinates (meters)
- `LAT`, `LON`, `DEPTH` - Geographic coordinates
- `MAG` - Magnitude
- `EX`, `EY`, `EZ` - Location uncertainties (meters)

**Focal Mechanism Data (if available):**
- `Strike1`, `Dip1`, `Rake1` - First nodal plane
- `Strike2`, `Dip2`, `Rake2` - Second nodal plane
- `A` - Active plane indicator (1, 2, or 0=unknown)
- `pref_foc` - Preferred focal plane (1.0, 2.0, or -1=not determined)
- `epsilon` - Angular misfit between focal mechanism and computed plane (degrees)

**Computed Rupture Plane:**
- `mean_azi` - Fault strike azimuth (0-360°)
- `mean_dip` - Fault dip angle (0-90°)
- `nor_x_mean`, `nor_y_mean`, `nor_z_mean` - Fault normal vector
- `nr_fits` - Number of successful fits
- `kappa` - Concentration parameter (higher = more consistent orientation)
- `beta`, `lambda_2_3` - Eigenvalue ratios (quality metrics)

**Rupture Geometry:**
- `R` - Number of events within search radius
- `N` - Total neighbors considered
- `R/N` - Neighbor ratio
- `rupt_area` - Estimated rupture area (m²)
- `rupt_r` - Estimated rupture radius (m)

**Clustering Results:**
- `clust_labels` - Orientation cluster assignment
- `spatial_cluster` - Spatial sub-cluster assignment
- `final_cluster_id` - Combined cluster identifier (e.g., "F1", "F2")
- `class` - Final classification label

**Stress Analysis (if enabled):**
- `Sn_eff` - Effective normal stress (MPa)
- `Tau` - Shear stress (MPa)
- `rake` - Calculated slip rake angle
- `I` - Instability index (Tau / Sn_eff)
- `sliptend` - Slip tendency
- `dilatend` - Dilation tendency

**Special Values:**
- `-999.0` - Parameter not computed or not applicable
- `-1` - Event excluded from analysis (outlier)
- `NULL` - Missing data

---

### active_plane_determination_summary.csv Columns

- `ID`, `Date`, `X`, `Y`, `Z`, `MAG` - Event identifiers
- `Strike1`, `Dip1`, `Rake1`, `Strike2`, `Dip2`, `Rake2` - Nodal planes
- `A` - Original active plane indicator
- `pref_foc` - Selected preferred plane (1.0 or 2.0)
- `epsilon` - Angular difference (degrees)
- `plane_determination_method` - How the plane was selected
- `mean_azi`, `mean_dip` - Fitted rupture plane orientation

**Plane Determination Methods:**
- `"Pre-specified (A=1 or A=2)"` - User/catalog provided
- `"Newly determined (A=0, geometric selection)"` - Algorithm selected
- `"Not determined (A specified but no rupture plane)"` - Fitting failed
- `"Not determined (no computed rupture plane)"` - Insufficient neighbors

**Quality Interpretation:**
- `epsilon < 15°` - Excellent agreement
- `epsilon < 30°` - Good agreement
- `epsilon > 45°` - Poor agreement

---

## SQL Query Examples

### Example 1: Basic Statistics

```sql
SELECT 
    COUNT(*) as total_events,
    ROUND(AVG(MAG), 2) as avg_magnitude,
    ROUND(MIN(MAG), 2) as min_magnitude,
    ROUND(MAX(MAG), 2) as max_magnitude,
    ROUND(AVG(Z), 0) as avg_depth_m,
    COUNT(*) FILTER (WHERE mean_azi IS NOT NULL) as events_with_planes,
    COUNT(*) FILTER (WHERE epsilon IS NOT NULL) as events_with_focal
FROM read_csv_auto('output_A0/HyFI_results.csv')
```

---

### Example 2: Filter by Quality Metrics

```sql
-- High-quality events with good focal mechanism fit
SELECT 
    ID, Date, MAG, mean_azi, mean_dip, epsilon, kappa, beta
FROM read_csv_auto('output_A0/HyFI_results.csv')
WHERE 
    epsilon < 30                -- Good fit to focal mechanism
    AND kappa > 5               -- High concentration
    AND beta < 0.5              -- Planar geometry
    AND nr_fits >= 5            -- Multiple successful fits
ORDER BY epsilon
```

---

### Example 3: Fault Orientation Analysis

```sql
-- Fault strike distribution
SELECT 
    CASE 
        WHEN mean_azi >= 0 AND mean_azi < 30 THEN 'N-NE (0-30°)'
        WHEN mean_azi >= 30 AND mean_azi < 60 THEN 'NE-E (30-60°)'
        WHEN mean_azi >= 60 AND mean_azi < 90 THEN 'E-SE (60-90°)'
        WHEN mean_azi >= 90 AND mean_azi < 120 THEN 'SE-S (90-120°)'
        WHEN mean_azi >= 120 AND mean_azi < 150 THEN 'S-SW (120-150°)'
        WHEN mean_azi >= 150 AND mean_azi < 180 THEN 'SW-W (150-180°)'
        ELSE 'Other'
    END as strike_class,
    COUNT(*) as num_events,
    ROUND(AVG(MAG), 2) as avg_magnitude
FROM read_csv_auto('output_A0/HyFI_results.csv')
WHERE mean_azi IS NOT NULL
GROUP BY strike_class
ORDER BY strike_class
```

```sql
-- Fault dip classification
SELECT 
    CASE 
        WHEN mean_dip < 30 THEN 'Shallow (<30°)'
        WHEN mean_dip >= 30 AND mean_dip < 60 THEN 'Moderate (30-60°)'
        WHEN mean_dip >= 60 THEN 'Steep (>60°)'
    END as dip_class,
    COUNT(*) as num_events,
    ROUND(AVG(kappa), 2) as avg_concentration
FROM read_csv_auto('output_A0/HyFI_results.csv')
WHERE mean_dip IS NOT NULL
GROUP BY dip_class
ORDER BY dip_class
```

---

### Example 4: Temporal Analysis

```sql
-- Events by month
SELECT 
    DATE_TRUNC('month', CAST(Date AS TIMESTAMP)) as month,
    COUNT(*) as num_events,
    ROUND(AVG(MAG), 2) as avg_magnitude,
    ROUND(MAX(MAG), 2) as max_magnitude
FROM read_csv_auto('output_A0/HyFI_results.csv')
GROUP BY month
ORDER BY month
```

```sql
-- Rolling statistics
SELECT 
    Date,
    MAG,
    AVG(MAG) OVER (
        ORDER BY Date 
        ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
    ) as rolling_avg_mag,
    COUNT(*) OVER (
        ORDER BY Date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as cumulative_count
FROM read_csv_auto('output_A0/HyFI_results.csv')
ORDER BY Date
LIMIT 20
```

---

### Example 5: Stress Analysis Results

```sql
-- High instability faults
SELECT 
    ID, Date, MAG, mean_azi, mean_dip,
    ROUND(sliptend, 3) as slip_tendency,
    ROUND(I, 3) as instability_index,
    ROUND(Sn_eff, 1) as normal_stress_MPa,
    ROUND(Tau, 1) as shear_stress_MPa
FROM read_csv_auto('output_A0/HyFI_results.csv')
WHERE 
    sliptend IS NOT NULL 
    AND sliptend != -999.0
    AND (I > 0.6 OR sliptend > 0.6)
ORDER BY instability_index DESC
```

```sql
-- Stress analysis statistics by fault system
SELECT 
    final_cluster_id as fault_system,
    COUNT(*) as num_events,
    ROUND(AVG(sliptend), 3) as avg_slip_tendency,
    ROUND(AVG(I), 3) as avg_instability,
    ROUND(MAX(I), 3) as max_instability
FROM read_csv_auto('output_A0/HyFI_results.csv')
WHERE sliptend IS NOT NULL AND sliptend != -999.0
GROUP BY fault_system
ORDER BY avg_instability DESC
```

---

### Example 6: Spatial Queries

```sql
-- Events in a specific region (bounding box)
SELECT 
    ID, Date, X, Y, Z, MAG, mean_azi, mean_dip
FROM read_csv_auto('output_A0/HyFI_results.csv')
WHERE 
    X BETWEEN 590000 AND 595000
    AND Y BETWEEN 128000 AND 133000
    AND Z BETWEEN -6000 AND -4000
ORDER BY MAG DESC
```

```sql
-- Events within radius of a point (using Euclidean distance)
SELECT 
    ID, Date, MAG,
    ROUND(SQRT(
        POW(X - 592500, 2) + 
        POW(Y - 130500, 2) + 
        POW(Z - (-5000), 2)
    ), 1) as distance_m
FROM read_csv_auto('output_A0/HyFI_results.csv')
WHERE SQRT(
    POW(X - 592500, 2) + 
    POW(Y - 130500, 2) + 
    POW(Z - (-5000), 2)
) <= 1000  -- Within 1000 meters
ORDER BY distance_m
```

---

### Example 7: Focal Mechanism Validation

```sql
-- Validation quality statistics
SELECT 
    CASE 
        WHEN epsilon < 15 THEN 'Excellent (<15°)'
        WHEN epsilon >= 15 AND epsilon < 30 THEN 'Good (15-30°)'
        WHEN epsilon >= 30 AND epsilon < 45 THEN 'Fair (30-45°)'
        ELSE 'Poor (≥45°)'
    END as quality,
    COUNT(*) as num_events,
    ROUND(AVG(kappa), 2) as avg_concentration
FROM read_csv_auto('output_A0/active_plane_determination_summary.csv')
WHERE epsilon IS NOT NULL
GROUP BY quality
ORDER BY 
    CASE quality
        WHEN 'Excellent (<15°)' THEN 1
        WHEN 'Good (15-30°)' THEN 2
        WHEN 'Fair (30-45°)' THEN 3
        ELSE 4
    END
```

```sql
-- Events by determination method
SELECT 
    plane_determination_method,
    COUNT(*) as num_events,
    ROUND(AVG(epsilon), 1) as avg_misfit_degrees,
    ROUND(MIN(epsilon), 1) as min_misfit,
    ROUND(MAX(epsilon), 1) as max_misfit
FROM read_csv_auto('output_A0/active_plane_determination_summary.csv')
WHERE epsilon IS NOT NULL
GROUP BY plane_determination_method
ORDER BY avg_misfit_degrees
```

---

### Example 8: Fault System Classification

```sql
-- Events by fault system
SELECT 
    final_cluster_id as fault_system,
    COUNT(*) as num_events,
    ROUND(AVG(MAG), 2) as avg_magnitude,
    ROUND(MAX(MAG), 2) as max_magnitude,
    ROUND(AVG(mean_azi), 1) as avg_strike,
    ROUND(AVG(mean_dip), 1) as avg_dip,
    ROUND(AVG(kappa), 2) as avg_concentration
FROM read_csv_auto('output_A0/HyFI_results.csv')
WHERE final_cluster_id IS NOT NULL
GROUP BY fault_system
ORDER BY num_events DESC
```

---

### Example 9: Multi-Project Analysis

```python
import duckdb
from pathlib import Path

# Query across multiple output directories
output_dirs = list(Path('.').glob('output_*'))

results = []
for output_dir in output_dirs:
    csv_file = output_dir / 'HyFI_results.csv'
    if csv_file.exists():
        result = duckdb.query(f"""
            SELECT 
                '{output_dir.name}' as project,
                COUNT(*) as events,
                ROUND(AVG(MAG), 2) as avg_mag,
                ROUND(MAX(MAG), 2) as max_mag,
                COUNT(*) FILTER (WHERE mean_azi IS NOT NULL) as with_planes,
                COUNT(*) FILTER (WHERE epsilon < 30) as good_fit
            FROM read_csv_auto('{csv_file}')
        """).df()
        results.append(result)

# Combine all results
import pandas as pd
df_all = pd.concat(results, ignore_index=True)
print(df_all)
```

---

### Example 10: Magnitude Binning and Statistics

```sql
-- Events by magnitude class
SELECT 
    CASE 
        WHEN MAG < 1.0 THEN 'M < 1.0'
        WHEN MAG >= 1.0 AND MAG < 2.0 THEN 'M 1.0-2.0'
        WHEN MAG >= 2.0 AND MAG < 3.0 THEN 'M 2.0-3.0'
        WHEN MAG >= 3.0 AND MAG < 4.0 THEN 'M 3.0-4.0'
        ELSE 'M ≥ 4.0'
    END as mag_class,
    COUNT(*) as num_events,
    COUNT(*) FILTER (WHERE epsilon IS NOT NULL AND epsilon < 30) as good_fit,
    ROUND(AVG(kappa), 2) as avg_kappa,
    ROUND(AVG(beta), 3) as avg_beta,
    ROUND(AVG(rupt_area), 0) as avg_rupture_area_m2
FROM read_csv_auto('output_A0/HyFI_results.csv')
WHERE mean_azi IS NOT NULL
GROUP BY mag_class
ORDER BY 
    CASE mag_class
        WHEN 'M < 1.0' THEN 1
        WHEN 'M 1.0-2.0' THEN 2
        WHEN 'M 2.0-3.0' THEN 3
        WHEN 'M 3.0-4.0' THEN 4
        ELSE 5
    END
```

---

### Example 11: Advanced Aggregations with CASE

```sql
-- Comprehensive fault characterization
SELECT 
    final_cluster_id as fault_system,
    COUNT(*) as total_events,
    
    -- Magnitude stats
    ROUND(MIN(MAG), 2) as min_mag,
    ROUND(MAX(MAG), 2) as max_mag,
    ROUND(AVG(MAG), 2) as avg_mag,
    
    -- Orientation stats  
    ROUND(AVG(mean_azi), 1) as avg_strike,
    ROUND(AVG(mean_dip), 1) as avg_dip,
    
    -- Quality metrics
    ROUND(AVG(kappa), 2) as avg_kappa,
    ROUND(AVG(beta), 3) as avg_beta,
    COUNT(*) FILTER (WHERE epsilon < 30) as good_focal_fit,
    
    -- Stress indicators
    ROUND(AVG(sliptend), 3) as avg_slip_tendency,
    COUNT(*) FILTER (WHERE sliptend > 0.6) as high_slip_tendency,
    
    -- Spatial extent
    ROUND(MAX(Z) - MIN(Z), 0) as depth_range_m,
    ROUND(SQRT(POW(MAX(X) - MIN(X), 2) + POW(MAX(Y) - MIN(Y), 2)), 0) as horizontal_extent_m
    
FROM read_csv_auto('output_A0/HyFI_results.csv')
WHERE final_cluster_id IS NOT NULL
GROUP BY fault_system
ORDER BY total_events DESC
```

---

### Example 12: JOIN Multiple CSV Files

```sql
-- Join main results with active plane validation
SELECT 
    h.ID,
    h.Date,
    h.MAG,
    h.mean_azi,
    h.mean_dip,
    h.kappa,
    a.plane_determination_method,
    a.epsilon,
    CASE 
        WHEN a.epsilon < 15 THEN 'Excellent'
        WHEN a.epsilon < 30 THEN 'Good'
        WHEN a.epsilon < 45 THEN 'Fair'
        ELSE 'Poor'
    END as fit_quality
FROM read_csv_auto('output_A0/HyFI_results.csv') h
INNER JOIN read_csv_auto('output_A0/active_plane_determination_summary.csv') a
    ON h.ID = a.ID
WHERE h.mean_azi IS NOT NULL
ORDER BY a.epsilon
```

---

## Advanced Techniques

### Using CTEs (Common Table Expressions)

```sql
-- Calculate statistics and then filter
WITH fault_stats AS (
    SELECT 
        final_cluster_id,
        COUNT(*) as num_events,
        AVG(MAG) as avg_mag,
        AVG(kappa) as avg_kappa
    FROM read_csv_auto('output_A0/HyFI_results.csv')
    WHERE final_cluster_id IS NOT NULL
    GROUP BY final_cluster_id
)
SELECT *
FROM fault_stats
WHERE num_events >= 10 AND avg_kappa > 3
ORDER BY num_events DESC
```

### Percentile Calculations

```sql
-- Calculate percentiles for key metrics
SELECT 
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY MAG) as mag_p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY MAG) as mag_median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY MAG) as mag_p75,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY epsilon) as epsilon_p90,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY epsilon) as epsilon_p95
FROM read_csv_auto('output_A0/HyFI_results.csv')
WHERE epsilon IS NOT NULL
```

### Saving Query Results

```python
import duckdb

# Execute query and save to new CSV
duckdb.query("""
    COPY (
        SELECT ID, Date, MAG, mean_azi, mean_dip, epsilon, kappa
        FROM read_csv_auto('output_A0/HyFI_results.csv')
        WHERE epsilon < 30 AND kappa > 5
    ) TO 'high_quality_events.csv' (HEADER, DELIMITER ',')
""")

print("Results saved to high_quality_events.csv")
```

### Creating Persistent Database

```python
import duckdb

# Create a persistent database from CSV files
con = duckdb.connect('hyfi_analysis.db')

# Import CSV data into tables
con.execute("""
    CREATE TABLE hyfi_results AS 
    SELECT * FROM read_csv_auto('output_A0/HyFI_results.csv')
""")

con.execute("""
    CREATE TABLE active_planes AS 
    SELECT * FROM read_csv_auto('output_A0/active_plane_determination_summary.csv')
""")

# Now you can query the tables
result = con.execute("""
    SELECT COUNT(*) as total FROM hyfi_results
""").df()

print(result)
con.close()
```

---

## Tips and Best Practices

### 1. Filter Early

```sql
-- Good: Filter before aggregation
SELECT final_cluster_id, AVG(MAG) as avg_mag
FROM read_csv_auto('output_A0/HyFI_results.csv')
WHERE mean_azi IS NOT NULL  -- Filter first
GROUP BY final_cluster_id

-- Less efficient: Filter after aggregation
-- (DuckDB optimizes this, but better to be explicit)
```

### 2. Use FILTER Clause for Conditional Counts

```sql
-- Clean and efficient
SELECT 
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE epsilon < 30) as good_fit,
    COUNT(*) FILTER (WHERE epsilon >= 30) as poor_fit
FROM read_csv_auto('output_A0/HyFI_results.csv')
```

### 3. Handle NULL Values

```sql
-- Explicitly handle NULLs and special values
SELECT *
FROM read_csv_auto('output_A0/HyFI_results.csv')
WHERE 
    mean_azi IS NOT NULL 
    AND sliptend IS NOT NULL
    AND sliptend != -999.0  -- Exclude special "not computed" value
```

### 4. Round Floating Point Results

```sql
-- Make results more readable
SELECT 
    ROUND(AVG(MAG), 2) as avg_mag,
    ROUND(AVG(kappa), 3) as avg_kappa
FROM read_csv_auto('output_A0/HyFI_results.csv')
```

---

## Common SQL Patterns

### Count Distinct Values

```sql
SELECT COUNT(DISTINCT final_cluster_id) as num_fault_systems
FROM read_csv_auto('output_A0/HyFI_results.csv')
WHERE final_cluster_id IS NOT NULL
```

### Top N Records

```sql
-- Largest magnitude events
SELECT ID, Date, MAG, mean_azi, mean_dip
FROM read_csv_auto('output_A0/HyFI_results.csv')
ORDER BY MAG DESC
LIMIT 10
```

### String Pattern Matching

```sql
-- Find events in specific cluster types
SELECT *
FROM read_csv_auto('output_A0/HyFI_results.csv')
WHERE final_cluster_id LIKE 'F%'  -- Starts with 'F'
```

### Date/Time Functions

```sql
-- Extract year and month
SELECT 
    EXTRACT(YEAR FROM CAST(Date AS TIMESTAMP)) as year,
    EXTRACT(MONTH FROM CAST(Date AS TIMESTAMP)) as month,
    COUNT(*) as num_events
FROM read_csv_auto('output_A0/HyFI_results.csv')
GROUP BY year, month
ORDER BY year, month
```

---

## CLI Command Quick Reference

| Command | Purpose | Key Options |
|---------|---------|-------------|
| `overview` | Fault systems overview | `--min-events N` |
| `summary` | Database-wide statistics | - |
| `instability` | High instability faults | `--limit N` |
| `spatial` | Events within radius | `--x X --y Y --radius R --limit N` |
| `fault-detail` | Complete fault information | `--fault-id ID` |
| `list-faults` | All fault system IDs | - |
| `query` | Custom SQL query | `--sql "SQL_QUERY"` |
| `tables` | Show table structure | - |

**Common Options:**
- `--database-dir PATH` - Required for all commands
- `--limit N` - Limit number of results
- `--format {table,json,csv}` - Output format (default: table)

---

## Troubleshooting

### Column Not Found

If you get "column not found" errors:
- Check the exact column name (case-sensitive)
- Verify the CSV file has the expected columns
- Use `DESCRIBE` to see available columns:

```python
import duckdb
result = duckdb.query("DESCRIBE SELECT * FROM read_csv_auto('output_A0/HyFI_results.csv')").df()
print(result)
```

### Type Conversion Errors

```sql
-- Explicitly cast types when needed
SELECT CAST(Date AS TIMESTAMP) as datetime
FROM read_csv_auto('output_A0/HyFI_results.csv')
```

### Performance Issues

For very large files:
- Use `WHERE` clauses to filter early
- Select only needed columns: `SELECT ID, MAG` instead of `SELECT *`
- Consider creating a persistent database for repeated queries

---

## See Also

- [Output Guide](output.md): Complete description of all output files and columns
- [Examples](examples.md): Configuration and workflow examples
- [Workflows Guide](workflows.md): Multi-sequence processing documentation
- [DuckDB Documentation](https://duckdb.org/docs/): Official DuckDB documentation
