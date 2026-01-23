# HyFI Database Querying

This guide explains how to query **HyFI** CSV output files using DuckDB and SQL. DuckDB is a fast, embeddable analytical database that can query CSV files directly without loading them into memory. It is especially useful to query large-scale datasets (as e.g. possible with the multi-sequence processing).

## Overview

Multi-sequence **HyFI** analysis produces a database (CSV files) containing earthquake analysis results:

| CSV File | Description | Location |
|----------|-------------|----------|
| **HyFI_database_metadata.csv** | Main database that incorporates all detected active faults, as well as the corresponding key attributes | `output_dir/HyFI_Database/` |
| **HyFI_database_segmentation.csv** | Results of the multi-sequence segmentation that lists all detected clustered earthquake sequences | `output_dir/HyFI_Database/` |
| **HyFI_database_hypocenters.csv** | Original hypocenter catalog with added key results of HyFI | `output_dir/HyFI_Database/` |
| **HyFI_database_focals.csv** | Original focal mechanism catalog with added key results of HyFI | `output_dir/HyFI_Database/` |

---

xxx TODO: check that duckdb is installed by default in the hyfi environment! then remove installation instructions here

## Installation

```bash
pip install duckdb
```

---

## Query Methods

HyFI provides different ways to query results:

1. **HyFI Query CLI** - Command-line tool for common queries (easiest)
2. **Custom SQL** - Direct DuckDB SQL queries (most flexible)

---

## HyFI Query CLI

The HyFI query CLI provides easy command-line access to common database queries for multi-sequence workflows.

### Basic Usage

All commands require specifying the database directory (`--database-dir PATH`), followed by the respective command:

```bash
python -m hyfi.query.cli --database-dir output_SECOS_VS/HyFI_Database <command>
```


### Available CLI commands

#### Commands Quick Overview

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
- `--limit N` - Limit number of results


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

---

Happy fault imaging! 🎉