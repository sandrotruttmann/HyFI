# DuckDB for HyFI - Quick Start Guide

## What Just Happened? ✨

DuckDB just analyzed all your HyFI results across 8 projects in seconds, without needing a database server!

## Key Results from Demo:

### 1. **Fast Multi-Project Analysis**
```
Project          Events  Faults  AvgMag  MaxMag
output_A0          545      1     0.88    3.50
output_A37         327      1     0.92    3.22
output_A24         294      1     1.06    4.25
output_A18         158      1     0.84    2.90
```

### 2. **Quality Analysis**
- Total events with good fit (ε < 15°): 8 events
- Best fit: ε = 3.55° (event KP201911080308)
- Avg slip tendency: 0.497

### 3. **Spatial Queries**
- Found 10 events within 2km of center point (594000, 130000)
- Distances calculated in milliseconds

## Why DuckDB is Perfect for HyFI:

✅ **No setup** - Just `pip install duckdb`  
✅ **Query CSVs directly** - No import needed  
✅ **Fast** - Processes GBs in seconds  
✅ **SQL** - Familiar query language  
✅ **Python native** - Works with pandas seamlessly  
✅ **Multi-file** - Query across all projects at once  

## Run the Demo:

```bash
# Install (already done)
pip install duckdb

# Run
python duckdb_demo_fixed.py
```

## Quick Examples:

### Example 1: Find Best Fitting Events
```python
import duckdb
import pandas as pd

df = pd.read_csv('output_A0/HyFI_results.csv')

result = duckdb.sql("""
    SELECT ID, MAG, epsilon, sliptend
    FROM df
    WHERE epsilon < 15
    ORDER BY epsilon
    LIMIT 5
""").df()

print(result)
```

### Example 2: Compare All Projects
```python
import pandas as pd
import duckdb
from pathlib import Path

# Load all results
all_results = []
for output_dir in Path('.').glob('output_A*'):
    if (output_dir / 'HyFI_results.csv').exists():
        temp_df = pd.read_csv(output_dir / 'HyFI_results.csv')
        temp_df['project'] = output_dir.name
        all_results.append(temp_df)

df_all = pd.concat(all_results)

# Compare magnitudes across projects
result = duckdb.sql("""
    SELECT 
        project,
        COUNT(*) as events,
        AVG(MAG) as avg_mag,
        MAX(MAG) as max_mag
    FROM df_all
    GROUP BY project
    ORDER BY avg_mag DESC
""").df()

print(result)
```

### Example 3: Active Plane Statistics
```python
df_active = pd.read_csv('output_A0/active_plane_determination_summary.csv')

result = duckdb.sql("""
    SELECT 
        plane_determination_method,
        COUNT(*) as count,
        AVG(epsilon) as avg_angular_diff,
        MIN(epsilon) as best_fit,
        MAX(epsilon) as worst_fit
    FROM df_active
    WHERE epsilon IS NOT NULL
    GROUP BY plane_determination_method
""").df()

print(result)
```

### Example 4: Spatial Analysis
```python
df = pd.read_csv('output_A0/HyFI_results.csv')

# Events within 5km radius
result = duckdb.sql("""
    SELECT 
        ID, MAG, class,
        ROUND(SQRT(
            POWER(X - 594000, 2) + 
            POWER(Y - 130000, 2)
        ) / 1000, 2) as distance_km
    FROM df
    WHERE X IS NOT NULL
        AND SQRT(POWER(X - 594000, 2) + POWER(Y - 130000, 2)) < 5000
    ORDER BY distance_km
""").df()

print(result)
```

## Advanced: Persistent Database (Optional)

For even faster repeated queries:

```python
import duckdb

# Create persistent database file
con = duckdb.connect('hyfi_analysis.duckdb')

# Import all your data once
con.execute("""
    CREATE TABLE hyfi_results AS 
    SELECT * FROM read_csv_auto('output_*/HyFI_results.csv')
""")

# Now queries are lightning fast
con.sql("""
    SELECT class, COUNT(*), AVG(MAG), AVG(epsilon)
    FROM hyfi_results
    GROUP BY class
""").show()

con.close()

# Later, just connect and query
con = duckdb.connect('hyfi_analysis.duckdb')
con.sql("SELECT * FROM hyfi_results WHERE MAG > 2.0").show()
```

## When to Use DuckDB vs PostgreSQL:

| Task | Best Tool |
|------|-----------|
| Quick analysis of existing CSVs | **DuckDB** |
| Multi-project comparisons | **DuckDB** |
| Exploratory data analysis | **DuckDB** |
| Research workflows | **DuckDB** |
| Production web app | **PostgreSQL** |
| Multi-user concurrent access | **PostgreSQL** |
| Very complex transactions | **PostgreSQL** |
| Long-term data warehouse | **PostgreSQL** |

## Next Steps:

1. ✅ DuckDB is installed and working
2. ✅ Demo shows analysis across all projects
3. Use DuckDB for your analysis workflows
4. Consider PostgreSQL only if you need:
   - Web applications
   - Multi-user access
   - Complex data management

## Documentation:

- DuckDB: https://duckdb.org/
- SQL Reference: https://duckdb.org/docs/sql/introduction
- Python API: https://duckdb.org/docs/api/python/overview

---

**Bottom Line**: DuckDB gives you SQL power without database overhead! 🚀
