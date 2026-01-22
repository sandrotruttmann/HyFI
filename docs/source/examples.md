# Examples

The HyFI repository includes several example datasets and configuration files to help you get started.

## Example Datasets

The `data_examples/` directory contains example hypocenter datasets:

- `A0_data.csv` - Basic example dataset
- `A18_data.csv` - Alternative dataset
- `A24_data.csv` - Additional example
- `A37_data.csv` - Advanced example
- `SECOS_20250305_HyFI.csv` - Real-world SECOS data

## Example Configurations

The `example_projects/` directory contains pre-configured analysis setups:

### Single-Sequence Configurations

- `config_A0.json` - Basic single-sequence analysis
- `config_A18.json` - Configuration for A18 dataset
- `config_A24.json` - Configuration for A24 dataset
- `config_A37.json` - Configuration for A37 dataset
- `config_TEMPLATE.json` - Complete template with all available parameters

### Multi-Sequence Configurations

- `multi_sequence_example.json` - Full multi-scale segmentation workflow with Class A and B clustering
- `segmentation_only_example.json` - Catalog segmentation without full fault analysis

## Running Examples

### Single-Sequence Analysis

To run a basic single-sequence analysis:

```bash
hyfi run example_projects/config_A0.json
```

Or with the full command:

```bash
python -m hyfi.cli.main example_projects/config_A0.json
```

This will process the data and generate outputs in the corresponding output directory (e.g., `output_A0/`).

### Multi-Sequence Analysis

To run a multi-sequence workflow with catalog segmentation:

```bash
hyfi run example_projects/multi_sequence_example.json
```

This workflow will:
1. Load the hypocenter catalog
2. Segment the catalog into distinct sequences using multi-scale DBSCAN clustering
3. Analyze each sequence independently with the full HyFI workflow
4. Merge results into a unified database and VTP file

See [Workflows](workflows.md) for detailed documentation on single-sequence vs multi-sequence processing.

### Segmentation-Only Workflow

For quick catalog exploration without full fault analysis:

```bash
hyfi run example_projects/segmentation_only_example.json
```

This is useful for:
- Exploring catalog structure before full analysis
- Exporting sequences for external visualization
- Testing segmentation parameters

## Output Files

### Single-Sequence Output

Each single-sequence run generates:

- `3D_model.html` - Interactive 3D visualization
- `HyFI_results.csv` - Main results table
- `active_plane_determination_summary.csv` - Summary of active planes
- `active_plane_statistics.txt` - Statistical analysis
- `execution_summary.json` - Execution metadata
- `parameter_optimization_report.json` - Optimization results
- `vtp_export/` - ParaView-compatible files (if enabled)
- `obj_export/` - Wavefront OBJ files (if enabled)

### Multi-Sequence Output

Multi-sequence workflows produce:

```
output_SECOS_VS_filtered/
├── A1/                    # First Class A sequence
│   ├── HyFI_results.csv
│   ├── 3D_model.html
│   └── vtp_export/
├── A2/                    # Second Class A sequence
├── B1/                    # First Class B sequence
├── Z_outliers/            # Unassigned events
├── HyFI_Database/
│   └── hyfi_results.db    # SQLite database with all sequences
├── merged_sequences.vtp   # Combined VTP file
└── segmentation_summary.json  # Clustering metadata
```

## Validating Configurations

Before running, validate your configuration:

```bash
hyfi config validate example_projects/config_A0.json
```

## Creating Custom Configurations

Create a new configuration from the template:

```bash
hyfi config create --output my_project.json
```

Edit the parameters in `my_project.json` to match your dataset and analysis requirements.

## Querying Results

HyFI provides three query methods:

### 1. Query CLI (Easiest)

```bash
# Get fault systems overview
python -m hyfi.query.cli --database-dir output_SECOS_VS/HyFI_Database overview

# Show high instability faults
python -m hyfi.query.cli --database-dir output_SECOS_VS/HyFI_Database instability --limit 5

# Events within radius
python -m hyfi.query.cli --database-dir output_SECOS_VS/HyFI_Database spatial \
    --x 592500 --y 130500 --radius 2.0
```

### 2. Python Query API

```python
from hyfi.query import HyFIDatabase, HyFIQueries
from pathlib import Path

# Initialize database
with HyFIDatabase() as db:
    db.load_csv_tables(Path('output_SECOS_VS/HyFI_Database'))
    
    # Use pre-built queries
    queries = HyFIQueries(db)
    overview = queries.fault_systems_overview(min_events=10)
    unstable = queries.high_instability_faults(limit=5)
    spatial = queries.spatial_query(592500, 130500, radius_km=2.0)
```

### 3. Custom SQL with DuckDB

```python
import duckdb

# Query CSV files directly
result = duckdb.query("""
    SELECT fault_id, COUNT(*) as num_events, AVG(MAG) as avg_mag
    FROM read_csv_auto('output_SECOS_VS/HyFI_Database/hypocenters.csv')
    WHERE MAG >= 2.0
    GROUP BY fault_id
    ORDER BY num_events DESC
""").df()

print(result)
```

**See [Querying Results](querying.md) for complete documentation** covering:
- **CLI Command Reference** - All 8 CLI commands with examples
- **Python API Guide** - 10+ pre-built query functions  
- **Custom SQL Examples** - 20+ SQL queries for analysis
- CSV file structure and column descriptions
- Spatial queries, temporal analysis, stress analysis
- Performance tips and troubleshooting

## Visualization

### Web Browser

Open the 3D HTML visualization:

```bash
firefox output_A0/3D_model.html
```

### ParaView

For advanced visualization:

1. Open ParaView
2. Load `output_A0/vtp_export/*.vtp` files
3. Apply the state file: `ParaView/HyFI_3Dmodel_template.pvsm`

### Python

Generate custom visualizations:

```python
import pyvista as pv

# Load VTP file
mesh = pv.read('output_A0/vtp_export/fault_network.vtp')

# Create plot
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='magnitude', cmap='viridis')
plotter.show()
```

## Additional Resources

For more information:

- **[Workflows Guide](workflows.md)**: Detailed explanation of single-sequence vs multi-sequence processing
- **[Configuration Guide](configuration.md)**: Complete configuration reference
- **[Input Parameters](input_parameters.md)**: Detailed parameter documentation
- **[Output Guide](output.md)**: Understanding output files and formats
