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

- `config_A0.json` - Configuration for A0 dataset
- `config_A18.json` - Configuration for A18 dataset
- `config_A24.json` - Configuration for A24 dataset
- `config_A37.json` - Configuration for A37 dataset
- `multi_sequence_example.json` - Multi-sequence workflow example
- `config_TEMPLATE.json` - Template for creating new configurations

## Running Examples

To run any of these examples:

```bash
hyfi run --config-file example_projects/config_A0.json
```

This will process the data and generate outputs in the corresponding output directory (e.g., `output_A0/`).

## Output Files

Each run generates several output files:

- `3D_model.html` - Interactive 3D visualization
- `HyFI_results.csv` - Main results table
- `active_plane_determination_summary.csv` - Summary of active planes
- `active_plane_statistics.txt` - Statistical analysis
- `execution_summary.json` - Execution metadata
- `parameter_optimization_report.json` - Optimization results
- Various export formats in subdirectories (VTP, OBJ, PLY, etc.)

For more information on configuration options, see [Configuration](configuration.md) and [Input Parameters](input_parameters.md).
