# Examples

The HyFI repository includes several example datasets and configuration files to help you get started.

## Single-Sequence Example Datasets

The `data_examples/` directory contains example hypocenter datasets:

- `A0_data.csv` - Basic example dataset
- `A18_data.csv` - Alternative dataset
- `A24_data.csv` - Additional example
- `A37_data.csv` - Advanced example
- `SECOS_20250305_HyFI.csv` - Real-world SECOS data

## Single-Sequence Example Configurations

- `config_A0.json` - Basic single-sequence analysis
- `config_A18.json` - Configuration for A18 dataset
- `config_A24.json` - Configuration for A24 dataset
- `config_A37.json` - Configuration for A37 dataset
- `config_TEMPLATE.json` - Complete template with all available parameters


## Running Examples

To run a basic single-sequence analysis:

```bash
hyfi run example_projects/config_A0.json
```

