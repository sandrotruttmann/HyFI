

[![DOI](https://zenodo.org/badge/582537470.svg)](https://zenodo.org/badge/latestdoi/582537470)



# HyFI: Hypocenter-based 3D Imaging of Active Faults

The code in this repository allows to image active faults in 3D based on relatively relocated hypocenter datasets after the method presented in Truttmann et al. (2023) (https://doi.org/10.1029/2023JB026352).

## Scientific publication
If you use the code in this repository please cite the following scientific publication:
- Truttmann, S., Diehl, T., Herwegh, M. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps. Journal of Geophysical Research: Solid Earth, https://doi.org/10.1029/2023JB026352.


## Installation
To make this code work on your machine you can simply clone this repository and install HyFI as a package:

```bash
git clone https://github.com/sandrotruttmann/hypo_fault_imaging.git
cd hypo_fault_imaging
pip install -e .
```

## Requirements
The necessary dependencies are automatically installed when you install the package. Alternatively, you can create a conda environment from the provided .yaml file:

```bash
conda env create -f hyfi.yaml
```

## Usage

### JSON DAG Configuration (Recommended)

HyFI now supports JSON-based DAG (Directed Acyclic Graph) configurations that allow you to define your entire analysis workflow without writing Python code. This approach provides better structure, validation, and reproducibility.

#### Quick Start with JSON DAG

1. **Run the Lenk example:**
   ```bash
   hyfi run --config-file example_projects/lenk_project.json
   ```

2. **Validate your configuration:**
   ```bash
   hyfi config validate example_projects/lenk_project.json
   ```

3. **Create a new project template:**
   ```bash
   hyfi config create --format json --output my_project.json
   ```

#### JSON Configuration Structure

A JSON DAG configuration file has three main sections:

**1. Metadata**
```json
{
  "metadata": {
    "workflow_name": "My Fault Analysis",
    "workflow_version": "1.0.0",
    "created_date": "2025-09-02T00:00:00",
  }
}
```

**2. Global Settings**
```json
{
  "global_settings": {
    "output_directory": "/path/to/output/",
  }
}
```

**3. Workflow DAG**
The workflow defines analysis steps and their dependencies:

```json
{
  "workflow_dag": {
    "input_data": {
      "hypocenter_file": "/path/to/hypocenter_data.csv",
      "hypocenter_separator": ",",
      "focal_mechanism_file": "/path/to/focal_mechanisms.csv",
      "focal_mechanism_separator": ",",
    },
    "fault_network": {
      "parameters": {
        "monte_carlo_simulations": 1000,
        "search_radius_meters": 300.0,
        "search_time_window_hours": 8766.0,
        "magnitude_type": "ML"
      },
    },
    "model_validation": {
      "enabled": true,
      "parameters": {
        "check_magnitude_consistency": true,
        "check_location_consistency": true,
      },
    },
    "auto_classification": {
      "enabled": true,
      "parameters": {
        "number_of_clusters": 2,
        "clustering_algorithm": "vmf_soft",
        "rotate_poles_before_analysis": true
      },
    },
    "stress_analysis": {
      "enabled": true,
      "parameters": {
        "stress_field": {
          "sigma1_trend_degrees": 130,
          "sigma1_plunge_degrees": 20,
          "sigma3_trend_degrees": 34,
          "sigma3_plunge_degrees": 12,
          "stress_shape_ratio": 0.3
        },
        "mechanical_properties": {
          "pore_pressure_mpa": 0.0,
          "friction_coefficient": 0.75
        }
      },
    },
    "visualization": {
      "enabled": true,
      "parameters": {
        "generate_3d_model": true,
        "generate_stereonet": true,
        "interactive_plots": true
      },
    }
  }
}
```

#### Available Node Types

- **`input_data`**: Load hypocenter and focal mechanism data
- **`fault_network`**: 3D fault network reconstruction using Monte Carlo simulation
- **`model_validation`**: Validate results against focal mechanisms
- **`auto_classification`**: Automatic fault structure classification
- **`stress_analysis`**: Stress field analysis and failure assessment
- **`visualization`**: Generate 3D plots and summary visualizations
- **`output`**: Save results in various formats

#### Creating New Projects

1. **Copy an existing configuration:**
   ```bash
   cp example_projects/lenk_project.json my_new_project.json
   ```

2. **Edit the configuration:**
   - Update file paths in the `input_data` node
   - Modify parameters for your specific analysis
   - Change the output directory in `global_settings`
   - Update the workflow name in `metadata`

3. **Run the analysis:**
   ```bash
   hyfi config validate my_new_project.json
   hyfi run --config-file my_new_project.json
   ```

#### Enabling/Disabling Analysis Steps

Control which analysis steps run by setting the `enabled` field:

```json
{
  "model_validation": {
    "enabled": false,  // Skip model validation
    ...
  },
  "stress_analysis": {
    "enabled": true,   // Run stress analysis
    ...
  }
}
```

#### Command Reference

- `hyfi run --config-file <file>` - Run analysis
- `hyfi config validate <file>` - Validate configuration  
- `hyfi config create --format json --output <file>` - Create template
- `hyfi analyze --hypo-file <file> --output-dir <dir>` - Quick analysis
- `hyfi --help` - Show all available commands

#### Output Structure

After execution, the output directory contains:
```
output_directory/
```


### Alternative Configuration Approaches

#### Using Configuration Objects (Legacy Support)
```python
from hyfi import ProjectConfig, FaultImagingWorkflow

# Create configuration with validation
config = ProjectConfig(
    project_title='My Analysis',
    hypo_file='path/to/hypocenter_data.txt',
    hypo_sep='\t'
)

# Customize individual modules
config.fault_network.n_mc = 2000
config.fault_network.r_nn = 150.0
config.model_validation.validation_bool = True
config.model_validation.foc_file = 'path/to/focal_mechanisms.txt'

# Validate configuration
config.validate()

# Run analysis
workflow = FaultImagingWorkflow(config)
results = workflow.run_full_analysis()
```

#### Creating Configuration Templates
```bash
# Create a JSON DAG template
hyfi config create --format json --output my_analysis.json

# Create a legacy configuration template  
hyfi config create --format json --output legacy_config.json
```

### Example Files

- `examples/example_dag_config.json` - Complete workflow with all steps
- `examples/dag_example.py` - DAG creation and validation examples
- `examples/run_dag_example.py` - DAG execution tutorial
- `samples/runfile.py` - Legacy workflow example

#### Data Format Requirements

**Hypocenter Data**: Must follow hypoDD standard with these columns:
- ID, LAT, LON, DEPTH, X, Y, Z, EX, EY, EZ, YR, MO, DY, HR, MI, SC, MAG, NCCP, NCCS, NCTP, NCTS, RCC, RCT, CID
- Relocation errors should be given as one standard deviation (σ)

**Focal Mechanism Data**: Must follow the required header naming convention:
- ID, LAT, LON, DEPTH, X, Y, Z, YR, MO, DY, HR, MI, SC, MAG, A, Strike1, Dip1, Rake1, Strike2, Dip2, Rake2, Pazim, Pdip, Tazim, Tdip, Q, Type, Loc

Please note:
- The input files must match these exact formats.
- The hypocenter and focal mechanism data must share the same event IDs.

Since the choice of the input parameters r_nn and dt_nn is critical, an example for the sensitivity analysis of these parameters is provided in the file "inputparams_sensitivity.py" (for details see Truttmann et al. (2023)).

#### Getting Help

- Use `hyfi --help` for command-line help
- Use `hyfi config validate <file>` to check configuration issues
- Check the execution logs in the output directory for detailed error information


#### TO BE DELETED LATER: Migration from Python Runfiles

Convert existing Python runfiles to JSON DAG format:

```bash
python convert_runfile_to_dag.py examples_archive/runfile_Lenk.py projects/lenk_converted.json
```
