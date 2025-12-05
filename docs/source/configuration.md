# HyFI Configuration

HyFI supports JSON-based DAG (Directed Acyclic Graph) configurations that allow you to define your entire analysis workflow without writing Python code. This approach provides better structure, validation, and reproducibility.

## JSON Configuration Structure

A JSON DAG configuration file has three main sections:

### 1. Metadata

Information about the workflow:

```json
{
  "metadata": {
    "workflow_name": "My Fault Analysis",
    "workflow_version": "1.0.0",
    "created_date": "2025-11-11T00:00:00",
    "description": "Optional description of the analysis"
  }
}
```

### 2. Global Settings

Common settings applied across the workflow:

```json
{
  "global_settings": {
    "output_directory": "./hyfi_output",
    "log_level": "INFO"
  }
}
```

### 3. Workflow DAG

The workflow defines analysis steps and their dependencies. See the complete example in `config_TEMPLATE.json`.

## Available Workflow Nodes

- **`input_data`**: Load hypocenter and focal mechanism data
- **`fault_network`**: 3D fault network reconstruction using Monte Carlo simulation
- **`model_validation`**: Validate results against focal mechanisms
- **`auto_classification`**: Automatic fault structure classification
- **`stress_analysis`**: Stress field analysis and failure assessment
- **`visualization`**: Generate 3D plots and summary visualizations

## Minimal Configuration Example

Here's a minimal working configuration:

```json
{
  "metadata": {
    "workflow_name": "Basic Analysis",
    "workflow_version": "1.0.0"
  },
  "global_settings": {
    "output_directory": "./output"
  },
  "workflow_dag": {
    "input_data": {
      "hypocenter_file": "data_examples/A0_data.csv",
      "hypocenter_separator": ","
    },
    "fault_network": {
      "parameters": {
        "core_network": {
          "search_radius_meters": 100.0,
          "search_time_window_hours": 9999999
        }
      }
    }
  }
}
```

For a complete list of all available parameters, see the [Input Parameters](input_parameters.md) documentation.
