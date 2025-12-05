# Quickstart

This quickstart guide will help you run your first HyFI analysis in minutes.

## Prerequisites

Make sure you have completed the [Installation](installation) steps before proceeding.

## Running Your First Analysis

### Step 1: Activate the HyFI Environment

Open a terminal and activate the mamba environment:

```bash
mamba activate hyfi
```

You should see `(hyfi)` appear at the beginning of your terminal prompt.

### Step 2: Navigate to the HyFI Directory

```bash
cd /path/to/HyFI
```

Replace `/path/to/HyFI` with the actual path where you installed HyFI (e.g., `~/HyFI` or `C:\Users\YourName\HyFI`).

### Step 3: Navigate to Example Projects

```bash
cd example_projects
```

### Step 4: Run an Example Configuration

HyFI uses JSON configuration files to define the entire analysis workflow. Run the example project:

```bash
hyfi run config_A0.json
```

Or use the full Python command:

```bash
python -m hyfi.cli.main config_A0.json
```

The analysis will run and you'll see progress output in the terminal.

### Step 5: View the Results

Once complete, navigate to the output directory:

```bash
cd ../output_A0
ls -l
```

You should see:
- `HyFI_results.csv` - Main results table with fault parameters
- `3D_model.html` - Interactive 3D visualization (open in browser)
- `vtp_export/` - ParaView-compatible files
- `execution_summary.json` - Analysis metadata and statistics

Open the 3D visualization:

```bash
# Linux/macOS
open 3D_model.html

# Or just open it in your web browser
firefox 3D_model.html
```

## Advanced Usage

### Validating Configuration

Before running, validate your configuration file:

```bash
hyfi config validate config_A0.json
```

### Creating a New Project

Create a new project template:

```bash
hyfi config create --output my_project.json
```

Edit the input parameters specified in `my_project.json` to configure your analysis, then run:

```bash
hyfi run my_project.json
```

### Using Python Scripts

You can also run HyFI from Python:

```python
from hyfi.core.dag_executor import DAGExecutor
from hyfi.config.parameters import load_config

# Load configuration
config = load_config("config_A0.json")

# Execute workflow
executor = DAGExecutor(config.workflow_dag)
results = executor.execute()
```

## Next Steps

- **[Configuration Guide](configuration)**: Learn about JSON configuration structure
- **[Input Parameters](input_parameters)**: Detailed parameter documentation
- **[HyFI Modules](modules/index)**: Understand each analysis module
- **[Output Guide](output)**: Explore output files and formats

For detailed information about all configuration options, see the [Input Parameters](input_parameters) section.
