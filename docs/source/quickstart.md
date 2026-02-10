# Quickstart

This quickstart guide will help you run your first simple HyFI analysis of a single earthquake sequence.

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
hyfi run -c config_A0.json
```

The analysis will run and you'll see progress output in the terminal.

### Step 5: View the Results

Once complete, navigate to the output directory:

```bash
cd ../output_A0
ls -l
```

You should see (among other files):
- `HyFI_results.csv` - Main results table that incorporates the input data, complemented with the HyFI results for each hypocenter
- `3D_model.html` - Simple interactive 3D visualization (open in browser)
- `vtp_export/` - ParaView-compatible 3d visualization files
- `execution_summary.json` - Analysis metadata and statistics


### Step 6: Opening VTP Files in ParaView

1. **Launch ParaView**

2. **Open VTP files:**
   - File → Open
   - Navigate to your HyFI output directory (e.g., `output_A0/vtp_export/`)
   - Select one or more `.vtp` files (e.g., `faults_compiled.vtp`, `rupture_planes.vtp`)
   - Click "Apply" in the Properties panel

3. **Visualize your data:**
   - Use the toolbar to change representation (Surface, Wireframe, Points)
   - Apply filters (Clip, Slice, Threshold) to analyze specific regions
   - Color by different attributes (cluster_id, magnitude, depth, etc.)
   - Export images or animations


## Next Steps

Congrats! You have successfully went through the **[Installation Guide](installation)** and **[Quickstart](quickstart)** of **HyFI**. You find more information on the following pages:

- **[Processing Workflow](workflows)**: Learn about single- vs multi-sequence processing
- **[Configuration](configuration)**: Learn how to set up JSON configuration files for your analysis
- **[Input Parameters](input_parameters)**: Detailed documentation of all configuration parameters
- **[Modules](modules/index)**: Understand the different analysis modules (e.g. fault network reconstruction)
- **[Output Guide](output)**: Learn about the generated output files and visualizations
- **[Queries](querying)**: Learn how to efficiently query the HyFI results using SQL and DuckDB.

---

Happy fault imaging! 🎉
