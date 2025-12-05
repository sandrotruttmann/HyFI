# Installation Guide

This guide will walk you through installing HyFI from scratch, even if you have no previous installations.

---

## Step 1: Install Miniforge (mamba/Mamba Package Manager)

**Miniforge** provides both `mamba` and `mamba` package managers. **We recommend using mamba** as it's significantly faster for package installation.

**Download and install Miniforge from:**
https://github.com/mamba-forge/miniforge/releases

Choose the appropriate installer for your operating system and follow the installation instructions provided on the website.

**Verify installation** by opening a new terminal and running:
```bash
mamba --version
mamba --version
```

---

## Step 2: Download HyFI

You have two options to get HyFI:

### Option A: Using Git (Recommended for developers)

If you plan to modify the code or stay up-to-date with the latest changes:

```bash
git clone https://github.com/sandrotruttmann/HyFI.git
cd HyFI
git checkout dev  # Use the development branch
```

### Option B: Direct Download (No Git required)

If you just want to use HyFI without Git:

1. **Download the repository as a ZIP file:**
   - Go to: https://github.com/sandrotruttmann/HyFI
   - Click the green "Code" button
   - Select "Download ZIP"

2. **Extract the ZIP file:**
   - Extract to your desired location (e.g., `~/HyFI` or `C:\Users\YourName\HyFI`)

3. **Navigate to the extracted folder:**
   ```bash
   cd path/to/HyFI
   ```

---

## Step 3: Create the Mamba Environment

The `hyfi.yaml` file contains all the required dependencies.

1. **Create the environment using mamba:**

   ```bash
   mamba env create -f hyfi.yaml
   ```

   This may take several minutes as it downloads and installs all packages.

2. **Activate the environment:**

   ```bash
   mamba activate hyfi
   ```

   You should see `(hyfi)` appear at the beginning of your terminal prompt.

---

## Step 4: Install HyFI

Install HyFI using pip:

```bash
pip install .
```

This installs HyFI from the local source code into your mamba environment.

---

## Step 5: Verify Installation

1. **Check that HyFI is installed:**

   ```bash
   python -c "import hyfi; print(hyfi.__version__)"
   ```

2. **Check that the CLI works:**

   ```bash
   hyfi --help
   ```

   You should see the HyFI command-line interface help text.

3. **Check key dependencies:**

   ```bash
   python -c "import pyvista, obspy, numpy, pandas; print('All key packages imported successfully!')"
   ```

---

## Step 6: Run a Test Example

Navigate to the example projects directory and run a test configuration:

```bash
cd example_projects
hyfi run config_TEMPLATE.json
```

Or use Python directly:

```bash
python -m hyfi.cli.main config_TEMPLATE.json
```

Check the output directory for results (VTP, OBJ, PLY files, visualizations, etc.).

---

## Step 7: Install ParaView (Optional, for 3D Visualization)

**ParaView** is a powerful open-source application for visualizing and analyzing 3D scientific data. HyFI exports VTP files that can be opened in ParaView for interactive visualization of hypocenters, focal mechanisms, rupture planes, fault systems and others.

### Download and Install ParaView

**Official website:** https://www.paraview.org/download/

### Opening HyFI Results in ParaView

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

**Note:** While ParaView is optional, it's highly recommended for interactive 3D exploration of your HyFI results. The OBJ and PLY exports can also be opened in other 3D software like Blender, MeshLab, or CloudCompare.

---

## Next Steps

Now that HyFI is installed, continue with:

- **[Configuration Guide](configuration)**: Learn how to set up JSON configuration files for your analysis
- **[Input Parameters](input_parameters)**: Detailed documentation of all configuration parameters
- **[HyFI Modules](modules/index)**: Understand the different analysis modules (fault network, validation, classification, stress analysis)
- **[Output Guide](output)**: Learn about the generated output files and visualizations
- **[Examples](examples)**: Explore example projects and use cases

### Quick Start

Try running one of the example configurations:

```bash
cd example_projects
hyfi run config_A0.json
```

Check the `output_A0/` directory for results including:
- `HyFI_results.csv` - Main results table
- `3D_model.html` - Interactive 3D visualization
- `vtp_export/` - ParaView-compatible files

---

Happy fault imaging! 🎉
