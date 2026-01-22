# Installation Guide

This guide will walk you through installing HyFI from scratch, even if you have no previous installations.

---

## Step 1: Install Miniforge (Mamba Package Manager)

**Miniforge** provides both `conda` and `mamba` package managers. **We recommend using mamba** as it's significantly faster for package installation.

**Download and install Miniforge from:**
https://conda-forge.org/download/

Choose the appropriate installer for your operating system and follow the installation instructions provided on the website.

**Verify successful installation** by opening a new terminal and running:
```bash
mamba --version
```

---

## Step 2: Download HyFI

You have two options to get HyFI:

### Option A: Direct Download (Recommended)

If you just want to use a static version of HyFI without Git:

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

### Option B: Using Git (For Developers)

If you plan to modify the code or stay up-to-date with the latest changes:

```bash
git clone https://github.com/sandrotruttmann/HyFI.git
cd HyFI
git checkout main
```

---

## Step 3: Create the Mamba Environment

The `hyfi.yaml` file contains all the required dependencies.

1. **Create the environment using mamba:**

   ```bash
   mamba env create -f hyfi.yaml
   ```

   This may take several minutes as it downloads and installs all required packages.

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

---

## Step 6: Run a Test Example

Navigate to the example projects directory and run a test configuration:

```bash
cd example_projects
hyfi run config_TEMPLATE.json
```

Check the output directory specified in the DAG (.json) for results (output data, visualizations, VTP, etc.).

---

## Step 7: Install ParaView (Optional, for 3D Visualization)

**ParaView** is a powerful open-source application for visualizing and analyzing 3D data. HyFI exports VTP files that can be opened in ParaView for interactive visualization of hypocenters, focal mechanisms, rupture planes, fault systems and others.

### Download and Install ParaView

Download and install ParaView from the Official website: https://www.paraview.org/download/

---

## Next Steps

You are now ready to run your first HyFI Analysis! Please continue to **[Quickstart](quickstart)**.

---

Happy fault imaging! 🎉
