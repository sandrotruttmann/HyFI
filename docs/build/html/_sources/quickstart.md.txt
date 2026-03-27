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
cd config
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


## Using Your Own Data

### ECOS Catalog Parser

If you have ECOS earthquake catalog files, HyFI provides a built-in parser to convert them to the required input format. The parser supports two types of ECOS catalogs:

- **ConsolidatedMergeCat**: Hypocenter catalog with event locations and magnitudes
- **AssociateFM**: Focal mechanism catalog with nodal plane solutions

#### Quick Start

Convert ECOS catalogs to HyFI format (use the actual filenames from your ECOS export):

```bash
hyfi parse-ecos --hypo <ConsolidatedMergeCat_file.csv> --focals <AssociateFM_file.csv>
```

**Example with actual ECOS filenames:**

```bash
hyfi parse-ecos --hypo ECOS_Merge_Bull+AbsRel+DDC+DDR_20260116.ConsolidatedMergeCat.csv \
                 --focals ECOS_Merge_Bull+AbsRel+DDC+DDR_20260116.AssociateFM.csv
```

At least one of `--hypo` or `--focals` must be specified. Output files are automatically created with `_HyFI` suffix:
- `ECOS_Merge_Bull...ConsolidatedMergeCat.csv` → `ECOS_Merge_Bull...ConsolidatedMergeCat_HyFI.csv`
- `ECOS_Merge_Bull...AssociateFM.csv` → `ECOS_Merge_Bull...AssociateFM_HyFI.csv`

#### Input Format

The parser expects **pipe-delimited (|) CSV files with an embedded comment header block** that defines column names:

```
# Column definitions
# KP-ID|Lat|Lon|Depth|...
EV001|45.5234|9.7654|8.5|...
EV002|45.5245|9.7665|9.2|...
```

#### Output Format (HyFI Standard)

The parser generates standard HyFI CSV files with required columns:

**Hypocenter file (17 columns):**
```
ID,LAT,LON,DEPTH,X,Y,Z,EX,EY,EZ,YR,MO,DY,HR,MI,SC,MAG
```

**Focal mechanism file (27 columns):**
Hypocenter columns plus:
```
A,Strike1,Dip1,Rake1,Strike2,Dip2,Rake2,Pazim,Pdip,Tazim,Tdip,Q,Type
```

### Other Data Formats

For earthquake catalogs from other sources, ensure your CSV files have these required columns:

| Column | Type | Description |
|--------|------|-------------|
| ID | string | Event identifier |
| LAT | float | Latitude (degrees, WGS84) |
| LON | float | Longitude (degrees, WGS84) |
| DEPTH | float | Depth (kilometers) |
| X | float | Easting in local projection (meters) |
| Y | float | Northing in local projection (meters) |
| Z | float | Depth in meters below datum (negative values) |
| EX, EY, EZ | float | Location uncertainties (meters) |
| YR, MO, DY, HR, MI | int | Year, month, day, hour, minute |
| SC | float | Second with fractional part |
| MAG | float | Magnitude |

For focal mechanisms, additionally include:
- `A` (int): Quality flag (1/2=valid, 0=invalid)
- `Strike1, Dip1, Rake1`: First nodal plane (degrees)
- `Strike2, Dip2, Rake2`: Second nodal plane (degrees)
- `Pazim, Pdip`: P-axis (degrees)
- `Tazim, Tdip`: T-axis (degrees)
- `Q` (int): Quality rating (1-5 scale)
- `Type` (string): Event type classification

The file separator must be specified in your HyFI configuration (`hypocenter_separator` and `focal_mechanism_separator`): `","` (CSV), `"\t"` (TSV), or `";"` (semicolon).

---

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
