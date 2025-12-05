<div align="center">
  <img src="hyfi_logo.svg" alt="HyFI Logo" width="500">
</div>

<div align="center">

[![DOI](https://zenodo.org/badge/582537470.svg)](https://zenodo.org/badge/latestdoi/582537470)
[![Documentation Status](https://readthedocs.org/projects/hyfi/badge/?version=latest)](https://hyfi.readthedocs.io/en/latest/?badge=latest)

</div>

# HyFI: Hypocenter-based 3D Imaging of Active Faults

**HyFI** is a Python package for imaging active faults in 3D based on relatively relocated earthquake hypocenter datasets. The method uses nearest neighbor learning and principal component analysis (PCA) to reconstruct 3D fault geometries from spatiotemporal hypocenter and focal mechanism catalogs.


## 📖 Citation

If you use HyFI in your research, please cite:

**Truttmann, S., Diehl, T., Herwegh, M. (2023).** Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps. *Journal of Geophysical Research: Solid Earth*. https://doi.org/10.1029/2023JB026352

## 📚 Documentation

**Full documentation is available at [hyfi.readthedocs.io](https://hyfi.readthedocs.io/)**

- [Installation Guide](https://hyfi.readthedocs.io/en/latest/installation.html)
- [Quickstart Tutorial](https://hyfi.readthedocs.io/en/latest/quickstart.html)
- [Configuration Guide](https://hyfi.readthedocs.io/en/latest/configuration.html)
- [Module Documentation](https://hyfi.readthedocs.io/en/latest/modules/index.html)
- [API Reference](https://hyfi.readthedocs.io/en/latest/api/modules.html)

## ⚡ Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/sandrotruttmann/HyFI.git
cd HyFI

# Create conda environment
mamba env create -f hyfi.yaml
mamba activate hyfi

# Install package
pip install .
```

### Run Example

```bash
cd example_projects
hyfi run config_A0.json
```

Check `output_A0/` for results including:
- `HyFI_results.csv` - Main results table
- `3D_model.html` - Interactive 3D visualization
- `vtp_export/` - ParaView-compatible files

## 🛠️ Command-Line Interface

```bash
# Run analysis
hyfi run <config.json>

# Validate configuration
hyfi config validate <config.json>

# Create new project template
hyfi config create --format json --output my_project.json

# Show help
hyfi --help
```

## 💾 Input Data Format

HyFI requires:
- **Hypocenter data**: Relocated earthquake locations with uncertainties (hypoDD format)
- **Focal mechanisms** (optional): For model validation

See the [Input Parameters Guide](https://hyfi.readthedocs.io/en/latest/input_parameters.html) for detailed format specifications.

## 📈 Output

HyFI generates:
- `HyFI_results.csv` - Computed fault parameters for each event
- `3D_model.html` - Interactive Plotly visualization
- `vtp_export/` - VTK files for ParaView
- `execution_summary.json` - Analysis metadata and statistics

See the [Output Guide](https://hyfi.readthedocs.io/en/latest/output_guide.html) for complete documentation.

## 🤝 Contributing

HyFI is an open-source, community-driven project, and contributions of all kinds are warmly welcomed. Whether you want to report a bug, suggest an improvement, or open a Pull Request, we’re happy to collaborate with you!

## 📝 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or issues, please [open an issue](https://github.com/sandrotruttmann/HyFI/issues) on GitHub.
