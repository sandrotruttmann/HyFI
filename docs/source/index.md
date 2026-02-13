# Welcome to HyFI's documentation!

[![DOI](https://zenodo.org/badge/582537470.svg)](https://zenodo.org/badge/latestdoi/582537470)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/sandrotruttmann/HyFI)

**HyFI** (Hypocenter-based 3D Imaging of Active Faults) is a Python package for imaging active faults in 3D based on relatively relocated hypocenter datasets.


## Scientific Publication

If you use the code in this repository please cite the following scientific publication:

Truttmann, S., Diehl, T., Herwegh, M. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps. *Journal of Geophysical Research: Solid Earth*. https://doi.org/10.1029/2023JB026352

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation.md
quickstart.md
```

```{toctree}
:maxdepth: 3
:caption: User Guide
workflows.md
configuration.md
input_parameters.md
modules/index.md
terminology.md
output.md
querying.md
```

```{toctree}
:maxdepth: 1
:caption: Examples

single_examples.md
multi_examples.md
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/modules
```

```{toctree}
:maxdepth: 1
:caption: Additional Resources

license.md
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`



- **[Processing Workflow](workflows)**: Understand the different processing workflows of HyFI to analyze simple or more complex earthquake catalogs
- **[Configuration Guide](configuration)**: Learn how to set up JSON configuration files for your analysis
- **[Input Parameters](input_parameters)**: Detailed documentation of all configuration parameters
- **[HyFI Modules](modules/index)**: Understand the different analysis modules (fault network, validation, classification, stress analysis)
- **[Output Guide](output)**: Learn about the generated output files and visualizations
- **[Queries](querying)**: Learn how to efficiently query the HyFI results using SQL and DuckDB.
- **[Examples](examples)**: Explore example projects and use cases

---

Happy fault imaging! 🎉
