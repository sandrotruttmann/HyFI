# HyFI Modules

**HyFI** consists of several interconnected analysis modules that work together to reconstruct 3D fault geometry from earthquake hypocenter data.

## Core Analysis Modules

The following modules form the main analysis pipeline:

```{toctree}
:maxdepth: 2

fault_network
model_validation
auto_classification
stress_analysis
visualization
```

## Workflow

A typical HyFI analysis follows this sequence:

1. **Fault Network Reconstruction**: Generate 3D fault planes from hypocenter distributions
2. **Model Validation** (optional): Compare results with focal mechanism solutions
3. **Automatic Classification** (optional): Group fault planes into orientation clusters
4. **Stress Analysis** (optional): Calculate slip tendencies under regional stress field
5. **Visualization**: Generate interactive 3D models and export results

Each module can be configured independently through the JSON configuration file. See the [Configuration Guide](../configuration) for details.
