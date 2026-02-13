# HyFI Core Modules

**HyFI** consists of several interconnected analysis modules that work together to reconstruct 3D fault geometry from earthquake hypocenter data.

## Core Analysis Modules

The following modules form the core of the **HyFI** analysis pipeline:

```{toctree}
:maxdepth: 1

fault_network
model_validation
auto_classification
stress_analysis
visualization
```

## Core Module Execution Order

A typical single-sequence HyFI analysis follows this sequence of module execution:

1. **Fault Network Reconstruction**: Generate 3D rupture planes from hypocenter distributions for each earthquake individually
2. **Model Validation** (optional): Compare results of rupture planes with focal planes for earthquakes with a focal mechanism solution
3. **Automatic Classification** (optional): Group earthquakes into clusters that likely belong to the same active fault by clustering both the rupture plane orientations as well as spatial attributes
4. **Stress Analysis** (optional): Calculate stress parameters (e.g. slip and dilation tendencies) under the given (regional) stress field
5. **Visualization**: Interpolate the grouped rupture planes to produce active fault meshes, generate interactive 3D models and export the results (e.g. as VTP)

Each module can be configured independently through the JSON configuration file. See **[HyFI Configuration](../configuration)** for details.

---

Happy fault imaging! 🎉
