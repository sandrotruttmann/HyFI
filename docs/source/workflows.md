# HyFI Processing Workflows

**HyFI** supports two main processing workflows: Single-Sequence and Multi-Sequence processing. Each workflow is suited for different analysis scenarios and catalog complexities.

---

## Single-Sequence Processing

Single-sequence processing analyzes all hypocenters in your catalog as one continuous seismic sequence. This is the standard workflow for analyzing a single earthquake swarm, cluster, or seismically active area.


### When to Use Single-Sequence Processing

- Analyzing a well-defined earthquake cluster or swarm
- Working with pre-filtered or already segmented seismic catalogs
- Studying a specific fault zone or localized seismicity


### Configuration Structure

To set up a single-sequence HyFI analysis, the user has to configure the **[HyFI Input Parameters](input_parameters)** using the single-sequence workflow DAG configuration structure. For detailed information on the structure of the configuration DAG please see the **[HyFI Configuration](configuration)**.


### Running Single-Sequence Analysis

Once the configuration file with the proper structure is setup and saved in JSON format, single-sequence processing can be started with:

```bash
hyfi run config_single_TEMPLATE.json
```

This will create a new directory as specified in the configuration file where all **[HyFI Output](output)** in the single-sequence format is saved to.

---

## Multi-Sequence Processing

Multi-sequence processing segments your catalog into distinct seismic sequences based on spatial, temporal, or spatiotemporal clustering, then analyzes each sequence independently. This is ideal for large-scale, complex, and noisy catalogs with multiple distinct earthquake clusters or regions.

### When to Use Multi-Sequence Processing

- Large catalogs covering extensive spatial regions
- Catalogs containing multiple distinct earthquake clusters
- Analyzing seismicity from multiple fault systems simultaneously
- Hierarchical analysis with different clustering scales
- Complex regional seismicity requiring segmentation


### Multi-Sequence Processing Explained

The segmentation process works hierarchically:

1. **Step 1: Data Loading**: Loading the full hypocenter catalog, incorporating multiple earthquake sequences.

2. **Step 2: Multi-Step Catalog Segmentation**: The hypocenter catalog is first divided into distinct sequences using clustering algorithms (DBSCAN, HDBSCAN, etc.). For this, multiple sequential clustering layers can be defined, where outliers from one step are passed to the next step with different parameters (e.g., tight clustering for Class A events, relaxed clustering for Class B events).

3. **Step 3: Per-Sequence Analysis**: Each segmented sequence is then analyzed independently with the full **HyFI** workflow (fault network reconstruction, classification, stress analysis, etc.).

4. **Step 4: Merge into HyFI Database**: Finally, the results of the individually processed clusters are merged into a common **HyFI** Database, incorporating metadata of identified faults, visualizations etc.


### Configuration Structure

To set up a multi-sequence HyFI analysis, the user has to configure the **[HyFI Input Parameters](input_parameters)** using the multi-sequence workflow DAG configuration structure. For detailed information on the structure of the configuration DAG please see the **[HyFI Configuration](configuration)**.


### Running Multi-Sequence Analysis

Once the configuration file with the proper structure is setup and saved in JSON format, multi-sequence processing can be started with:

```bash
hyfi run config_multi_TEMPLATE.json
```

This will create a new directory as specified in the configuration file where all **[HyFI Output](output)** in the multi-sequence format is saved to. Note that the multi-sequence outputs incorporates a nested structure of the individually processed clusters.


---

Happy fault imaging! 🎉
