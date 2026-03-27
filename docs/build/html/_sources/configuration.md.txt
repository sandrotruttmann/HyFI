# HyFI Configuration

**HyFI** uses JSON-based DAG (Directed Acyclic Graph) configurations to define analysis workflows. This approach provides clear structure, validation, and reproducibility without requiring Python code.

:::{note}
For information about different processing workflows (single-sequence vs multi-sequence), see the [Workflows Guide](workflows.md).
:::

---

## Configuration File Structure

All **HyFI** configurations consist of three main sections:

1. **`metadata`**: Workflow identification and documentation
2. **`global_settings`**: Common settings applied across the entire workflow
3. **`workflow_dag`**: Analysis steps and their parameters (also defines whether the DAG refers to single- or multi-sequence processing)

---

## Single-Sequence DAG Configuration

Single-sequence configurations analyze all hypocenters as one continuous earthquake sequence.

### Main Sections

The workflow DAG contains the following analysis steps:

- **`input_data`**: Load hypocenter catalog and optional focal mechanism data
- **`fault_network`**: Reconstruct 3D fault network using Monte Carlo simulation and PCA
- **`model_validation`** *(optional)*: Validate computed fault planes against focal mechanisms
- **`auto_classification`** *(optional)*: Automatic fault structure classification
- **`stress_analysis`** *(optional)*: Stress field analysis and fault stability assessment  
- **`visualization`** *(optional)*: Generate 3D visualizations and summary plots

### Minimal Configuration Example

```json
{
  "metadata": {
    "workflow_name": "Single Sequence Analysis",
    "workflow_version": "1.0.0"
  },
  "global_settings": {
    "output_directory": "./output_single",
    "log_level": "INFO"
  },
  "workflow_dag": {
    "input_data": {
      "hypocenter_file": "data_examples/A0_data.csv",
      "hypocenter_separator": ",",
      "focal_mechanism_file": "data_examples/A0_focals.csv",
      "focal_mechanism_separator": ","
    },
    "fault_network": {
      "parameters": {
        "core_network": {
          "search_radius_meters": 100.0,
          "search_time_window_hours": 9999999
        }
      }
    },
    "model_validation": {
      "enabled": true
    },
    "auto_classification": {
      "enabled": true
    },
    "stress_analysis": {
      "enabled": true,
      "parameters": {
        "regional_stress": {
          "sigma_1_azimuth": 144.0,
          "sigma_1_plunge": 6.0,
          "sigma_3_azimuth": 51.0,
          "sigma_3_plunge": 22.0,
        }
      }
    },
    "visualization": {
      "enabled": true
    }
  }
}
```

---

## Multi-Sequence DAG Configuration

Multi-sequence configurations segment the catalog into distinct sequences using clustering, then analyze each sequence independently.

### Main Sections

In comparison to the single-sequence processing, the workflow DAG for multi-sequence processing incorporates an additional hierarchical level that defines the step-wise segmentation and processing parameters of the individual sequences:

- **`step_1_load_data`**: Load the full hypocenter catalog and optional focal mechanisms
- **`step_2_catalog_segmentation`**: Segment catalog into sequences using clustering algorithms (e.g., DBSCAN, HDBSCAN)
- **`step_3_per_sequence_analysis`**: Apply **HyFI** analysis workflow to each segmented sequence
  - Contains the same sub-steps as single-sequence: `fault_network`, `model_validation`, `auto_classification`, `stress_analysis`, `visualization`
- **`step_4_merge_and_export`**: Merge all sequence results into a unified database and export combined visualizations

### Minimal Configuration Example

```json
{
  "metadata": {
    "workflow_name": "Multi-Sequence Analysis",
    "workflow_version": "1.0.0"
  },
  "global_settings": {
    "output_directory": "./output_multi",
    "parallel_processing": true,
    "max_workers": 4
  },
  "workflow_dag": {
    "step_1_load_data": {
      "hypocenter_file": "data_examples/SECOS_20250305_HyFI.csv",
      "hypocenter_separator": ","
    },
    "step_2_catalog_segmentation": {
      "enabled": true,
      "segmentation_steps": [
        {
          "step_name": "Class_A",
          "method": "dbscan",
          "features": ["spatial"],
          "cluster_dimension": "3d",
          "dbscan_eps": 350.0,
          "dbscan_min_samples": 10,
          "min_cluster_size": 20,
          "outlier_handling": "next_step"
        }
      ],
      "final_outlier_handling": "keep"
    },
    "step_3_per_sequence_analysis": {
      "fault_network": {
        "parameters": {
          "core_network": {
            "search_radius_meters": 100.0,
            "search_time_window_hours": 9999999
          }
        }
      },
      "model_validation": {
        "enabled": true
      },
      "auto_classification": {
        "enabled": true
      },
      "stress_analysis": {
        "enabled": true,
        "parameters": {
          "regional_stress": {
            "sigma_1_azimuth": 144.0,
            "sigma_1_plunge": 6.0,
            "sigma_3_azimuth": 51.0,
            "sigma_3_plunge": 22.0
          }
        }
      },
      "visualization": {
        "enabled": true
      }
    },
    "step_4_merge_and_export": {
      "enabled": true,
      "merge_vtp_files": true,
      "sql_database": {
        "enabled": true,
        "database_path": "./output_multi/HyFI_Database/hyfi_results.db"
      }
    }
  }
}
```

---

## Complete Parameter Reference

For a complete list of all available parameters and detailed explanations, see the [Input Parameters](input_parameters.md) documentation.

---

Happy fault imaging! 🎉
