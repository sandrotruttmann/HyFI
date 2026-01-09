#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template creation utilities for multi-sequence workflows.

This module provides functions to create template configurations
for multi-sequence fault imaging analysis.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from .multi_sequence_config import MultiSequenceConfig, ClusteringConfig, SegmentationStep
from .parameters import ProjectConfig


def create_multi_sequence_template(
    output_file: str,
    catalog_file: str = "path/to/full_catalog.txt",
    project_title: str = "Multi-Sequence Fault Imaging",
    output_dir: str = "./multi_sequence_output",
    clustering_method: str = "dbscan",
    multi_step: bool = False
) -> MultiSequenceConfig:
    """
    Create a template multi-sequence configuration and save to file.
    
    Parameters
    ----------
    output_file : str
        Path to save the template configuration
    catalog_file : str
        Path to the full earthquake catalog
    project_title : str
        Project title
    output_dir : str
        Output directory for results
    clustering_method : str
        Primary clustering method to use
    multi_step : bool
        Whether to create a multi-step clustering template
        
    Returns
    -------
    MultiSequenceConfig
        The created configuration object
    """
    
    # Create template single-sequence configuration
    template_config = ProjectConfig(
        project_title="Template for Individual Clusters",
        hypo_file=Path(""),  # Will be set dynamically
        hypo_sep='\t',
        out_dir=Path("")  # Will be set dynamically
    )
    
    # Configure modules with reasonable defaults
    template_config.fault_network.n_mc = 1000
    template_config.fault_network.r_nn = 100.0
    template_config.fault_network.dt_nn = 26298.0
    
    template_config.model_validation.validation_bool = True
    template_config.auto_class.autoclass_bool = True
    template_config.stress_analysis.stress_bool = True
    
    # Create segmentation steps
    segmentation_steps = []
    
    if multi_step:
        # Create multi-step configuration
        segmentation_steps = [
            SegmentationStep(
                step_name="primary_clustering",
                description=f"Primary {clustering_method} clustering",
                method=clustering_method,
                features=['spatial'],
                dbscan_eps=1000.0,
                dbscan_min_samples=10,
                min_cluster_size=25,
                process_outliers=True,
                outlier_handling='next_step'
            ),
            SegmentationStep(
                step_name="secondary_clustering", 
                description="Secondary clustering with tighter parameters",
                method=clustering_method,
                features=['spatial'],
                dbscan_eps=500.0,
                dbscan_min_samples=5,
                min_cluster_size=10,
                process_outliers=True,
                outlier_handling='next_step'
            ),
            SegmentationStep(
                step_name="tertiary_temporal",
                description="Temporal clustering for remaining outliers",
                method='temporal',
                features=['temporal'],
                temporal_window_days=30,
                min_cluster_size=5,
                process_outliers=False,
                outlier_handling='keep'
            )
        ]
    else:
        # Single-step configuration
        segmentation_steps = [
            SegmentationStep(
                step_name="primary_clustering",
                description=f"Primary {clustering_method} clustering",
                method=clustering_method,
                features=['spatial'],
                dbscan_eps=1000.0,
                dbscan_min_samples=10,
                min_cluster_size=20,
                process_outliers=False,
                outlier_handling='keep'
            )
        ]
    
    # Create clustering configuration
    clustering_config = ClusteringConfig(
        segmentation_steps=segmentation_steps,
        final_outlier_handling='keep',
        max_outlier_ratio=0.3
    )
    
    # Create multi-sequence configuration
    multi_config = MultiSequenceConfig(
        project_title=project_title,
        catalog_file=Path(catalog_file),
        catalog_sep='\t',
        output_directory=Path(output_dir),
        clustering=clustering_config,
        template_config=template_config
    )
    
    # Save to file
    save_multi_sequence_config(multi_config, output_file)
    
    return multi_config


def create_advanced_multi_step_template(
    output_file: str,
    catalog_file: str = "path/to/full_catalog.txt",
    project_title: str = "Advanced Multi-Step Analysis",
    output_dir: str = "./advanced_multi_step_output"
) -> MultiSequenceConfig:
    """
    Create an advanced multi-step segmentation template with various clustering methods.
    
    This template demonstrates the full capabilities of multi-step segmentation,
    combining different clustering algorithms and feature sets.
    """
    
    # Create template single-sequence configuration
    template_config = ProjectConfig(
        project_title="Template for Individual Clusters",
        hypo_file=Path(""),
        hypo_sep='\t',
        out_dir=Path("")
    )
    
    # Configure with optimized parameters for multi-step analysis
    template_config.fault_network.n_mc = 500  # Reduced for faster processing
    template_config.fault_network.r_nn = 100.0
    template_config.fault_network.dt_nn = 26298.0
    
    template_config.model_validation.validation_bool = True
    template_config.auto_class.autoclass_bool = True
    template_config.stress_analysis.stress_bool = True
    
    # Create advanced segmentation steps
    segmentation_steps = [
        # Step 1: Large-scale spatial clustering
        SegmentationStep(
            step_name="macro_spatial",
            description="Macro-scale spatial clustering for major faults",
            method='dbscan',
            features=['spatial'],
            dbscan_eps=2000.0,  # Large search radius
            dbscan_min_samples=15,
            min_cluster_size=50,
            process_outliers=True,
            outlier_handling='next_step'
        ),
        
        # Step 2: Medium-scale spatial clustering
        SegmentationStep(
            step_name="meso_spatial",
            description="Meso-scale spatial clustering for fault segments", 
            method='dbscan',
            features=['spatial'],
            dbscan_eps=1000.0,
            dbscan_min_samples=10,
            min_cluster_size=20,
            process_outliers=True,
            outlier_handling='next_step'
        ),
        
        # Step 3: Fine-scale spatial clustering
        SegmentationStep(
            step_name="micro_spatial",
            description="Micro-scale spatial clustering for detailed structures",
            method='dbscan',
            features=['spatial'],
            dbscan_eps=500.0,
            dbscan_min_samples=5,
            min_cluster_size=8,
            process_outliers=True,
            outlier_handling='next_step'
        ),
        
        # Step 4: Temporal clustering for remaining events
        SegmentationStep(
            step_name="temporal_sequences",
            description="Temporal clustering for seismic sequences",
            method='temporal',
            features=['temporal'],
            temporal_window_days=60,
            min_cluster_size=5,
            process_outliers=True,
            outlier_handling='next_step'
        ),
        
        # Step 5: HDBSCAN for complex patterns
        SegmentationStep(
            step_name="final_hdbscan",
            description="HDBSCAN for complex remaining patterns",
            method='hdbscan',
            features=['spatial', 'temporal'],
            hdbscan_min_cluster_size=5,
            hdbscan_min_samples=3,
            spatial_weight=0.7,
            min_cluster_size=3,
            process_outliers=False,
            outlier_handling='keep'
        )
    ]
    
    # Create clustering configuration
    clustering_config = ClusteringConfig(
        segmentation_steps=segmentation_steps,
        final_outlier_handling='keep',
        max_outlier_ratio=0.15  # Stricter outlier threshold
    )
    
    # Create multi-sequence configuration
    multi_config = MultiSequenceConfig(
        project_title=project_title,
        catalog_file=Path(catalog_file),
        catalog_sep='\t',
        output_directory=Path(output_dir),
        clustering=clustering_config,
        template_config=template_config
    )
    
    # Save to file
    save_multi_sequence_config(multi_config, output_file)
    
    return multi_config


def create_multi_sequence_dag_template(
    output_file: str,
    catalog_file: str = "path/to/full_catalog.txt",
    project_title: str = "Multi-Sequence DAG Analysis",
    output_dir: str = "./multi_sequence_dag_output"
) -> Dict[str, Any]:
    """
    Create a template multi-sequence DAG configuration.
    
    Parameters
    ----------
    output_file : str
        Path to save the template DAG
    catalog_file : str
        Path to the full earthquake catalog
    project_title : str
        Project title
    output_dir : str
        Output directory for results
        
    Returns
    -------
    Dict[str, Any]
        The created DAG configuration
    """
    
    from .multi_sequence_schema import MultiSequenceWorkflowDAG
    
    # Create DAG configuration
    dag = MultiSequenceWorkflowDAG(
        workflow_name=project_title,
        output_directory=output_dir
    )
    
    # Update input data node
    dag.nodes["input_data"].hypocenter_file = catalog_file
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(dag.to_dict(), f, indent=2)
    
    return dag.to_dict()


def save_multi_sequence_config(config: MultiSequenceConfig, output_file: str):
    """Save multi-sequence configuration to JSON file."""
    
    config_dict = config.to_dict()
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Multi-sequence configuration template saved to: {output_path}")


def load_multi_sequence_config(config_file: str) -> MultiSequenceConfig:
    """Load multi-sequence configuration from JSON file."""
    
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    return MultiSequenceConfig.from_dict(config_dict)


def convert_single_to_multi_sequence_template(
    single_config_file: str,
    output_file: str,
    catalog_file: str,
    clustering_method: str = "dbscan"
) -> MultiSequenceConfig:
    """
    Convert a single-sequence configuration to a multi-sequence template.
    
    Parameters
    ----------
    single_config_file : str
        Path to existing single-sequence configuration
    output_file : str
        Path to save the multi-sequence template
    catalog_file : str
        Path to the full earthquake catalog
    clustering_method : str
        Clustering method to use
        
    Returns
    -------
    MultiSequenceConfig
        The created multi-sequence configuration
    """
    
    from .io import auto_load_config
    
    # Load existing single-sequence configuration
    single_config = auto_load_config(single_config_file)
    
    # Create clustering configuration
    clustering_config = ClusteringConfig(
        method=clustering_method,
        features=['spatial'],
        dbscan_eps=1000.0,
        dbscan_min_samples=10,
        min_cluster_size=20
    )
    
    # Create multi-sequence configuration using the single config as template
    multi_config = MultiSequenceConfig(
        project_title=f"Multi-Sequence: {single_config.project_title}",
        catalog_file=Path(catalog_file),
        catalog_sep=single_config.hypo_sep,
        output_directory=Path(single_config.out_dir).parent / "multi_sequence_output",
        clustering=clustering_config,
        template_config=single_config
    )
    
    # Save to file
    save_multi_sequence_config(multi_config, output_file)
    
    return multi_config


def create_example_configs():
    """Create example configurations for different clustering scenarios."""
    
    examples_dir = Path("example_multi_sequence_configs")
    examples_dir.mkdir(exist_ok=True)
    
    # Example 1: DBSCAN spatial clustering
    create_multi_sequence_template(
        output_file=str(examples_dir / "dbscan_spatial.json"),
        catalog_file="data/full_catalog.txt",
        project_title="DBSCAN Spatial Clustering",
        clustering_method="dbscan"
    )
    
    # Example 2: HDBSCAN clustering  
    config = create_multi_sequence_template(
        output_file=str(examples_dir / "hdbscan.json"),
        catalog_file="data/full_catalog.txt", 
        project_title="HDBSCAN Clustering",
        clustering_method="hdbscan"
    )
    config.clustering.hdbscan_min_cluster_size = 25
    save_multi_sequence_config(config, str(examples_dir / "hdbscan.json"))
    
    # Example 3: Temporal clustering
    config = create_multi_sequence_template(
        output_file=str(examples_dir / "temporal.json"),
        catalog_file="data/full_catalog.txt",
        project_title="Temporal Clustering", 
        clustering_method="temporal"
    )
    config.clustering.temporal_window_days = 60
    save_multi_sequence_config(config, str(examples_dir / "temporal.json"))
    
    # Example 4: Spatial-temporal clustering
    config = create_multi_sequence_template(
        output_file=str(examples_dir / "spatial_temporal.json"),
        catalog_file="data/full_catalog.txt",
        project_title="Spatial-Temporal Clustering",
        clustering_method="spatial_temporal"
    )
    config.clustering.features = ['spatial', 'temporal']
    config.clustering.spatial_weight = 0.6
    save_multi_sequence_config(config, str(examples_dir / "spatial_temporal.json"))
    
    print(f"Example configurations created in: {examples_dir}")


if __name__ == "__main__":
    # Create example configurations when run as script
    create_example_configs()
