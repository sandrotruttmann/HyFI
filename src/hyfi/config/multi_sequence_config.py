#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration class for multi-sequence workflows.

This module extends the existing configuration system to support
multi-sequence analysis with clustering capabilities.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from pathlib import Path

from .parameters import ProjectConfig
from .validation import (
    ConfigValidationError,
    validate_file_exists,
    validate_positive_number,
    validate_range,
    validate_choice,
    validate_separator,
    validate_project_title,
    validate_output_directory
)


@dataclass
class SegmentationStep:
    """Configuration for a single segmentation step."""
    
    # Step identification
    step_name: str = "step_1"
    description: str = ""
    
    # Clustering method
    method: str = 'dbscan'  # 'dbscan', 'hdbscan', 'temporal', 'spatial_temporal'
    
    # Features to use for clustering
    features: List[str] = field(default_factory=lambda: ['spatial'])  # 'spatial', 'temporal', 'magnitude'
    
    # DBSCAN parameters
    dbscan_eps: float = 1000.0  # Maximum distance between samples (meters for spatial)
    dbscan_min_samples: int = 10  # Minimum samples per cluster
    dbscan_metric: str = 'euclidean'
    
    # HDBSCAN parameters  
    hdbscan_min_cluster_size: int = 15
    hdbscan_min_samples: Optional[int] = None
    
    # Temporal clustering parameters
    temporal_window_days: int = 30  # Time window for temporal clustering
    
    # Spatial-temporal weighting
    spatial_weight: float = 0.7  # Weight for spatial vs temporal features (0-1)
    
    # Coordinate system options
    use_raw_coordinates: bool = False  # Use raw coordinates without normalization (old implementation style)
    
    # Minimum cluster size to process
    min_cluster_size: int = 20
    
    # What to do with outliers from this step
    process_outliers: bool = True  # Whether to pass outliers to next step
    outlier_handling: str = 'next_step'  # 'next_step', 'merge_smallest', 'discard'
    
    def validate(self):
        """Validate segmentation step parameters."""
        validate_choice(self.method, ['dbscan', 'hdbscan', 'temporal', 'spatial_temporal'], f"{self.step_name} clustering method")
        
        valid_features = ['spatial', 'temporal', 'magnitude']
        for feature in self.features:
            if feature not in valid_features:
                raise ConfigValidationError(f"Invalid clustering feature in {self.step_name}: {feature}. Must be one of {valid_features}")
        
        if len(self.features) == 0:
            raise ConfigValidationError(f"At least one clustering feature must be specified for {self.step_name}")
        
        validate_positive_number(self.dbscan_eps, f"{self.step_name}_dbscan_eps")
        validate_positive_number(self.dbscan_min_samples, f"{self.step_name}_dbscan_min_samples")
        validate_positive_number(self.hdbscan_min_cluster_size, f"{self.step_name}_hdbscan_min_cluster_size")
        validate_positive_number(self.temporal_window_days, f"{self.step_name}_temporal_window_days")
        validate_range(self.spatial_weight, 0, 1, f"{self.step_name}_spatial_weight")
        validate_positive_number(self.min_cluster_size, f"{self.step_name}_min_cluster_size")
        
        validate_choice(self.outlier_handling, ['next_step', 'merge_smallest', 'discard'], f"{self.step_name}_outlier_handling")


@dataclass
class ClusteringConfig:
    """Configuration for multi-step catalog clustering/segmentation."""
    
    # Multi-step segmentation
    segmentation_steps: List[SegmentationStep] = field(default_factory=lambda: [
        SegmentationStep(
            step_name="primary_clustering",
            description="Primary spatial clustering with DBSCAN",
            method='dbscan',
            features=['spatial'],
            dbscan_eps=1000.0,
            dbscan_min_samples=10,
            min_cluster_size=20,
            process_outliers=True
        )
    ])
    
    # Global outlier handling
    final_outlier_handling: str = 'keep'  # 'keep', 'discard', 'merge_largest'
    max_outlier_ratio: float = 0.3  # Maximum allowed ratio of outliers to total events
    
    def validate(self):
        """Validate clustering configuration parameters."""
        if not self.segmentation_steps:
            raise ConfigValidationError("At least one segmentation step must be defined")
        
        # Validate each segmentation step
        for step in self.segmentation_steps:
            step.validate()
        
        # Validate step names are unique
        step_names = [step.step_name for step in self.segmentation_steps]
        if len(step_names) != len(set(step_names)):
            raise ConfigValidationError("Segmentation step names must be unique")
        
        # Validate global parameters
        validate_choice(self.final_outlier_handling, ['keep', 'discard', 'merge_largest'], "final_outlier_handling")
        validate_range(self.max_outlier_ratio, 0, 1, "max_outlier_ratio")
    
    def get_step(self, step_name: str) -> Optional[SegmentationStep]:
        """Get a segmentation step by name."""
        for step in self.segmentation_steps:
            if step.step_name == step_name:
                return step
        return None
    
    def add_step(self, step: SegmentationStep):
        """Add a new segmentation step."""
        # Validate step name is unique
        if any(s.step_name == step.step_name for s in self.segmentation_steps):
            raise ConfigValidationError(f"Step name '{step.step_name}' already exists")
        
        self.segmentation_steps.append(step)
    
    # Backward compatibility properties
    @property
    def method(self) -> str:
        """Get primary clustering method (backward compatibility)."""
        return self.segmentation_steps[0].method if self.segmentation_steps else 'dbscan'
    
    @property
    def features(self) -> List[str]:
        """Get primary clustering features (backward compatibility)."""
        return self.segmentation_steps[0].features if self.segmentation_steps else ['spatial']
    
    @property
    def dbscan_eps(self) -> float:
        """Get primary DBSCAN eps (backward compatibility)."""
        return self.segmentation_steps[0].dbscan_eps if self.segmentation_steps else 1000.0
    
    @property
    def dbscan_min_samples(self) -> int:
        """Get primary DBSCAN min_samples (backward compatibility)."""
        return self.segmentation_steps[0].dbscan_min_samples if self.segmentation_steps else 10
    
    @property
    def min_cluster_size(self) -> int:
        """Get primary min cluster size (backward compatibility)."""
        return self.segmentation_steps[0].min_cluster_size if self.segmentation_steps else 20


@dataclass
class MultiSequenceConfig:
    """Configuration for multi-sequence fault imaging workflow."""
    
    # General settings
    project_title: str = "Multi-Sequence Fault Imaging Analysis"
    
    # Input catalog file
    catalog_file: Optional[Union[str, Path]] = None
    catalog_sep: str = '\t'
    
    # Output settings
    output_directory: Union[str, Path] = field(default_factory=lambda: Path.cwd() / "multi_sequence_output")
    
    # Clustering configuration
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    
    # Template configuration for individual cluster analysis
    template_config: ProjectConfig = field(default_factory=ProjectConfig)
    
    # DAG-based cluster workflow template (new format)
    cluster_workflow_template: Optional[dict] = None
    
    # Multi-sequence specific options
    parallel_processing: bool = False  # Future feature
    max_workers: int = 4  # For parallel processing
    save_individual_results: bool = True
    
    # Coordinate system for KML export
    coordinate_system: str = "EPSG:21781"  # Input coordinate system for KML export
    
    def __post_init__(self):
        """Convert string paths to Path objects and validate basic properties."""
        if isinstance(self.catalog_file, str):
            self.catalog_file = Path(self.catalog_file)
        if isinstance(self.output_directory, str):
            self.output_directory = Path(self.output_directory)
        
        # Convert relative paths to absolute paths in template config to avoid issues
        # when processing clusters from subdirectories
        if self.template_config.model_validation.foc_file:
            foc_path = Path(self.template_config.model_validation.foc_file)
            if not foc_path.is_absolute():
                # Convert relative path to absolute path relative to current working directory
                self.template_config.model_validation.foc_file = foc_path.resolve()
        
        # Basic validation on initialization
        self.project_title = validate_project_title(self.project_title)
        self.catalog_sep = validate_separator(self.catalog_sep)
    
    @property
    def clustering_method(self) -> str:
        """Get the primary clustering method (for backward compatibility)."""
        return self.clustering.method
    
    @property
    def clustering_features(self) -> List[str]:
        """Get the primary clustering features (for backward compatibility)."""
        return self.clustering.features
    
    @property
    def segmentation_steps(self) -> List[SegmentationStep]:
        """Get all segmentation steps."""
        return self.clustering.segmentation_steps
    
    def add_segmentation_step(self, 
                             step_name: str,
                             method: str = 'dbscan',
                             features: List[str] = None,
                             **kwargs) -> SegmentationStep:
        """
        Add a new segmentation step.
        
        Parameters
        ----------
        step_name : str
            Name for the segmentation step
        method : str
            Clustering method
        features : List[str]
            Features to use for clustering
        **kwargs
            Additional parameters for the segmentation step
            
        Returns
        -------
        SegmentationStep
            The created segmentation step
        """
        if features is None:
            features = ['spatial']
        
        step = SegmentationStep(
            step_name=step_name,
            method=method,
            features=features,
            **kwargs
        )
        
        self.clustering.add_step(step)
        return step
    
    def add_segmentation_step(self, 
                             step_name: str,
                             method: str = 'dbscan',
                             features: List[str] = None,
                             **kwargs) -> SegmentationStep:
        """
        Add a new segmentation step.
        
        Parameters
        ----------
        step_name : str
            Name for the segmentation step
        method : str
            Clustering method
        features : List[str]
            Features to use for clustering
        **kwargs
            Additional parameters for the segmentation step
            
        Returns
        -------
        SegmentationStep
            The created segmentation step
        """
        if features is None:
            features = ['spatial']
        
        step = SegmentationStep(
            step_name=step_name,
            method=method,
            features=features,
            **kwargs
        )
        
        self.clustering.add_step(step)
        return step
    
    def validate(self):
        """Validate all configuration parameters."""
        # Validate file paths
        validate_file_exists(self.catalog_file, required=True)
        self.output_directory = validate_output_directory(self.output_directory)
        
        # Validate clustering configuration
        self.clustering.validate()
        
        # Validate template configuration (skip file path validation as they are set dynamically)
        # Note: We don't validate file paths in template_config as they will be created dynamically
        if hasattr(self.template_config, 'fault_network'):
            self.template_config.fault_network.validate()
        
        # For model validation, temporarily disable file validation
        if hasattr(self.template_config, 'model_validation'):
            # Save original foc_file value
            original_foc_file = self.template_config.model_validation.foc_file
            # Temporarily set to None to skip validation
            self.template_config.model_validation.foc_file = None
            self.template_config.model_validation.validate()
            # Restore original value
            self.template_config.model_validation.foc_file = original_foc_file
        
        if hasattr(self.template_config, 'auto_class'):
            self.template_config.auto_class.validate()
        if hasattr(self.template_config, 'stress_analysis'):
            self.template_config.stress_analysis.validate()
        
        # Validate multi-sequence options
        if self.parallel_processing:
            validate_positive_number(self.max_workers, "max_workers")
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MultiSequenceConfig':
        """Create MultiSequenceConfig from dictionary."""
        
        # Handle unified DAG format
        if 'workflow_dag' in config_dict and 'step_2_catalog_segmentation' in config_dict['workflow_dag']:
            return cls._from_unified_dag_format(config_dict)
        
        # Handle legacy format with clustering and template_config/cluster_workflow_template
        else:
            return cls._from_legacy_format(config_dict)
    
    @classmethod
    def _from_unified_dag_format(cls, config_dict: dict) -> 'MultiSequenceConfig':
        """Create MultiSequenceConfig from unified DAG format."""
        workflow_dag = config_dict['workflow_dag']
        clustering_dict = workflow_dag['step_2_catalog_segmentation']
        
        # Create segmentation steps from the configuration
        segmentation_steps = []
        if 'segmentation_steps' in clustering_dict:
            for step_dict in clustering_dict['segmentation_steps']:
                step = SegmentationStep(**step_dict)
                segmentation_steps.append(step)
        
        # Create clustering configuration
        clustering_config = ClusteringConfig(
            segmentation_steps=segmentation_steps,
            final_outlier_handling=clustering_dict.get('final_outlier_handling', 'keep'),
            max_outlier_ratio=clustering_dict.get('max_outlier_ratio', 0.3)
        )
        
        # Get input data configuration (support both step_1_load_data and legacy input_data)
        input_data_config = workflow_dag.get('step_1_load_data', workflow_dag.get('input_data', {}))
        
        # Create a minimal template_config for backward compatibility
        template_dict = {
            'project_title': 'Template for Individual Clusters',
            'hypo_file': '',
            'hypo_sep': input_data_config.get('hypocenter_separator', ','),
            'out_dir': '',
            'n_mc': 1,
            'r_nn': 200.0,
            'dt_nn': 999999.0,
            'mag_type': 'ML',
            'validation_bool': False,
            'foc_file': input_data_config.get('focal_mechanism_file', ''),
            'foc_sep': input_data_config.get('focal_mechanism_separator', ','),
            'foc_mag_check': True,
            'foc_loc_check': True,
            'autoclass_bool': False,
            'n_clusters': 2,
            'algorithm': 'vmf_soft',
            'rotation': True,
            'stress_bool': False,
            'S1_trend': 301,
            'S1_plunge': 23,
            'S3_trend': 43,
            'S3_plunge': 26,
            'stress_R': 0.35,
            'PP': 0,
            'fric_coeff': 0.75
        }
        
        template_config = ProjectConfig.from_dict(template_dict)
        
        # Store the entire workflow_dag as the cluster_workflow_template
        cluster_workflow_template = config_dict.copy()
        # Remove the catalog segmentation and merge/export steps for individual clusters
        if 'workflow_dag' in cluster_workflow_template:
            if 'step_2_catalog_segmentation' in cluster_workflow_template['workflow_dag']:
                del cluster_workflow_template['workflow_dag']['step_2_catalog_segmentation']
            if 'step_4_merge_and_export' in cluster_workflow_template['workflow_dag']:
                del cluster_workflow_template['workflow_dag']['step_4_merge_and_export']
        
        # Get paths and resolve relative to config file if needed
        catalog_file = input_data_config.get('hypocenter_file', '')
        output_dir = config_dict.get('global_settings', {}).get('output_directory', Path.cwd() / "multi_sequence_output")
        
        # Store these for resolution in __post_init__ if needed
        return cls(
            project_title=config_dict.get('metadata', {}).get('workflow_name', 'Multi-Sequence Fault Imaging Analysis'),
            catalog_file=catalog_file,
            catalog_sep=input_data_config.get('hypocenter_separator', '\t'),
            output_directory=output_dir,
            clustering=clustering_config,
            template_config=template_config,
            cluster_workflow_template=cluster_workflow_template,
            parallel_processing=config_dict.get('global_settings', {}).get('parallel_processing', False),
            max_workers=config_dict.get('global_settings', {}).get('max_workers', 4),
            save_individual_results=config_dict.get('global_settings', {}).get('save_individual_results', True),
            coordinate_system=config_dict.get('global_settings', {}).get('coordinate_system', 'EPSG:2056')
        )
    
    @classmethod
    def _from_legacy_format(cls, config_dict: dict) -> 'MultiSequenceConfig':
        """Create MultiSequenceConfig from legacy format."""
        # Extract clustering configuration
        clustering_dict = config_dict.get('clustering', {})
        
        # Create segmentation steps from the configuration
        segmentation_steps = []
        if 'segmentation_steps' in clustering_dict:
            for step_dict in clustering_dict['segmentation_steps']:
                step = SegmentationStep(**step_dict)
                segmentation_steps.append(step)
        
        # Create clustering configuration
        clustering_config = ClusteringConfig(
            segmentation_steps=segmentation_steps,
            final_outlier_handling=clustering_dict.get('final_outlier_handling', 'keep'),
            max_outlier_ratio=clustering_dict.get('max_outlier_ratio', 0.3)
        )
        
        # Extract template configuration (for individual cluster processing)
        template_dict = config_dict.get('template_config', {})
        
        # If no template_config but cluster_workflow_template exists, create a minimal template_config
        if not template_dict and config_dict.get('cluster_workflow_template'):
            template_dict = {
                'project_title': 'Template for Individual Clusters',
                'hypo_file': '',
                'hypo_sep': ',',
                'out_dir': '',
                'n_mc': 1,
                'r_nn': 200.0,
                'dt_nn': 999999.0,
                'mag_type': 'ML',
                'validation_bool': False,
                'foc_file': '',
                'foc_sep': ',',
                'foc_mag_check': True,
                'foc_loc_check': True,
                'autoclass_bool': False,
                'n_clusters': 2,
                'algorithm': 'vmf_soft',
                'rotation': True,
                'stress_bool': False,
                'S1_trend': 301,
                'S1_plunge': 23,
                'S3_trend': 43,
                'S3_plunge': 26,
                'stress_R': 0.35,
                'PP': 0,
                'fric_coeff': 0.75
            }
        
        template_config = ProjectConfig.from_dict(template_dict)
        
        # Extract cluster workflow template (new DAG format)
        cluster_workflow_template = config_dict.get('cluster_workflow_template', None)
        
        return cls(
            project_title=config_dict.get('project_title', 'Multi-Sequence Fault Imaging Analysis'),
            catalog_file=config_dict.get('catalog_file', ''),
            catalog_sep=config_dict.get('catalog_sep', '\t'),
            output_directory=config_dict.get('output_directory', Path.cwd() / "multi_sequence_output"),
            clustering=clustering_config,
            template_config=template_config,
            cluster_workflow_template=cluster_workflow_template,
            parallel_processing=config_dict.get('parallel_processing', False),
            max_workers=config_dict.get('max_workers', 4),
            save_individual_results=config_dict.get('save_individual_results', True),
            coordinate_system=config_dict.get('coordinate_system', 'EPSG:21781')
        )
    
    def to_dict(self) -> dict:
        """Convert MultiSequenceConfig to dictionary."""
        result = {
            'project_title': self.project_title,
            'catalog_file': str(self.catalog_file),
            'catalog_sep': self.catalog_sep,
            'output_directory': str(self.output_directory),
            
            # Clustering parameters
            'clustering_method': self.clustering.method,
            'clustering_features': self.clustering.features,
            'dbscan_eps': self.clustering.dbscan_eps,
            'dbscan_min_samples': self.clustering.dbscan_min_samples,
            'dbscan_metric': self.clustering.dbscan_metric,
            'hdbscan_min_cluster_size': self.clustering.hdbscan_min_cluster_size,
            'hdbscan_min_samples': self.clustering.hdbscan_min_samples,
            'temporal_window_days': self.clustering.temporal_window_days,
            'spatial_weight': self.clustering.spatial_weight,
            'min_cluster_size': self.clustering.min_cluster_size,
            
            # Template configuration
            'template_config': self.template_config.to_dict(),
            
            # Multi-sequence options
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers,
            'save_individual_results': self.save_individual_results,
            'coordinate_system': self.coordinate_system
        }
        
        return result
    
    def create_template_config_for_cluster(self, cluster_name: str, cluster_file: Path) -> ProjectConfig:
        """Create a configuration for a specific cluster."""
        cluster_config = ProjectConfig(
            project_title=f"{self.project_title} - {cluster_name}",
            hypo_file=cluster_file,
            hypo_sep=self.catalog_sep,
            out_dir=self.output_directory / cluster_name
        )
        
        # Copy module configurations from template
        cluster_config.fault_network = self.template_config.fault_network
        cluster_config.model_validation = self.template_config.model_validation
        cluster_config.auto_class = self.template_config.auto_class
        cluster_config.stress_analysis = self.template_config.stress_analysis
        
        return cluster_config
