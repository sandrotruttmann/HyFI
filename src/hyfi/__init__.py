#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyFI: Hypocenter-based 3D Imaging of Active Faults

A Python package for 3D imaging of active faults based on relatively relocated 
hypocenter datasets. Based on the methodology described in:

Truttmann, S., Diehl, T., Herwegh, M. (2023). Hypocenter-based 3D Imaging of 
Active Faults: Method and Applications in the Southwestern Swiss Alps. 
Journal of Geophysical Research: Solid Earth. 
https://doi.org/10.1029/2023JB026352
"""

__version__ = "0.1.0"
__author__ = "Sandro Truttmann"
__email__ = "sandro.truttmann@gmail.com"
__all__ = [
    # Core functionality
    "FaultImagingWorkflow",
    
    # Configuration system
    "ProjectConfig", 
    "FaultNetworkConfig",
    "ModelValidationConfig", 
    "AutoClassConfig",
    "StressAnalysisConfig",
    "HyFIWorkflowDAG",
    
    # Configuration utilities
    "load_config_from_json", 
    "load_dag_from_json",
    "save_config_to_json",
    "save_dag_to_json",
    "auto_load_config",
    "create_template_config",
    "create_template_dag",
    "convert_legacy_to_dag",
    
    # Validation
    "validate_file_exists",
    "validate_positive_number",
    "validate_range",
    "ConfigValidationError",
    
    # Individual modules (as modules, not classes)
    "fault_network",
    "model_validation", 
    "auto_class",
    "stress_analysis",
    
    # Query system
    "query",
    "visualisation"
]

# Import core components
try:
    from .core.workflow import FaultImagingWorkflow
    # Import functions from modules (not classes)
    from .core import fault_network
    from .core import model_validation
    from .core import auto_class
    from .core import stress_analysis
    from .visualization import visualisation
    
    # Import configuration system
    from .config.parameters import (
        ProjectConfig, 
        FaultNetworkConfig,
        ModelValidationConfig,
        AutoClassConfig,
        StressAnalysisConfig
    )
    
    from .config.schema import HyFIWorkflowDAG
    
    from .config.io import (
        load_config_from_json,
        load_dag_from_json,
        save_config_to_json,
        save_dag_to_json,
        auto_load_config,
        create_template_config,
        create_template_dag,
        convert_legacy_to_dag
    )
    
    from .config.validation import (
        validate_file_exists,
        validate_positive_number,
        validate_range,
        ConfigValidationError
    )
    
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Some HyFI components could not be imported: {e}")
    
    # Provide minimal interface
    __all__ = ["__version__", "__author__", "__email__"]

# Main imports for easy access
from .core.fault_network import faultnetwork3D
from .core.model_validation import focal_validation
from .core.auto_class import auto_classification
from .core.stress_analysis import fault_stress
from .core.workflow import FaultImagingWorkflow
from .visualization.visualisation import model_3d, faults_stereoplot
from .utils.utilities import save_data

# Module imports for direct access
from .core import fault_network, model_validation, auto_class, stress_analysis
from .visualization import visualisation
from .utils import utilities
from . import query

# Configuration management
from .config.parameters import (
    ProjectConfig, 
    FaultNetworkConfig, 
    ModelValidationConfig,
    AutoClassConfig, 
    StressAnalysisConfig
)
from .config.validation import ConfigValidationError
from .config.io import (
    auto_load_config,
    save_config_to_json,
    create_template_config
)

__all__ = [
    'faultnetwork3D',
    'focal_validation', 
    'auto_classification',
    'fault_stress',
    'FaultImagingWorkflow',
    'model_3d',
    'faults_stereoplot',
    'save_data',
    'ProjectConfig',
    'FaultNetworkConfig',
    'ModelValidationConfig',
    'AutoClassConfig',
    'StressAnalysisConfig',
    'ConfigValidationError',
    'auto_load_config',
    'save_config_to_json',
    'create_template_config',
    # Individual modules
    'fault_network',
    'model_validation',
    'auto_class', 
    'stress_analysis',
    'visualisation',
    'query',
    'utilities'
]
