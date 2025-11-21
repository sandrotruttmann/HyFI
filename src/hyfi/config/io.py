#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file I/O utilities for HyFI.

This module handles loading and saving configuration files.
JSON is the only supported format for all configurations.
"""

import json
from pathlib import Path
from typing import Dict, Any, Union, Optional, TYPE_CHECKING
import logging

from .parameters import ProjectConfig
from .schema import HyFIWorkflowDAG
from .validation import ConfigValidationError

if TYPE_CHECKING:
    from .multi_sequence_config import MultiSequenceConfig


logger = logging.getLogger(__name__)


# ==============================================================================
# JSON DAG Configuration (Only Supported Format for DAGs)
# ==============================================================================

def load_dag_from_json(file_path: Union[str, Path]) -> HyFIWorkflowDAG:
    """
    Load DAG configuration from a JSON file.
    
    JSON is the only supported format for DAG configurations.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the JSON DAG configuration file
        
    Returns
    -------
    HyFIWorkflowDAG
        Loaded DAG configuration object
        
    Raises
    ------
    ConfigValidationError
        If file cannot be loaded or parsed
    """
    try:
        return HyFIWorkflowDAG.load_from_file(file_path)
    except FileNotFoundError:
        raise ConfigValidationError(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ConfigValidationError(f"Error parsing JSON file: {e}")
    except Exception as e:
        raise ConfigValidationError(f"Error loading DAG configuration: {e}")


def save_dag_to_json(dag: HyFIWorkflowDAG, file_path: Union[str, Path]) -> None:
    """
    Save DAG configuration to a JSON file.
    
    Parameters
    ----------
    dag : HyFIWorkflowDAG
        DAG configuration object to save
    file_path : Union[str, Path]
        Path where to save the JSON file
        
    Raises
    ------
    ConfigValidationError
        If file cannot be written
    """
    try:
        dag.save_to_file(file_path)
        logger.info(f"DAG configuration saved to: {file_path}")
    except Exception as e:
        raise ConfigValidationError(f"Error saving DAG configuration to JSON: {e}")


def create_template_dag(file_path: Union[str, Path], 
                       workflow_name: str = "My HyFI Analysis",
                       input_file: str = "path/to/hypocenter_data.txt",
                       output_dir: str = "./hyfi_output") -> HyFIWorkflowDAG:
    """
    Create a template DAG configuration and save it to JSON file.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path where to save the template
    workflow_name : str
        Name for the workflow
    input_file : str
        Path to input hypocenter data file
    output_dir : str
        Output directory for results
        
    Returns
    -------
    HyFIWorkflowDAG
        Created template DAG
        
    Raises
    ------
    ConfigValidationError
        If file cannot be written
    """
    # Create a template DAG with reasonable defaults
    dag = HyFIWorkflowDAG(
        workflow_name=workflow_name,
        output_directory=output_dir
    )
    
    # Customize input data node
    dag.nodes["input_data"].hypocenter_file = input_file
    
    # Save to JSON file
    save_dag_to_json(dag, file_path)
    
    print(f"Template DAG configuration saved to: {file_path}")
    print(f"This JSON file represents your analysis workflow as a directed acyclic graph.")
    print(f"Edit the file to customize your analysis parameters and workflow structure.")
    
    return dag


def validate_dag_file(file_path: Union[str, Path]) -> bool:
    """
    Validate a DAG configuration file.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the JSON DAG configuration file
        
    Returns
    -------
    bool
        True if validation passes
        
    Raises
    ------
    ConfigValidationError
        If validation fails
    """
    try:
        dag = load_dag_from_json(file_path)
        dag.validate_dag()
        logger.info(f"DAG configuration is valid: {file_path}")
        return True
    except Exception as e:
        raise ConfigValidationError(f"DAG validation failed: {e}")


# ==============================================================================
# Auto-detection and Conversion (JSON Only)
# ==============================================================================

def auto_load_config(file_path: Union[str, Path]) -> Union[HyFIWorkflowDAG, 'MultiSequenceConfig', ProjectConfig]:
    """
    Automatically detect JSON file format and load configuration.
    
    For JSON files, detects between DAG format, multi-sequence format, and legacy format.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the JSON configuration file
        
    Returns
    -------
    Union[HyFIWorkflowDAG, MultiSequenceConfig, ProjectConfig]
        Loaded configuration object based on detected format
        
    Raises
    ------
    ConfigValidationError
        If file format is not supported or file cannot be loaded
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    # Only handle JSON files
    if suffix == '.json':
        try:
            # Read and analyze file content
            with open(path, 'r') as f:
                content = json.load(f)
            
            # Check for DAG configuration (unified format)
            if 'workflow_dag' in content or 'metadata' in content:
                # Check if it's a multi-sequence DAG by looking for multi_sequence_clustering node
                if 'workflow_dag' in content and 'multi_sequence_clustering' in content['workflow_dag']:
                    from .multi_sequence_config import MultiSequenceConfig
                    return MultiSequenceConfig.from_dict(content)
                else:
                    return load_dag_from_json(path)
            
            # Check for legacy multi-sequence configuration
            elif 'clustering' in content and ('template_config' in content or 'cluster_workflow_template' in content):
                from .multi_sequence_config import MultiSequenceConfig
                return MultiSequenceConfig.from_dict(content)
            
            # Default to legacy configuration
            else:
                return load_config_from_json(path)
                
        except Exception as e:
            raise ConfigValidationError(f"Error loading JSON configuration: {e}")
    else:
        # Try JSON formats only
        try:
            return load_dag_from_json(path)
        except:
            try:
                return load_config_from_json(path)
            except:
                raise ConfigValidationError(f"Only JSON configuration files are supported: {path}")


def create_template_config(file_path: Union[str, Path], format: str = 'json') -> None:
    """
    Create a template configuration file.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path where to save the template
    format : str
        File format - only 'json' is supported
        
    Raises
    ------
    ConfigValidationError
        If format is not supported or file cannot be written
    """
    
    if format.lower() == 'json':
        # Create DAG template (recommended - JSON only for DAGs)
        create_template_dag(
            file_path=file_path,
            workflow_name='My HyFI Analysis',
            input_file='path/to/hypocenter_data.txt',
            output_dir='./hyfi_output'
        )
    else:
        raise ConfigValidationError(f"Unsupported format: {format}. Only 'json' format is supported.")
        
    print(f"Template configuration saved to: {file_path}")
    print(f"Edit this file to customize your analysis parameters.")


def load_config_from_json(file_path: Union[str, Path]) -> ProjectConfig:
    """
    Load configuration from a JSON file (legacy format).
    
    Note: For DAG configurations, use load_dag_from_json() instead.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the JSON configuration file
        
    Returns
    -------
    ProjectConfig
        Loaded configuration object
        
    Raises
    ------
    ConfigValidationError
        If file cannot be loaded or parsed
    """
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return ProjectConfig.from_dict(config_dict)
    except FileNotFoundError:
        raise ConfigValidationError(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ConfigValidationError(f"Error parsing JSON file: {e}")
    except Exception as e:
        raise ConfigValidationError(f"Error loading configuration: {e}")


def save_config_to_json(config: ProjectConfig, file_path: Union[str, Path]) -> None:
    """
    Save configuration to a JSON file (legacy format).
    
    Note: For DAG configurations, use save_dag_to_json() instead.
    
    Parameters
    ----------
    config : ProjectConfig
        Configuration object to save
    file_path : Union[str, Path]
        Path where to save the JSON file
        
    Raises
    ------
    ConfigValidationError
        If file cannot be written
    """
    try:
        config_dict = config.to_dict()
        # Convert Path objects to strings for JSON serialization
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    except Exception as e:
        raise ConfigValidationError(f"Error saving configuration to JSON: {e}")


def convert_legacy_to_dag(legacy_config: ProjectConfig) -> HyFIWorkflowDAG:
    """
    Convert a legacy ProjectConfig to a DAG configuration (JSON format).
    
    Parameters
    ----------
    legacy_config : ProjectConfig
        Legacy configuration to convert
        
    Returns
    -------
    HyFIWorkflowDAG
        Converted DAG configuration
    """
    # Create new DAG
    dag = HyFIWorkflowDAG(
        workflow_name=legacy_config.project_title,
        output_directory=str(legacy_config.out_dir)
    )
    
    # Map legacy parameters to DAG nodes
    dag.nodes["input_data"].hypocenter_file = str(legacy_config.hypo_file)
    dag.nodes["input_data"].hypocenter_separator = legacy_config.hypo_sep
    
    # Fault network parameters
    dag.nodes["fault_network"].parameters.update({
        "monte_carlo_simulations": legacy_config.fault_network.n_mc,
        "search_radius_meters": legacy_config.fault_network.r_nn,
        "search_time_window_hours": legacy_config.fault_network.dt_nn,
        "magnitude_type": legacy_config.fault_network.mag_type
    })
    
    # Model validation parameters
    if legacy_config.model_validation.foc_file:
        dag.nodes["input_data"].focal_mechanism_file = str(legacy_config.model_validation.foc_file)
        dag.nodes["input_data"].focal_mechanism_separator = legacy_config.model_validation.foc_sep
    
    dag.nodes["model_validation"].enabled = legacy_config.model_validation.validation_bool
    dag.nodes["model_validation"].parameters.update({
        "check_magnitude_consistency": legacy_config.model_validation.foc_mag_check,
        "check_location_consistency": legacy_config.model_validation.foc_loc_check
    })
    
    # Auto classification parameters
    dag.nodes["auto_classification"].enabled = legacy_config.auto_class.autoclass_bool
    dag.nodes["auto_classification"].parameters.update({
        "number_of_clusters": legacy_config.auto_class.n_clusters,
        "clustering_algorithm": legacy_config.auto_class.algorithm,
        "rotate_poles_before_analysis": legacy_config.auto_class.rotation
    })
    
    # Stress analysis parameters
    dag.nodes["stress_analysis"].enabled = legacy_config.stress_analysis.stress_bool
    dag.nodes["stress_analysis"].parameters["stress_field"].update({
        "use_shapefile": legacy_config.stress_analysis.use_shapefile_stress,
        "shapefile_path": legacy_config.stress_analysis.stress_field_shapefile,
        "sigma1_trend_degrees": legacy_config.stress_analysis.S1_trend,
        "sigma1_plunge_degrees": legacy_config.stress_analysis.S1_plunge,
        "sigma3_trend_degrees": legacy_config.stress_analysis.S3_trend,
        "sigma3_plunge_degrees": legacy_config.stress_analysis.S3_plunge,
        "stress_shape_ratio": legacy_config.stress_analysis.stress_R
    })
    
    dag.nodes["stress_analysis"].parameters["mechanical_properties"].update({
        "pore_pressure_mpa": legacy_config.stress_analysis.PP,
        "friction_coefficient": legacy_config.stress_analysis.fric_coeff
    })
    
    return dag

