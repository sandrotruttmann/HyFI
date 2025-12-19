#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAG-based configuration schema for HyFI.

This module defines the structured configuration schema that represents
the analysis workflow as a directed acyclic graph (DAG).
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime

from .validation import ConfigValidationError


@dataclass
class InputDataNode:
    """Input data configuration node."""
    node_type: str = "input_data"
    hypocenter_file: str = ""
    hypocenter_separator: str = "\t"
    focal_mechanism_file: Optional[str] = None
    focal_mechanism_separator: str = ";"
    description: str = "Input earthquake data"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FaultNetworkNode:
    """Fault network reconstruction configuration node."""
    node_type: str = "fault_network"
    depends_on: List[str] = None
    parameters: Dict[str, Any] = None
    description: str = "3D fault network reconstruction using Monte Carlo simulation"
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = ["input_data"]
        if self.parameters is None:
            self.parameters = {
                "monte_carlo_simulations": 1000,
                "search_radius_meters": 100.0,
                "search_time_window_hours": 999999.0,
                "magnitude_type": "ML",
                "auto_optimize_parameters": False,
                "optimization_method": "optuna",
                "optimization_grid_points": 25,
                "optimization_plot_results": False,
                "optimization_r_nn_range": None,  # [min_meters, max_meters] or None for auto
                "optimization_dt_nn_range": None  # [min_hours, max_hours] or None for auto
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelValidationNode:
    """Model validation configuration node."""
    node_type: str = "model_validation"
    depends_on: List[str] = None
    enabled: bool = True
    parameters: Dict[str, Any] = None
    description: str = "Validate fault planes using focal mechanism data"
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = ["fault_network"]
        if self.parameters is None:
            self.parameters = {
                "check_magnitude_consistency": True,
                "check_location_consistency": True,
                "maximum_distance_km": 5.0,
                "maximum_magnitude_difference": 0.5
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AutoClassificationNode:
    """Automatic classification configuration node."""
    node_type: str = "auto_classification"
    depends_on: List[str] = None
    enabled: bool = True
    parameters: Dict[str, Any] = None
    description: str = "Automatic classification of fault structures"
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = ["model_validation"]
        if self.parameters is None:
            self.parameters = {
                "number_of_clusters": 2,
                "clustering_algorithm": "vmf_soft",
                "rotate_poles_before_analysis": True,
                "convergence_tolerance": 1e-6,
                "maximum_iterations": 100
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StressAnalysisNode:
    """Stress analysis configuration node."""
    node_type: str = "stress_analysis"
    depends_on: List[str] = None
    enabled: bool = True
    parameters: Dict[str, Any] = None
    description: str = "Fault stress analysis and failure assessment"
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = ["auto_classification"]
        if self.parameters is None:
            self.parameters = {
                "stress_field": {
                    "use_shapefile": False,  # Whether to use shapefile for stress parameters
                    "shapefile_path": None,  # Optional: path to .shp file with spatially-varying stress field
                    "sigma1_trend_degrees": None,
                    "sigma1_plunge_degrees": None,
                    "sigma3_trend_degrees": None,
                    "sigma3_plunge_degrees": None,
                    "stress_shape_ratio": None
                },
                "mechanical_properties": {
                    "pore_pressure_mpa": 0.0,
                    "friction_coefficient": 0.75
                }
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualizationNode:
    """Visualization configuration node."""
    node_type: str = "visualization"
    depends_on: List[str] = None
    enabled: bool = True
    parameters: Dict[str, Any] = None
    description: str = "3D visualization and result plotting"
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = ["stress_analysis"]
        if self.parameters is None:
            self.parameters = {
                "generate_3d_model": True,
                "generate_stereonet": True,
                "generate_summary_plots": True,
                "output_formats": ["html", "png"],
                "interactive_plots": True,
                # Interpolation parameters
                "enable_plane_interpolation": True,
                "poisson_depth": 2,
                "density_threshold": 0.01,
                "max_distance_factor": 2.5,
                "spatial_clustering_method": "adaptive",  # "kmeans", "dbscan", "adaptive"
                "n_spatial_clusters": 2,
                "min_events_per_cluster": 10,
                "export_interpolated_vtp": True,
                # Mesh stress calculation
                "enable_mesh_stress": True,
                "mesh_subdivisions": 2
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HyFIWorkflowDAG:
    """
    Complete HyFI workflow represented as a DAG.
    
    This class represents the entire analysis workflow as a directed
    acyclic graph where each node represents a processing step.
    """
    
    # Metadata
    workflow_name: str = "HyFI Analysis"
    workflow_version: str = "1.0.0"
    created_date: str = ""
    description: str = "Hypocenter-based 3D fault imaging workflow"
    
    # Global settings
    output_directory: str = "./hyfi_output"
    random_seed: Optional[int] = None
    parallel_processing: bool = True
    log_level: str = "INFO"
    
    # Workflow nodes (DAG structure)
    nodes: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.created_date:
            self.created_date = datetime.now().isoformat()
        
        if self.nodes is None:
            self.nodes = {
                "input_data": InputDataNode(),
                "fault_network": FaultNetworkNode(),
                "model_validation": ModelValidationNode(),
                "auto_classification": AutoClassificationNode(),
                "stress_analysis": StressAnalysisNode(),
                "visualization": VisualizationNode()
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DAG to dictionary representation."""
        result = {
            "metadata": {
                "workflow_name": self.workflow_name,
                "workflow_version": self.workflow_version,
                "created_date": self.created_date,
                "description": self.description
            },
            "global_settings": {
                "output_directory": self.output_directory,
                "random_seed": self.random_seed,
                "parallel_processing": self.parallel_processing,
                "log_level": self.log_level
            },
            "workflow_dag": {}
        }
        
        # Convert all nodes to dictionaries
        for node_id, node in self.nodes.items():
            if hasattr(node, 'to_dict'):
                result["workflow_dag"][node_id] = node.to_dict()
            else:
                result["workflow_dag"][node_id] = asdict(node)
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert DAG to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save DAG configuration to JSON file."""
        path = Path(file_path)
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HyFIWorkflowDAG':
        """Create DAG from dictionary representation."""
        
        # Extract metadata
        metadata = config_dict.get("metadata", {})
        global_settings = config_dict.get("global_settings", {})
        workflow_dag = config_dict.get("workflow_dag", {})
        
        # Create DAG instance
        dag = cls(
            workflow_name=metadata.get("workflow_name", "HyFI Analysis"),
            workflow_version=metadata.get("workflow_version", "1.0.0"),
            created_date=metadata.get("created_date", ""),
            description=metadata.get("description", "Hypocenter-based 3D fault imaging workflow"),
            output_directory=global_settings.get("output_directory", "./hyfi_output"),
            random_seed=global_settings.get("random_seed"),
            parallel_processing=global_settings.get("parallel_processing", True),
            log_level=global_settings.get("log_level", "INFO")
        )
        
        # Reconstruct nodes from dictionary
        node_classes = {
            "input_data": InputDataNode,
            "fault_network": FaultNetworkNode,
            "model_validation": ModelValidationNode,
            "auto_classification": AutoClassificationNode,
            "stress_analysis": StressAnalysisNode,
            "visualization": VisualizationNode
        }
        
        dag.nodes = {}
        for node_id, node_data in workflow_dag.items():
            node_type = node_data.get("node_type", node_id)
            if node_type in node_classes:
                # Create node instance from data
                node_class = node_classes[node_type]
                dag.nodes[node_id] = node_class(**{
                    k: v for k, v in node_data.items() 
                    if k in node_class.__dataclass_fields__
                })
            else:
                # Keep as generic dictionary for unknown node types
                dag.nodes[node_id] = node_data
        
        return dag
    
    @classmethod
    def from_json(cls, json_str: str) -> 'HyFIWorkflowDAG':
        """Create DAG from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'HyFIWorkflowDAG':
        """Load DAG configuration from JSON file."""
        path = Path(file_path)
        with open(path, 'r') as f:
            return cls.from_json(f.read())
    
    def validate_dag(self) -> None:
        """Validate the DAG structure and dependencies."""
        
        # Check for circular dependencies
        def has_cycle(node_id: str, visited: set, rec_stack: set) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            node = self.nodes.get(node_id)
            if node and hasattr(node, 'depends_on') and node.depends_on:
                for dep in node.depends_on:
                    if dep not in visited:
                        if has_cycle(dep, visited, rec_stack):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        visited = set()
        rec_stack = set()
        
        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle(node_id, visited, rec_stack):
                    raise ConfigValidationError(f"Circular dependency detected in DAG involving node: {node_id}")
        
        # Check that all dependencies exist
        for node_id, node in self.nodes.items():
            if hasattr(node, 'depends_on') and node.depends_on:
                for dep in node.depends_on:
                    if dep not in self.nodes:
                        raise ConfigValidationError(f"Node '{node_id}' depends on non-existent node '{dep}'")
        
        # Validate individual node configurations
        for node_id, node in self.nodes.items():
            if hasattr(node, 'validate'):
                try:
                    node.validate()
                except Exception as e:
                    raise ConfigValidationError(f"Validation failed for node '{node_id}': {e}")
    
    def get_execution_order(self) -> List[str]:
        """Get the execution order of nodes based on dependencies."""
        
        def topological_sort():
            in_degree = {node_id: 0 for node_id in self.nodes}
            
            # Calculate in-degrees
            for node_id, node in self.nodes.items():
                if hasattr(node, 'depends_on') and node.depends_on:
                    for dep in node.depends_on:
                        if dep in in_degree:
                            in_degree[node_id] += 1
            
            # Initialize queue with nodes having no dependencies
            queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
            result = []
            
            while queue:
                current = queue.pop(0)
                result.append(current)
                
                # Reduce in-degree for dependent nodes
                for node_id, node in self.nodes.items():
                    if hasattr(node, 'depends_on') and node.depends_on and current in node.depends_on:
                        in_degree[node_id] -= 1
                        if in_degree[node_id] == 0:
                            queue.append(node_id)
            
            if len(result) != len(self.nodes):
                raise ConfigValidationError("DAG contains cycles - cannot determine execution order")
            
            return result
        
        return topological_sort()
    
    def get_enabled_nodes(self) -> List[str]:
        """Get list of enabled nodes in execution order."""
        execution_order = self.get_execution_order()
        return [
            node_id for node_id in execution_order
            if not hasattr(self.nodes[node_id], 'enabled') or self.nodes[node_id].enabled
        ]
