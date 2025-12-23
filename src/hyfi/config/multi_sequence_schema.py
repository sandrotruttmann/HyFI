#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-sequence DAG node definitions for HyFI.

This module extends the DAG system to support multi-sequence workflows
with catalog clustering and parallel processing.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from ..config.schema import *


@dataclass
class CatalogClusteringNode:
    """Catalog clustering/segmentation configuration node."""
    node_type: str = "catalog_clustering"
    depends_on: List[str] = None
    parameters: Dict[str, Any] = None
    description: str = "Segment earthquake catalog into clusters using machine learning"
    enabled: bool = True
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = ["input_data"]
        if self.parameters is None:
            self.parameters = {
                "clustering_method": "dbscan",
                "clustering_features": ["spatial"],
                "cluster_dimension": "3d",
                "dbscan_eps": 1000.0,
                "dbscan_min_samples": 10,
                "dbscan_metric": "euclidean",
                "hdbscan_min_cluster_size": 15,
                "temporal_window_days": 30,
                "spatial_weight": 0.7,
                "min_cluster_size": 20
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterProcessingNode:
    """Cluster processing configuration node."""
    node_type: str = "cluster_processing"
    depends_on: List[str] = None
    parameters: Dict[str, Any] = None
    description: str = "Process each cluster through the standard HyFI workflow"
    enabled: bool = True
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = ["catalog_clustering"]
        if self.parameters is None:
            self.parameters = {
                "parallel_processing": False,
                "max_workers": 4,
                "save_individual_results": True,
                "template_workflow": {
                    "fault_network": {"enabled": True},
                    "model_validation": {"enabled": True},
                    "auto_classification": {"enabled": True},
                    "stress_analysis": {"enabled": True},
                    "visualization": {"enabled": True}
                }
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResultsAggregationNode:
    """Results aggregation configuration node."""
    node_type: str = "results_aggregation"
    depends_on: List[str] = None
    parameters: Dict[str, Any] = None
    description: str = "Aggregate and combine results from all clusters"
    enabled: bool = True
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = ["cluster_processing"]
        if self.parameters is None:
            self.parameters = {
                "statistical_analysis": True,
                "export_formats": ["csv", "vtp", "json"]
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiSequenceWorkflowDAG:
    """
    Multi-sequence workflow configuration as a directed acyclic graph.
    
    This extends the basic HyFI DAG to support multi-sequence analysis
    with catalog clustering and parallel processing capabilities.
    """
    
    # Workflow metadata
    workflow_name: str = "Multi-Sequence HyFI Analysis"
    workflow_type: str = "multi_sequence"
    description: str = "Multi-sequence hypocenter-based 3D fault imaging with catalog clustering"
    version: str = "1.0"
    created_date: str = ""
    
    # Global settings
    output_directory: str = "./multi_sequence_output"
    log_level: str = "INFO"
    
    # DAG nodes
    nodes: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.created_date:
            from datetime import datetime
            self.created_date = datetime.now().isoformat()
        
        if self.nodes is None:
            # Create default multi-sequence workflow
            self.nodes = {
                "input_data": InputDataNode(
                    hypocenter_file="path/to/catalog.txt",
                    description="Full earthquake catalog input"
                ),
                "catalog_clustering": CatalogClusteringNode(),
                "cluster_processing": ClusterProcessingNode(),
                "results_aggregation": ResultsAggregationNode(),
                "visualization": VisualizationNode(
                    depends_on=["results_aggregation"],
                    description="Multi-sequence visualization and reporting"
                )
            }
    
    def get_execution_order(self) -> List[str]:
        """
        Get the execution order of nodes based on dependencies.
        
        Returns
        -------
        List[str]
            Ordered list of node IDs for execution
        """
        # Simple topological sort for the multi-sequence DAG
        # For now, we use a fixed order, but this could be made more dynamic
        return [
            "input_data",
            "catalog_clustering", 
            "cluster_processing",
            "results_aggregation",
            "visualization"
        ]
    
    def get_enabled_nodes(self) -> List[str]:
        """
        Get list of enabled nodes in execution order.
        
        Returns
        -------
        List[str]
            Ordered list of enabled node IDs
        """
        all_nodes = self.get_execution_order()
        enabled_nodes = []
        
        for node_id in all_nodes:
            node = self.nodes.get(node_id)
            if node and getattr(node, 'enabled', True):
                enabled_nodes.append(node_id)
        
        return enabled_nodes
    
    def validate_dag(self):
        """Validate the DAG structure and configuration."""
        if not self.nodes:
            raise ConfigValidationError("DAG must have at least one node")
        
        # Check required nodes for multi-sequence workflow
        required_nodes = ["input_data", "catalog_clustering", "cluster_processing"]
        for node_id in required_nodes:
            if node_id not in self.nodes:
                raise ConfigValidationError(f"Required node '{node_id}' is missing from multi-sequence DAG")
        
        # Validate node dependencies
        for node_id, node in self.nodes.items():
            if hasattr(node, 'depends_on') and node.depends_on:
                for dep in node.depends_on:
                    if dep not in self.nodes:
                        raise ConfigValidationError(f"Node '{node_id}' depends on non-existent node '{dep}'")
        
        # Validate input data node
        input_node = self.nodes.get("input_data")
        if input_node and hasattr(input_node, 'hypocenter_file'):
            if not input_node.hypocenter_file or input_node.hypocenter_file == "path/to/catalog.txt":
                raise ConfigValidationError("Input data node must specify a valid hypocenter_file")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DAG to dictionary representation."""
        result = {
            "workflow_name": self.workflow_name,
            "workflow_type": self.workflow_type,
            "description": self.description,
            "version": self.version,
            "created_date": self.created_date,
            "output_directory": self.output_directory,
            "log_level": self.log_level,
            "nodes": {}
        }
        
        for node_id, node in self.nodes.items():
            if hasattr(node, 'to_dict'):
                result["nodes"][node_id] = node.to_dict()
            else:
                result["nodes"][node_id] = asdict(node)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultiSequenceWorkflowDAG':
        """Create DAG from dictionary representation."""
        # Reconstruct nodes from dictionary
        nodes = {}
        
        for node_id, node_data in data.get("nodes", {}).items():
            node_type = node_data.get("node_type", node_id)
            
            if node_type == "input_data":
                nodes[node_id] = InputDataNode(**{k: v for k, v in node_data.items() if k != "node_type"})
            elif node_type == "catalog_clustering":
                nodes[node_id] = CatalogClusteringNode(**{k: v for k, v in node_data.items() if k != "node_type"})
            elif node_type == "cluster_processing":
                nodes[node_id] = ClusterProcessingNode(**{k: v for k, v in node_data.items() if k != "node_type"})
            elif node_type == "results_aggregation":
                nodes[node_id] = ResultsAggregationNode(**{k: v for k, v in node_data.items() if k != "node_type"})
            elif node_type == "visualization":
                nodes[node_id] = VisualizationNode(**{k: v for k, v in node_data.items() if k != "node_type"})
            else:
                # Generic node handling
                nodes[node_id] = node_data
        
        return cls(
            workflow_name=data.get("workflow_name", "Multi-Sequence HyFI Analysis"),
            workflow_type=data.get("workflow_type", "multi_sequence"),
            description=data.get("description", "Multi-sequence hypocenter-based 3D fault imaging"),
            version=data.get("version", "1.0"),
            created_date=data.get("created_date", ""),
            output_directory=data.get("output_directory", "./multi_sequence_output"),
            log_level=data.get("log_level", "INFO"),
            nodes=nodes
        )
