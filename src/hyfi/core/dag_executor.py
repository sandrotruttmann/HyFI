#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAG-based workflow executor for HyFI.

This module executes analysis workflows defined as directed acyclic graphs.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
import numpy as np

from ..config.schema import HyFIWorkflowDAG
from ..config.validation import ConfigValidationError
from .fault_network import faultnetwork3D
from .model_validation import focal_validation
from .auto_class import auto_classification
from .stress_analysis import fault_stress
from ..visualization import visualisation
from ..utils.utilities import setup_output_directory, dag_params_to_legacy_params


logger = logging.getLogger(__name__)


class DAGExecutor:
    """
    Execute HyFI workflows defined as directed acyclic graphs.
    
    This class handles the execution of complex analysis workflows by processing
    nodes in the correct order based on their dependencies.
    """
    
    def __init__(self, dag_config: HyFIWorkflowDAG, config_source_file: Optional[str] = None, cluster_name: Optional[str] = None, 
                 global_fault_counter: int = 1, sequence_label: Optional[str] = None, segmentation_level: Optional[str] = None):
        """
        Initialize the DAG executor.
        
        Parameters
        ----------
        dag_config : HyFIWorkflowDAG
            DAG configuration object
        config_source_file : str, optional
            Path to the original configuration file (for copying to output)
        cluster_name : str, optional
            Name of the cluster being processed (used for special handling)
        global_fault_counter : int, optional
            Starting value for global fault system counter (default: 1)
        sequence_label : str, optional
            Label of the sequence being processed (e.g., 'A1', 'B2')
        segmentation_level : str, optional
            Segmentation level letter (e.g., 'A', 'B', 'C')
        """
        self.dag = dag_config
        self.config_source_file = config_source_file
        self.cluster_name = cluster_name
        self.global_fault_counter = global_fault_counter
        self.sequence_label = sequence_label
        self.segmentation_level = segmentation_level
        self.results = {}
        self.execution_log = []
        self.start_time = None
        self.end_time = None
        self.fault_system_metadata = []
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, dag_config.log_level.upper()))
        
        # Validate DAG before execution
        self.dag.validate_dag()
        
        # Setup output directory
        self.output_dir = Path(dag_config.output_directory)
        setup_output_directory(self.output_dir)
    
    def _flatten_visualization_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested visualization parameters to support both flat and nested structures.
        
        Parameters
        ----------
        params : dict
            Visualization parameters (may be nested or flat)
            
        Returns
        -------
        dict
            Flattened parameters with all values at top level
        """
        flattened = {}
        
        # Check if parameters are nested (new structure)
        nested_groups = ['basic_visualization', 'fault_surface_interpolation', 
                        'paraview_vtk_export', 'time_series_animation', 'geographic_export']
        
        is_nested = any(group in params for group in nested_groups)
        
        if is_nested:
            # Flatten nested structure
            for group_name, group_params in params.items():
                if isinstance(group_params, dict):
                    flattened.update(group_params)
                else:
                    flattened[group_name] = group_params
        else:
            # Already flat, just return as is
            flattened = params.copy()
        
        return flattened
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the complete DAG workflow.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing results from all executed nodes
        """
        self.start_time = datetime.now()
        
        try:
            # Get execution order
            execution_order = self.dag.get_enabled_nodes()
            
            # Execute nodes in order
            for node_id in execution_order:
                self._execute_node(node_id)
            
            # Automatically save all results at workflow completion
            self._save_workflow_results()
            
            self.end_time = datetime.now()
            duration = self.end_time - self.start_time
            
            # Execution summary is now saved in _save_workflow_results()
            # self._save_execution_summary()
            
            # Include fault system metadata in results
            results_with_metadata = self.results.copy()
            results_with_metadata['fault_system_metadata'] = self.fault_system_metadata
            results_with_metadata['next_fault_counter'] = self.global_fault_counter
            
            return results_with_metadata
            
        except Exception as e:
            self.end_time = datetime.now()
            logger.error(f"DAG execution failed: {e}")
            raise
    
    def _should_skip_module_for_outliers(self, node_id: str) -> bool:
        """
        Check if a module should be skipped for outlier clusters.
        
        Parameters
        ----------
        node_id : str
            ID of the node to check
            
        Returns
        -------
        bool
            True if the module should be skipped for outlier clusters
        """
        # Skip fault analysis modules for Z_outliers
        if self.cluster_name == 'Z_outliers':
            skip_modules = ['fault_network', 'model_validation', 'auto_classification', 'stress_analysis']
            return node_id in skip_modules
        return False
    
    def _execute_node(self, node_id: str) -> Any:
        """
        Execute a single node in the DAG.
        
        Parameters
        ----------
        node_id : str
            ID of the node to execute
            
        Returns
        -------
        Any
            Result from node execution
        """
        node = self.dag.nodes[node_id]
        node_start_time = time.time()
        
        try:
            # Check if node is enabled
            if hasattr(node, 'enabled') and not node.enabled:
                self.results[node_id] = None
                return None
            
            # Check if module should be skipped for outlier clusters
            if self._should_skip_module_for_outliers(node_id):
                logger.info(f"Skipping {node_id} for outlier cluster {self.cluster_name}")
                # For input_data, still load the data but mark as outliers
                if node_id == "input_data":
                    result = self._execute_input_data(node)
                    # Mark data as outliers for downstream processing
                    if 'hypocenter_data' in result:
                        result['hypocenter_data']['is_outlier_cluster'] = True
                    self.results[node_id] = result
                    return result
                else:
                    # For other modules, return None but create placeholder result
                    self.results[node_id] = {'skipped_for_outliers': True}
                    return None
            
            # Execute based on node type
            if node_id == "input_data":
                result = self._execute_input_data(node)
            elif node_id == "fault_network":
                result = self._execute_fault_network(node)
            elif node_id == "model_validation":
                result = self._execute_model_validation(node)
            elif node_id == "auto_classification":
                result = self._execute_auto_classification(node)
            elif node_id == "stress_analysis":
                result = self._execute_stress_analysis(node)
            elif node_id == "visualization":
                result = self._execute_visualization(node)
            else:
                raise ConfigValidationError(f"Unknown node type: {node_id}")
            
            # Store result
            self.results[node_id] = result
            
            # Log execution
            node_duration = time.time() - node_start_time
            log_entry = {
                'node_id': node_id,
                'start_time': node_start_time,
                'duration': node_duration,
                'success': True,
                'description': node.description
            }
            self.execution_log.append(log_entry)
            
            return result
            
        except Exception as e:
            node_duration = time.time() - node_start_time
            log_entry = {
                'node_id': node_id,
                'start_time': node_start_time,
                'duration': node_duration,
                'success': False,
                'error': str(e),
                'description': node.description
            }
            self.execution_log.append(log_entry)
            
            logger.error(f"Node {node_id} failed after {node_duration:.2f}s: {e}")
            raise
    
    def _execute_input_data(self, node) -> Dict[str, Any]:
        """Execute input data loading node."""
        import pandas as pd
        from ..utils.input_validation import InputFileValidator
        
        # Initialize validator
        validator = InputFileValidator()
        
        # Validate hypocenter file
        hypo_validation = validator.validate_hypocenter_file(
            node.hypocenter_file, 
            node.hypocenter_separator
        )
        
        if not hypo_validation['valid']:
            error_msg = f"Hypocenter file validation failed: {hypo_validation.get('error', 'Unknown error')}"
            if hypo_validation.get('missing_columns'):
                error_msg += f"\nMissing columns: {', '.join(hypo_validation['missing_columns'])}"
            if hypo_validation.get('recommendations'):
                error_msg += f"\nRecommendations: {'; '.join(hypo_validation['recommendations'])}"
            raise ValueError(error_msg)
                
        # Load hypocenter data
        hypo_file = Path(node.hypocenter_file)
        if not hypo_file.exists():
            raise FileNotFoundError(f"Hypocenter file not found: {hypo_file}")
        
        # Only load the first 24 columns (standard hypoDD format)
        hypo_data = pd.read_csv(hypo_file, sep=node.hypocenter_separator, usecols=range(24))
        
        result = {
            'hypocenter_data': hypo_data,
            'hypocenter_file': str(hypo_file),
            'validation_results': {
                'hypocenter': hypo_validation
            }
        }
        
        # Load focal mechanism data if specified
        if node.focal_mechanism_file:
            # Validate focal mechanism file
            focal_validation = validator.validate_focal_mechanism_file(
                node.focal_mechanism_file,
                node.focal_mechanism_separator
            )
            
            if not focal_validation['valid']:
                error_msg = f"Focal mechanism file validation failed: {focal_validation.get('error', 'Unknown error')}"
                if focal_validation.get('missing_columns'):
                    error_msg += f"\nMissing columns: {', '.join(focal_validation['missing_columns'])}"
                if focal_validation.get('recommendations'):
                    error_msg += f"\nRecommendations: {'; '.join(focal_validation['recommendations'])}"
                raise ValueError(error_msg)
            
            
            foc_file = Path(node.focal_mechanism_file)
            if foc_file.exists():
                # Only load the first 28 columns (standard focal mechanism format)
                foc_data = pd.read_csv(foc_file, sep=node.focal_mechanism_separator, usecols=range(28))
                result['focal_mechanism_data'] = foc_data
                result['focal_mechanism_file'] = str(foc_file)
                result['focal_mechanism_separator'] = node.focal_mechanism_separator
                result['validation_results']['focal_mechanism'] = focal_validation
            else:
                logger.warning(f"Focal mechanism file not found: {foc_file}")
        
        return result
    
    def _execute_fault_network(self, node) -> Dict[str, Any]:
        """Execute fault network reconstruction node."""
        from ..utils.utilities import fault_network_with_optimization
        
        # Convert DAG parameters to legacy format
        input_params = dag_params_to_legacy_params(self.dag, 'fault_network')
        
        # Add optimization parameters from node
        if hasattr(node, 'parameters'):
            params = node.parameters
            input_params.update({
                'auto_optimize_parameters': params.get('auto_optimize_parameters', False),
                'optimization_method': params.get('optimization_method', 'grid_search'),
                'optimization_random_state': params.get('optimization_random_state', 42),
                'optimization_plot_results': params.get('optimization_plot_results', False),
                'optimization_r_nn_range': params.get('optimization_r_nn_range', None),
                'optimization_dt_nn_range': params.get('optimization_dt_nn_range', None),
                # Grid search parameters
                'optimization_grid_points': params.get('optimization_grid_points', 25),
                # Optuna optimization parameters
                'optimization_n_trials': params.get('optimization_n_trials', 50),
                'optimization_sampler': params.get('optimization_sampler', 'tpe'),
                'optimization_n_startup_trials': params.get('optimization_n_startup_trials', 10),
                'optimization_early_stopping_rounds': params.get('optimization_early_stopping_rounds', None),
                'optimization_early_stopping_threshold': params.get('optimization_early_stopping_threshold', 1e-4),
                # Pareto optimization parameters
                'optimization_pareto_sampler': params.get('optimization_pareto_sampler', 'nsga2'),
                'optimization_pareto_population': params.get('optimization_pareto_population', 50)
            })
        
        # Add output directory for saving optimization plots
        input_params['out_dir'] = str(self.output_dir)
        
        # Run fault network reconstruction with optimization support
        result = fault_network_with_optimization(input_params)
        
        # The function now returns 4 outputs: (df_hyfi, df_per_X, df_per_Y, df_per_Z)
        if isinstance(result, tuple) and len(result) == 4:
            df_hyfi, df_per_X, df_per_Y, df_per_Z = result
            
            return {
                'df_hyfi': df_hyfi,  # Single enriched dataframe
                'df_per_X': df_per_X,
                'df_per_Y': df_per_Y,
                'df_per_Z': df_per_Z,
                'parameters': node.parameters,
                'legacy_params': input_params
            }
        else:
            logger.error("Unexpected return format from fault_network_with_optimization")
            return {
                'result': result,
                'parameters': node.parameters,
                'legacy_params': input_params
            }
    
    def _execute_model_validation(self, node) -> Dict[str, Any]:
        """Execute model validation node."""
        # Get dependencies
        fault_data = self.results.get('fault_network', {})
        
        # Check if fault_data is valid
        if not isinstance(fault_data, dict):
            logger.warning("Fault network results are not in expected format")
            return {'validation_results': None}
        
        # Check if node is enabled
        if not node.enabled:
            return {'validation_skipped': True, 'df_hyfi': fault_data.get('df_hyfi')}
        
        # Check if we have the required data from fault network
        if 'df_hyfi' not in fault_data:
            logger.warning("No fault network data available for validation")
            return {'validation_results': None}
        
        # Build model validation specific parameters
        base_params = fault_data.get('legacy_params', {})
        
        # Get focal mechanism parameters from input_data node
        input_data_results = self.results.get('input_data', {})
        
        # Add validation-specific parameters
        validation_params = base_params.copy()
        validation_params.update({
            'validation_bool': True,
            'foc_mag_check': node.parameters.get('check_magnitude_consistency', True),
            'foc_loc_check': node.parameters.get('check_location_consistency', True),
            'foc_max_mag_diff': node.parameters.get('maximum_magnitude_difference', 0.5),
            'foc_max_dist_km': node.parameters.get('maximum_distance_km', 5.0),
        })
        
        # Add focal mechanism file parameters if available
        if 'focal_mechanism_file' in input_data_results:
            validation_params.update({
                'foc_file': input_data_results['focal_mechanism_file'],
                'foc_sep': input_data_results.get('focal_mechanism_separator', ';'),
            })
        else:
            logger.warning("No focal mechanism file specified in input data")
        
        # Get required data
        df_hyfi = fault_data['df_hyfi']
        
        # Perform focal mechanism validation
        try:
            result_df_hyfi = focal_validation(df_hyfi, validation_params)
        except Exception as e:
            logger.error(f"Focal mechanism validation failed: {e}")
            # Return original dataframe with validation columns as NaN
            result_df_hyfi = df_hyfi.copy()
            result_df_hyfi['epsilon'] = np.nan
            result_df_hyfi['pref_foc'] = np.nan
        
        return {
            'validation_results': result_df_hyfi,
            'df_hyfi': result_df_hyfi,
            'parameters': node.parameters,
            'legacy_params': validation_params
        }
    
    def _execute_auto_classification(self, node) -> Dict[str, Any]:
        """Execute automatic classification node."""
        # Get dependencies - use model validation data if available, otherwise fault network data
        if 'model_validation' in self.results and self.results['model_validation']:
            prev_data = self.results['model_validation']
            # Defensive check - ensure prev_data is a dictionary
            if isinstance(prev_data, dict):
                df_hyfi = prev_data.get('df_hyfi')
            else:
                logger.warning("Model validation results are not in expected format, falling back to fault network data")
                prev_data = self.results.get('fault_network', {})
                df_hyfi = prev_data.get('df_hyfi') if isinstance(prev_data, dict) else None
        else:
            prev_data = self.results.get('fault_network', {}) 
            df_hyfi = prev_data.get('df_hyfi') if isinstance(prev_data, dict) else None
        
        # Check if node is enabled
        if not node.enabled:
            logger.info("Auto classification is disabled")
            return {'classification_skipped': True, 'df_hyfi': df_hyfi}
        
        if df_hyfi is None:
            logger.warning("No data available for classification")
            return {'classification_results': None}
        
        # Get legacy parameters
        fault_network_results = self.results.get('fault_network', {})
        input_params = fault_network_results.get('legacy_params', {}) if isinstance(fault_network_results, dict) else {}
        
        # Run auto classification using the updated single dataframe function
        result_df_hyfi, fault_metadata, next_counter = auto_classification(
            input_params, 
            df_hyfi,
            starting_fault_counter=self.global_fault_counter,
            sequence_label=self.sequence_label,
            segmentation_level=self.segmentation_level
        )
        
        # Store fault system metadata and update counter
        self.fault_system_metadata.extend(fault_metadata)
        self.global_fault_counter = next_counter
                
        return {
            'classification_results': result_df_hyfi,
            'df_hyfi': result_df_hyfi,
            'parameters': node.parameters,
            'fault_system_metadata': fault_metadata,
            'next_fault_counter': next_counter
        }
    
    def _execute_stress_analysis(self, node) -> Dict[str, Any]:
        """Execute stress analysis node."""
        # Get dependencies - use classification data if available, otherwise previous data
        if 'auto_classification' in self.results and self.results['auto_classification']:
            prev_data = self.results['auto_classification']
            if 'df_hyfi' in prev_data:
                df_hyfi = prev_data['df_hyfi']
            else:
                logger.warning("No df_hyfi in auto_classification results")
                return {'stress_analysis_results': None}
        elif 'model_validation' in self.results and self.results['model_validation']:
            prev_data = self.results['model_validation']
            if 'df_hyfi' in prev_data:
                df_hyfi = prev_data['df_hyfi']
            else:
                logger.warning("No df_hyfi in model_validation results")
                return {'stress_analysis_results': None}
        else:
            # Fallback to fault_network data
            prev_data = self.results.get('fault_network', {})
            if 'df_hyfi' in prev_data:
                df_hyfi = prev_data['df_hyfi']
            else:
                logger.warning("No df_hyfi available for stress analysis")
                return {'stress_analysis_results': None}
        
        # Check if node is enabled
        if not node.enabled:
            logger.info("Stress analysis is disabled")
            return {'stress_analysis_skipped': True, 'df_hyfi': df_hyfi}
        
        # Get stress analysis parameters
        stress_params = {
            'stress_bool': True,
            'S1_trend': node.parameters.get('stress_field', {}).get('sigma1_trend_degrees', 0),
            'S1_plunge': node.parameters.get('stress_field', {}).get('sigma1_plunge_degrees', 0),
            'S3_trend': node.parameters.get('stress_field', {}).get('sigma3_trend_degrees', 90),
            'S3_plunge': node.parameters.get('stress_field', {}).get('sigma3_plunge_degrees', 0),
            'stress_R': node.parameters.get('stress_field', {}).get('stress_shape_ratio', 0.5),
            'PP': node.parameters.get('mechanical_properties', {}).get('pore_pressure_mpa', 0.0),
            'fric_coeff': node.parameters.get('mechanical_properties', {}).get('friction_coefficient', 0.6),
            'use_shapefile_stress': node.parameters.get('stress_field', {}).get('use_shapefile', False),
            'stress_field_shapefile': node.parameters.get('stress_field', {}).get('shapefile_path', None)
        }
        
        try:
            # Run stress analysis using the single dataframe function
            result_df_hyfi, S2_trend, S2_plunge = fault_stress(df_hyfi, stress_params)
            
            return {
                'stress_analysis_results': (result_df_hyfi, S2_trend, S2_plunge),
                'df_hyfi': result_df_hyfi,
                'parameters': node.parameters,
                'legacy_params': stress_params
            }
        except Exception as e:
            logger.error(f"Stress analysis failed: {e}")
            # Return original dataframe with stress columns as NaN
            stress_columns = ['Sn_eff', 'Tau', 'rake', 'instab', 'sliptend', 'dilatend']
            for col in stress_columns:
                df_hyfi[col] = np.nan
            return {
                'stress_analysis_results': None,
                'df_hyfi': df_hyfi,
                'parameters': node.parameters
            }
    
    def _execute_visualization(self, node) -> Dict[str, Any]:
        """Execute visualization node."""
        # Check if node is enabled
        if not node.enabled:
            logger.info("Visualization is disabled")
            return {'visualization_skipped': True}
        
        # For outlier clusters, create simplified visualization with VTK export only
        if self.cluster_name == 'Z_outliers':
            return self._execute_outlier_visualization(node)
        
        # Get dependencies - use stress analysis data if available, otherwise previous data
        if 'stress_analysis' in self.results and self.results['stress_analysis']:
            prev_data = self.results['stress_analysis']
            if isinstance(prev_data, dict):
                df_hyfi = prev_data.get('df_hyfi')
                if df_hyfi is None:
                    logger.info("No df_hyfi in stress_analysis, falling back to auto_classification")
                    prev_data = self.results.get('auto_classification', {})
                    df_hyfi = prev_data.get('df_hyfi') if isinstance(prev_data, dict) else None
            else:
                # Fallback to earlier data
                prev_data = self.results.get('auto_classification', 
                            self.results.get('model_validation', 
                            self.results.get('fault_network', {})))
                df_hyfi = prev_data.get('df_hyfi') if isinstance(prev_data, dict) else None
        else:
            # Fallback to earlier data
            prev_data = self.results.get('auto_classification', 
                        self.results.get('model_validation', 
                        self.results.get('fault_network', {})))
            df_hyfi = prev_data.get('df_hyfi') if isinstance(prev_data, dict) else None

        if df_hyfi is None:
            logger.warning("No data available for visualization")
            return {'visualization_results': None}
        
        # Get all required data for visualization
        fault_network_results = self.results.get('fault_network', {})
        input_params = fault_network_results.get('legacy_params', {}) if isinstance(fault_network_results, dict) else {}
        
        # Run visualization using the updated single dataframe functions
        try:
            # Validate data type before passing to visualization
            if not hasattr(df_hyfi, 'to_csv'):
                logger.warning(f"df_hyfi is not a DataFrame: {type(df_hyfi)}")
                raise ValueError("df_hyfi is not a DataFrame")

            # Flatten visualization parameters to support both flat and nested structures
            viz_params_raw = node.parameters or {}
            viz_params = self._flatten_visualization_params(viz_params_raw)
            
            # Check if interpolation is enabled
            if viz_params.get('enable_plane_interpolation', False):
                
                try:
                    # Create interpolated fault planes using single dataframe
                    combined_mesh, individual_meshes, point_cloud, fault_disc_meshes, interpolation_metadata = visualisation.create_interpolated_fault_planes(
                        df_hyfi, viz_params
                    )
                    
                    # Store interpolation metadata (only for successfully interpolated fault systems)
                    if interpolation_metadata:
                        self.fault_system_metadata.extend(interpolation_metadata)
                    
                    # Calculate stress on mesh faces if enabled and stress data available
                    if viz_params.get('enable_mesh_stress', False) and combined_mesh is not None:
                        # Get stress parameters from stress_analysis results
                        stress_results = self.results.get('stress_analysis', {})
                        if stress_results and isinstance(stress_results, dict):
                            # Get stress parameters from input_params (used in stress analysis)
                            stress_params = {}
                            
                            # Check if stress analysis was performed and get parameters
                            if input_params.get('stress_bool', False):
                                # Get stress parameters directly from input_params
                                # These are the same parameters used in stress_analysis.fault_stress()
                                stress_params['S1_trend'] = input_params.get('S1_trend')
                                stress_params['S1_plunge'] = input_params.get('S1_plunge')
                                stress_params['S3_trend'] = input_params.get('S3_trend')
                                stress_params['S3_plunge'] = input_params.get('S3_plunge')
                                stress_params['stress_R'] = input_params.get('stress_R')
                                stress_params['PP'] = input_params.get('PP', 0.0)
                                stress_params['fric_coeff'] = input_params.get('fric_coeff', 0.75)
                                
                                # Verify we have valid parameters
                                if all(param is not None and not (isinstance(param, float) and np.isnan(param)) 
                                       for param in [stress_params['S1_trend'], stress_params['S1_plunge'],
                                                   stress_params['S3_trend'], stress_params['S3_plunge'],
                                                   stress_params['stress_R']]):
                                    # Calculate stress on mesh faces
                                    # Note: Mesh subdivision is already handled in create_interpolated_fault_planes()
                                    from hyfi.core.stress_analysis import calculate_mesh_stress
                                    combined_mesh = calculate_mesh_stress(combined_mesh, stress_params)
                                    
                                    # Also apply to individual meshes
                                    for mesh_info in individual_meshes:
                                        mesh_info['mesh'] = calculate_mesh_stress(mesh_info['mesh'], stress_params)
                                else:
                                    logger.info("Stress parameters incomplete - skipping mesh stress calculation")
                            else:
                                logger.info("Stress analysis not enabled - skipping mesh stress calculation")
                    
                    # Export to VTK if requested
                    if viz_params.get('export_vtp', False) and combined_mesh is not None:
                        output_dir = input_params.get('out_dir', str(self.output_dir))
                        visualisation.export_interpolated_planes_vtp(
                            combined_mesh, individual_meshes, point_cloud, output_dir, fault_disc_meshes, df_hyfi,
                            use_focal_constraints=input_params.get('use_focal_constraints', False),
                            export_time_series=viz_params.get('export_time_series', False),
                            time_step_hours=viz_params.get('time_step_hours', 24)
                        )
                                        
                except Exception as e:
                    logger.warning(f"Fault plane interpolation failed: {e}")
            else:
                logger.info("Fault plane interpolation disabled - skipping advanced visualization")

            # Generate standard visualizations using single dataframe approach
            try:
                # Generate 3D model
                if viz_params.get('generate_3d_model', True):
                    visualisation.model_3d_single_df(df_hyfi, input_params)
                    
                # Generate stereoplot (still needs legacy format for now)
                if viz_params.get('generate_stereoplot', True):
                    # Create temporary data_output for stereoplot (legacy function)
                    events_with_planes = df_hyfi[df_hyfi['rupt_plane_azi'].notna()].copy()
                    if not events_with_planes.empty:
                        output_cols = ['ID', 'rupt_plane_azi', 'rupt_plane_dip'] + [col for col in df_hyfi.columns 
                                     if col not in ['ID', 'rupt_plane_azi', 'rupt_plane_dip'] and col not in 
                                     ['LAT', 'LON', 'DEPTH', 'X', 'Y', 'Z', 'Date']]
                        data_output = events_with_planes[output_cols].copy()
                        visualisation.faults_stereoplot(input_params, data_output)
                                
            except Exception as e:
                logger.warning(f"Standard visualization failed: {e}")

            result = {
                'visualization_completed': True,
                'data_for_visualization': df_hyfi,
                'parameters': node.parameters
            }
            
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
            result = {
                'visualization_error': str(e),
                'data_for_visualization': df_hyfi,
                'parameters': node.parameters
            }
        
        
        return {
            'visualization_results': result,
            'df_hyfi': df_hyfi,
            'parameters': node.parameters
        }
    
    def _execute_outlier_visualization(self, node) -> Dict[str, Any]:
        """Execute simplified visualization for outlier clusters."""
        logger.info(f"Creating simplified visualization for outlier cluster: {self.cluster_name}")
        
        # Get input data (hypocenters only)
        input_data_results = self.results.get('input_data', {})
        if not input_data_results or 'hypocenter_data' not in input_data_results:
            logger.warning("No input data available for outlier visualization")
            return {'outlier_visualization_error': 'No input data'}
        
        hypocenter_data = input_data_results['hypocenter_data']
        
        try:
            # Create VTK export of hypocenter points
            viz_params = node.parameters or {}
            if viz_params.get('export_vtp', False):
                output_dir = str(self.output_dir)
                self._export_outlier_hypocenters_vtk(hypocenter_data, output_dir)
            
            # Create simple 3D visualization of outlier points
            if viz_params.get('generate_3d_model', True):
                self._create_outlier_3d_plot(hypocenter_data, str(self.output_dir))
            
            result = {
                'outlier_visualization_completed': True,
                'cluster_name': self.cluster_name,
                'n_outliers': len(hypocenter_data),
                'vtp_exported': viz_params.get('export_vtp', False),
                'parameters': node.parameters
            }
            
        except Exception as e:
            logger.error(f"Outlier visualization failed: {e}")
            result = {
                'outlier_visualization_error': str(e),
                'cluster_name': self.cluster_name,
                'parameters': node.parameters
            }
        
        return {
            'visualization_results': result,
            'outlier_data': hypocenter_data,
            'parameters': node.parameters
        }
    
    def _save_workflow_results(self) -> None:
        """
        Automatically save all workflow results using single dataframe approach.
        This replaces the need for an explicit output node.
        """
        # Get the final dataframe from the last processing step
        df_hyfi = None
        
        # Try to get df_hyfi from the latest processing steps in reverse order
        for step in ['stress_analysis', 'auto_classification', 'model_validation', 'fault_network']:
            if step in self.results and self.results[step]:
                step_data = self.results[step]
                if isinstance(step_data, dict) and 'df_hyfi' in step_data:
                    df_hyfi = step_data['df_hyfi']
                    break
        
        if df_hyfi is None:
            logger.warning("No df_hyfi data available for saving")
            return
        
        # Get input parameters for file naming and output directory
        fault_network_results = self.results.get('fault_network', {})
        input_params = fault_network_results.get('legacy_params', {}) if isinstance(fault_network_results, dict) else {}
        
        # Set output directory
        output_dir = input_params.get('out_dir', str(self.output_dir))
        
        try:
            # Calculate execution time properly
            if self.start_time:
                execution_time = (datetime.now() - self.start_time).total_seconds()
            else:
                execution_time = 0
            
            # Save the single dataframe as CSV
            output_file = f"{output_dir}/HyFI_results.csv"
            df_hyfi.to_csv(output_file, index=False)
            
            # Create execution summary
            summary = {
                'workflow_execution_time': execution_time,
                'total_events': int(len(df_hyfi)),
                'events_with_fault_planes': int(df_hyfi['rupt_plane_azi'].count()),
                'workflow_steps_completed': list(self.results.keys()),
                'execution_date': datetime.now().isoformat()
            }
            
            # Add step-specific statistics
            if 'epsilon' in df_hyfi.columns:
                summary['focal_mechanisms_validated'] = int(df_hyfi['epsilon'].count())
            if 'Sn_eff' in df_hyfi.columns:
                summary['stress_analysis_completed'] = int(df_hyfi['Sn_eff'].count())
            if 'final_cluster_id' in df_hyfi.columns:
                summary['fault_clusters'] = int(df_hyfi['final_cluster_id'].nunique())
            
            # Save summary
            import json
            summary_file = f"{output_dir}/execution_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving workflow results: {e}")
    
    def _export_outlier_hypocenters_vtk(self, hypocenter_data, output_dir):
        """Export outlier hypocenters as VTK point cloud."""
        try:
            import pyvista as pv
            import os
            
            # Create VTK export directory
            vtp_dir = os.path.join(output_dir, 'vtp_export')
            os.makedirs(vtp_dir, exist_ok=True)
            
            # Prepare hypocenter data (use Z coordinates directly as provided)
            hypocenter_data = hypocenter_data.copy()
            
            # Create point cloud from hypocenters
            points = hypocenter_data[['X', 'Y', 'Z']].values
            outlier_pcd = pv.PolyData(points)
            
            # Add hypocenter data as point arrays
            for col in ['ID', 'LAT', 'LON', 'DEPTH', 'ML', 'MAG']:
                if col in hypocenter_data.columns:
                    outlier_pcd[col] = hypocenter_data[col].values
            
            # Handle Date column separately
            if 'Date' in hypocenter_data.columns:
                try:
                    date_strings = hypocenter_data['Date'].astype(str).values
                    outlier_pcd['Date'] = date_strings
                except Exception as e:
                    logger.warning(f"Could not add Date column to VTK: {e}")
            
            # Export VTK file
            vtp_file = os.path.join(vtp_dir, f'{self.cluster_name}_hypocenters.vtp')
            outlier_pcd.save(vtp_file)
            
            logger.info(f"Exported {len(points)} outlier hypocenters to: {vtp_file}")
            
        except Exception as e:
            logger.error(f"Failed to export outlier hypocenters to VTK: {e}")
    
    def _create_outlier_3d_plot(self, hypocenter_data, output_dir):
        """Create simple 3D plot for outlier hypocenters."""
        try:
            import plotly.graph_objects as go
            import os
            
            # Prepare hypocenter data (use Z coordinates directly as provided)
            hypocenter_data = hypocenter_data.copy()
            
            # Create 3D scatter plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter3d(
                x=hypocenter_data['X'],
                y=hypocenter_data['Y'],
                z=hypocenter_data['Z'],
                mode='markers',
                marker=dict(
                    color='red',
                    size=4,
                    opacity=0.8
                ),
                name=f'{self.cluster_name} ({len(hypocenter_data)} events)',
                hovertemplate=(
                    '<b>Event ID:</b> %{customdata[0]}<br>'
                    '<b>Coordinates:</b> %{x:.0f}, %{y:.0f}, %{z:.0f}<br>'
                    '<b>Magnitude:</b> %{customdata[1]:.1f}<br>'
                    '<extra></extra>'
                ),
                customdata=hypocenter_data[['ID', 'ML']].values if 'ML' in hypocenter_data.columns else None
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Outlier Hypocenters: {self.cluster_name}',
                scene=dict(
                    xaxis_title='Easting [m]',
                    yaxis_title='Northing [m]', 
                    zaxis_title='Depth [m]',
                    aspectmode='auto'
                ),
                showlegend=True
            )
            
            # Save HTML file
            html_file = os.path.join(output_dir, f'{self.cluster_name}_3D_model.html')
            fig.write_html(html_file)
            
            logger.info(f"Created 3D plot for {len(hypocenter_data)} outliers: {html_file}")
            
        except Exception as e:
            logger.error(f"Failed to create outlier 3D plot: {e}")

        
    def _save_execution_summary(self) -> None:
        """Save execution summary to file."""
        summary = {
            'workflow_name': self.dag.workflow_name,
            'execution_start': self.start_time.isoformat(),
            'execution_end': self.end_time.isoformat(),
            'total_duration_seconds': (self.end_time - self.start_time).total_seconds(),
            'nodes_executed': len([log for log in self.execution_log if log['success']]),
            'nodes_failed': len([log for log in self.execution_log if not log['success']]),
            'execution_log': self.execution_log,
            'output_directory': str(self.output_dir)
        }
        
        summary_file = self.output_dir / 'execution_summary.json'
        
        # Save summary using the utility function (needs directory, not file path)
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get execution summary.
        
        Returns
        -------
        Dict[str, Any]
            Summary of execution results
        """
        if not self.start_time:
            return {'status': 'not_started'}
        
        summary = {
            'workflow_name': self.dag.workflow_name,
            'status': 'completed' if self.end_time else 'running',
            'nodes_executed': len([log for log in self.execution_log if log['success']]),
            'nodes_failed': len([log for log in self.execution_log if not log['success']]),
            'output_directory': str(self.output_dir)
        }
        
        if self.end_time:
            summary['total_duration'] = (self.end_time - self.start_time).total_seconds()
        
        return summary
