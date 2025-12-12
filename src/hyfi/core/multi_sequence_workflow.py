#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-sequence workflow for processing full earthquake catalogs.

This module provides functionality to segment earthquake catalogs into sequences
and process each sequence through the standard HyFI workflow.
"""

import datetime
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
import pyproj


from .workflow import FaultImagingWorkflow
from ..config.parameters import ProjectConfig
from ..config.multi_sequence_config import MultiSequenceConfig
from ..utils import utilities
from ..visualization import visualisation


logger = logging.getLogger(__name__)


class MultiSequenceWorkflow:
    """
    Multi-sequence workflow for processing full earthquake catalogs.
    
    This class segments earthquake catalogs using clustering algorithms
    and applies the standard HyFI workflow to each identified sequence.
    """
    
    def __init__(self, config: MultiSequenceConfig, config_source_file: Optional[str] = None):
        """
        Initialize the multi-sequence workflow.
        
        Parameters
        ----------
        config : MultiSequenceConfig
            Configuration object containing clustering and workflow parameters
        config_source_file : str, optional
            Path to the source configuration file (used for resolving relative paths)
        """
        self.config = config
        self.config_source_file = config_source_file
        
        # Resolve relative paths BEFORE validation if config_source_file is provided
        if config_source_file:
            config_dir = Path(config_source_file).parent.resolve()
            
            # Resolve catalog_file path
            if self.config.catalog_file:
                catalog_path = Path(self.config.catalog_file)
                if not catalog_path.is_absolute():
                    self.config.catalog_file = (config_dir / catalog_path).resolve()
            
            # Resolve output_directory path
            if self.config.output_directory:
                output_path = Path(self.config.output_directory)
                if not output_path.is_absolute():
                    self.config.output_directory = (config_dir / output_path).resolve()
            
            # Resolve focal mechanism file path in template config if present
            if (self.config.template_config and 
                hasattr(self.config.template_config, 'model_validation') and
                self.config.template_config.model_validation.foc_file):
                foc_path = Path(self.config.template_config.model_validation.foc_file)
                if not foc_path.is_absolute():
                    self.config.template_config.model_validation.foc_file = (config_dir / foc_path).resolve()
        
        # Now validate after paths are resolved
        self.config.validate()
        
        # Results storage
        self.full_catalog = None
        self.sequences = {}
        self.sequence_results = {}
        self.aggregated_results = {}
        
        # Global fault system counter (FS0001, FS0002, ...)
        self.global_fault_counter = 1
        self.fault_system_metadata = []  # Track all fault systems across sequences
        
        # Timing
        self.start_time = None
        self.end_time = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
    def run_full_multi_sequence_analysis(self) -> Dict[str, Any]:
        """
        Run the complete multi-sequence workflow.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing results from all sequences and aggregated analysis
        """
        print('')
        print('###   MULTI-SEQUENCE HYPOCENTER-BASED 3D FAULT IMAGING   ###')
        print('Processing full earthquake catalog with clustering...')
        print('')
        
        self.start_time = time.time()
        
        # Step 1: Load full catalog
        self._load_catalog()
        
        # Step 2: Apply clustering/segmentation
        self._segment_catalog()
        
        # Step 3: Process each sequence with single-sequence workflow
        self._process_sequences()
        
        # Step 4: Aggregate results across sequences
        self._aggregate_results()
        
        # Step 5: Create multi-sequence visualizations
        self._create_multi_sequence_visualizations()
        
        # Step 6: Create enriched CSV output
        self._create_enriched_csv_output()
        
        # Step 6.5: Export fault system metadata
        self._export_fault_system_metadata()
        
        # Step 7: Merge and export combined VTP files
        self._merge_and_export_vtp_files()
        
        # Step 8: Export to SQL database
        self._export_to_sql_database()
        
        # Step 9: Save all results
        self._save_multi_sequence_results()
        
        self.end_time = time.time()
        runtime = datetime.timedelta(seconds=(self.end_time - self.start_time))
        
        print('')
        print('Multi-sequence analysis completed!')
        print(f'Total runtime: {runtime}')
        print(f'Processed {len(self.sequences)} sequences')
        
        runtime_seconds = self.end_time - self.start_time
        hours, remainder = divmod(int(runtime_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        runtime_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        return {
            'sequences': self.sequences,
            'sequence_results': self.sequence_results,
            'aggregated_results': self.aggregated_results,
            'metadata': {
                'n_sequences': len(self.sequences),
                'total_events': len(self.full_catalog),
                'runtime_seconds': runtime_seconds,
                'runtime_formatted': runtime_formatted
            }
        }
    
    def _load_catalog(self):
        """Load the full earthquake catalog."""
        print("Loading full earthquake catalog...")
        
        # Load hypocenter data (only first 24 columns - standard hypoDD format)
        self.full_catalog = pd.read_csv(self.config.catalog_file, sep=self.config.catalog_sep, usecols=range(24))
        
        # Process temporal information
        self.full_catalog['Date'] = pd.to_datetime(pd.DataFrame({
            'year': self.full_catalog['YR'],
            'month': self.full_catalog['MO'],
            'day': self.full_catalog['DY'],
            'hour': self.full_catalog['HR'],
            'minute': self.full_catalog['MI'],
            'second': self.full_catalog['SC']
        }))
        
        # Use Z coordinates directly as provided in input data
        # (Z values are already negative, representing depth below surface)
        
        print(f"Loaded {len(self.full_catalog)} events")
        
    def _segment_catalog(self):
        """Segment the catalog into sequences using multi-step clustering."""
        print(f"Segmenting catalog using multi-step clustering...")
        print(f"Number of segmentation steps: {len(self.config.segmentation_steps)}")
        
        # Import the updated segmentation function
        from ..utils.catalog_segmentation import multi_step_catalog_segmentation
        
        # Apply multi-step segmentation
        self.sequences, self.clustering_results = multi_step_catalog_segmentation(
            self.full_catalog,
            self.config.segmentation_steps,
            self.config.clustering.final_outlier_handling,
            self.config.clustering.max_outlier_ratio
        )
        
        print(f"Multi-step clustering completed:")
        print(f"  Total sequences identified: {self.clustering_results['total_sequences']}")
        print(f"  Events clustered: {self.clustering_results['total_events_clustered']}")
        print(f"  Final outliers: {self.clustering_results['final_outliers']} ({self.clustering_results['outlier_ratio']:.2%})")
        
        # Print detailed step results
        for step_name, step_result in self.clustering_results['step_results'].items():
            print(f"  Step '{step_name}': {step_result['sequences_found']} sequences, "
                  f"{step_result['events_clustered']} events clustered, "
                  f"{step_result['outliers']} outliers")
        
        # Store results for later analysis
        self.aggregated_results['clustering_details'] = self.clustering_results
    
    def _process_sequences(self):
        """Process each sequence through the DAG-based workflow."""
        print("Processing individual sequences...")
        
        for sequence_name, sequence_data in self.sequences.items():
            if sequence_name == 'noise':
                continue  # Skip noise sequence
            
            print('\n')
            print("=" * 60)
            print(f"Processing {sequence_name} ({len(sequence_data)} events)...")
            print("=" * 60)
            print('\n')

            try:
                # Create sequence-specific DAG configuration
                sequence_dag_config = self._create_sequence_dag_config(sequence_name, sequence_data)
                
                # Run DAG-based workflow
                from .dag_executor import DAGExecutor
                executor = DAGExecutor(
                    sequence_dag_config, 
                    cluster_name=sequence_name,
                    global_fault_counter=self.global_fault_counter,
                    sequence_label=sequence_name,
                    segmentation_level=sequence_name[0] if sequence_name and len(sequence_name) > 0 else None
                )
                sequence_results = executor.execute()
                
                # Update global counter and collect metadata
                if 'fault_system_metadata' in sequence_results:
                    self.fault_system_metadata.extend(sequence_results['fault_system_metadata'])
                if 'next_fault_counter' in sequence_results:
                    self.global_fault_counter = sequence_results['next_fault_counter']
                
                # Store results
                self.sequence_results[sequence_name] = {
                    'workflow_results': sequence_results,
                    'summary': executor.get_execution_summary(),
                    'input_events': len(sequence_data),
                    'config': sequence_dag_config
                }
                
                print(f"Completed {sequence_name}: {executor.get_execution_summary()}")
                
            except Exception as e:
                logger.error(f"Failed to process {sequence_name}: {e}")
                self.sequence_results[sequence_name] = {
                    'error': str(e),
                    'input_events': len(sequence_data)
                }
    
    def _create_sequence_config(self, sequence_name: str, sequence_data: pd.DataFrame) -> ProjectConfig:
        """Create a configuration for processing a single sequence."""
        # Create a temporary file for this sequence
        sequence_dir = Path(self.config.output_directory) / sequence_name
        sequence_dir.mkdir(parents=True, exist_ok=True)
        
        sequence_file = sequence_dir / f"{sequence_name}_data.txt"
        sequence_data.to_csv(sequence_file, sep=self.config.catalog_sep, index=False)
        
        # Clone the template configuration
        sequence_config = ProjectConfig(
            project_title=f"{self.config.project_title} - {sequence_name}",
            hypo_file=sequence_file,
            hypo_sep=self.config.catalog_sep,
            out_dir=sequence_dir
        )
        
        # Copy module configurations from template
        sequence_config.fault_network = self.config.template_config.fault_network
        sequence_config.model_validation = self.config.template_config.model_validation
        sequence_config.auto_class = self.config.template_config.auto_class
        sequence_config.stress_analysis = self.config.template_config.stress_analysis
        
        # Ensure focal mechanism file path is absolute (it should already be resolved 
        # in __post_init__, but double-check for safety)
        if (sequence_config.model_validation.validation_bool and 
            sequence_config.model_validation.foc_file):
            foc_path = Path(sequence_config.model_validation.foc_file)
            if not foc_path.is_absolute():
                # Resolve relative to the original project root, not sequence directory
                original_cwd = Path.cwd()
                sequence_config.model_validation.foc_file = (original_cwd / foc_path).resolve()
        
        return sequence_config
    
    def _create_sequence_dag_config(self, sequence_name: str, sequence_data: pd.DataFrame):
        """Create a DAG configuration for processing a single sequence."""
        from ..config.schema import HyFIWorkflowDAG
        
        # Create a temporary file for this sequence
        sequence_dir = Path(self.config.output_directory) / sequence_name
        sequence_dir.mkdir(parents=True, exist_ok=True)
        
        sequence_file = sequence_dir / f"{sequence_name}_data.csv"
        sequence_data.to_csv(sequence_file, sep=self.config.catalog_sep, index=False)
        
        # Get the sequence workflow template from config
        if hasattr(self.config, 'cluster_workflow_template'):
            template = self.config.cluster_workflow_template
        elif hasattr(self.config, 'sequence_workflow_template'):
            template = self.config.sequence_workflow_template
        else:
            # Fallback to template_config if using old format
            template = {
                "metadata": {
                    "workflow_name": f"{self.config.project_title} - {sequence_name}",
                    "workflow_version": "1.0.0"
                },
                "global_settings": {
                    "output_directory": str(sequence_dir)
                },
                "workflow_dag": self._convert_template_config_to_dag()
            }
        
        # Create base DAG configuration dictionary
        dag_dict = {
            "metadata": {
                "workflow_name": f"{template['metadata']['workflow_name']} - {sequence_name}",
                "workflow_version": template['metadata'].get('workflow_version', '1.0.0'),
                "created_date": template['metadata'].get('created_date', '2025-10-03T00:00:00')
            },
            "global_settings": {
                "output_directory": str(sequence_dir)
            },
            "workflow_dag": {}
        }
        
        # Copy template DAG settings
        if 'workflow_dag' in template:
            dag_template = template['workflow_dag']
            
            # Detect which input data node name is used in template
            input_node_name = 'step_1_load_data' if 'step_1_load_data' in dag_template else 'input_data'
            template_input = dag_template.get(input_node_name, {})
            
            # Get focal mechanism file path and resolve if relative
            focal_mech_file = template_input.get('focal_mechanism_file', '')
            if focal_mech_file:
                focal_path = Path(focal_mech_file)
                # If path is relative, resolve it relative to the original config file
                if not focal_path.is_absolute() and self.config_source_file:
                    config_dir = Path(self.config_source_file).parent.resolve()
                    focal_mech_file = str((config_dir / focal_path).resolve())
            
            # Set up input_data node with sequence-specific file
            dag_dict['workflow_dag']['input_data'] = {
                "hypocenter_file": str(sequence_file),
                "hypocenter_separator": self.config.catalog_sep,
                "focal_mechanism_file": focal_mech_file,
                "focal_mechanism_separator": template_input.get('focal_mechanism_separator', ',')
            }
            
            # Copy all other DAG nodes from template (except the input node and multi-sequence specific nodes)
            for node_name, node_config in dag_template.items():
                if node_name not in ['input_data', 'step_1_load_data', 'step_2_catalog_segmentation', 'step_4_merge_and_export']:
                    # For step_3_per_sequence_analysis, unwrap it and copy its contents
                    if node_name == 'step_3_per_sequence_analysis':
                        # Copy all sub-nodes from step_3 and flatten nested parameters
                        for sub_node_name, sub_node_config in node_config.items():
                            if sub_node_name != 'description':
                                # Deep copy the config
                                import copy
                                copied_config = copy.deepcopy(sub_node_config)
                                
                                # Flatten nested parameter structures for backward compatibility
                                if isinstance(copied_config, dict) and 'parameters' in copied_config:
                                    copied_config['parameters'] = self._flatten_nested_params(copied_config['parameters'])
                                
                                dag_dict['workflow_dag'][sub_node_name] = copied_config
                    else:
                        dag_dict['workflow_dag'][node_name] = node_config.copy() if isinstance(node_config, dict) else node_config
        
        # Create HyFIWorkflowDAG from dictionary
        return HyFIWorkflowDAG.from_dict(dag_dict)
    
    def _flatten_nested_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested parameter structures for backward compatibility.
        
        Converts nested structures like:
        {
            "core_network": {"monte_carlo_simulations": 1},
            "outlier_detection": {"remove_outliers": true}
        }
        
        To flat structure:
        {
            "monte_carlo_simulations": 1,
            "remove_outliers": true
        }
        """
        if not isinstance(params, dict):
            return params
        
        flattened = {}
        for key, value in params.items():
            if isinstance(value, dict):
                # Recursively flatten nested dicts
                flattened.update(self._flatten_nested_params(value))
            else:
                flattened[key] = value
        
        return flattened
    
    def _convert_template_config_to_dag(self):
        """Convert old template_config format to DAG format."""
        if not hasattr(self.config, 'template_config'):
            return {}
        
        tc = self.config.template_config
        
        return {
            "fault_network": {
                "parameters": {
                    "monte_carlo_simulations": getattr(tc.fault_network, 'n_mc', 1),
                    "search_radius_meters": getattr(tc.fault_network, 'r_nn', 200.0),
                    "search_time_window_hours": getattr(tc.fault_network, 'dt_nn', 999999.0),
                    "magnitude_type": getattr(tc.fault_network, 'mag_type', 'ML'),
                    "auto_optimize_parameters": False,
                    "remove_outliers": True,
                    "use_focal_constraints": True
                }
            },
            "model_validation": {
                "enabled": getattr(tc.model_validation, 'validation_bool', False),
                "parameters": {
                    "check_magnitude_consistency": getattr(tc.model_validation, 'foc_mag_check', True),
                    "check_location_consistency": getattr(tc.model_validation, 'foc_loc_check', True)
                }
            },
            "auto_classification": {
                "enabled": getattr(tc.auto_class, 'autoclass_bool', False),
                "parameters": {
                    "auto_determine_clusters": False,
                    "max_clusters": getattr(tc.auto_class, 'n_clusters', 2),
                    "clustering_algorithm": getattr(tc.auto_class, 'algorithm', 'vmf_soft'),
                    "rotate_poles_before_analysis": getattr(tc.auto_class, 'rotation', True)
                }
            },
            "stress_analysis": {
                "enabled": getattr(tc.stress_analysis, 'stress_bool', False),
                "parameters": {
                    "stress_field": {
                        "sigma1_trend_degrees": getattr(tc.stress_analysis, 'S1_trend', 301),
                        "sigma1_plunge_degrees": getattr(tc.stress_analysis, 'S1_plunge', 23),
                        "sigma3_trend_degrees": getattr(tc.stress_analysis, 'S3_trend', 43),
                        "sigma3_plunge_degrees": getattr(tc.stress_analysis, 'S3_plunge', 26),
                        "stress_shape_ratio": getattr(tc.stress_analysis, 'stress_R', 0.35)
                    },
                    "mechanical_properties": {
                        "pore_pressure_mpa": getattr(tc.stress_analysis, 'PP', 0.0),
                        "friction_coefficient": getattr(tc.stress_analysis, 'fric_coeff', 0.75)
                    }
                }
            },
            "visualization": {
                "enabled": True,
                "parameters": {
                    "generate_3d_model": True,
                    "generate_stereonet": True,
                    "enable_plane_interpolation": True,
                    "export_vtp": True
                }
            }
        }
    
    def _aggregate_results(self):
        """Aggregate results across all sequences."""
        print("Aggregating results across sequences...")
        
        successful_sequences = [name for name, result in self.sequence_results.items() 
                             if 'workflow_results' in result]
        
        if not successful_sequences:
            print("No successful sequence analyses to aggregate")
            return
        
        # Aggregate fault networks
        all_fault_planes = []
        all_input_data = []
        sequence_statistics = {}
        
        for sequence_name in successful_sequences:
            result = self.sequence_results[sequence_name]['workflow_results']
            
            # Check if fault_network results exist and have the expected structure
            if 'fault_network' in result and isinstance(result['fault_network'], dict):
                # Check for data_output (fault planes)
                if 'data_output' in result['fault_network']:
                    fault_data = result['fault_network']['data_output']
                    if fault_data is not None and len(fault_data) > 0:
                        # Add sequence identifier
                        fault_data_copy = fault_data.copy()
                        fault_data_copy['source_cluster'] = sequence_name
                        all_fault_planes.append(fault_data_copy)
                
                # Check for data_input (hypocenter data)
                if 'data_input' in result['fault_network']:
                    input_data = result['fault_network']['data_input']
                    if input_data is not None and len(input_data) > 0:
                        input_data_copy = input_data.copy()
                        input_data_copy['source_cluster'] = sequence_name
                        all_input_data.append(input_data_copy)
            
            # For outlier sequences or sequences without fault network data,
            # try to get input data from visualization results
            elif 'visualization' in result and isinstance(result['visualization'], dict):
                if 'input_data' in result['visualization']:
                    input_data = result['visualization']['input_data']
                    if input_data is not None and len(input_data) > 0:
                        input_data_copy = input_data.copy()
                        input_data_copy['source_cluster'] = sequence_name
                        all_input_data.append(input_data_copy)
            
            # Collect statistics
            summary = self.sequence_results[sequence_name]['summary']
            sequence_statistics[sequence_name] = summary
        
        # Combine results
        if all_fault_planes:
            self.aggregated_results['combined_fault_planes'] = pd.concat(all_fault_planes, ignore_index=True)
            print(f"Aggregated {len(self.aggregated_results['combined_fault_planes'])} fault planes from {len(successful_sequences)} sequences")
        
        if all_input_data:
            self.aggregated_results['combined_input_data'] = pd.concat(all_input_data, ignore_index=True)
        
        self.aggregated_results['sequence_statistics'] = sequence_statistics
        self.aggregated_results['successful_sequences'] = successful_sequences
    
    def _create_enriched_csv_output(self):
        """Create an enriched CSV file combining input data with all analysis results."""
        print("Creating enriched CSV output with analysis results...")
        
        # Start with the original catalog as base
        enriched_data = self.full_catalog.copy()
        
        # Add clustering information
        enriched_data['sequence_label'] = 'unclustered'
        enriched_data['segmentation_level'] = None
        
        # Add sequence labels from the clustering results
        for sequence_name, sequence_events in self.sequences.items():
            if sequence_name == 'noise':
                continue
                
            # Find the sequence events in the main catalog by ID
            sequence_ids = sequence_events['ID'].values
            mask = enriched_data['ID'].isin(sequence_ids)
            enriched_data.loc[mask, 'sequence_label'] = sequence_name
            
            # Extract segmentation level from sequence name (A, B, C, etc.)
            if sequence_name and len(sequence_name) > 0:
                segmentation_level = sequence_name[0]  # First character (A, B, C)
                enriched_data.loc[mask, 'segmentation_level'] = segmentation_level
        
        # Mark outliers as unclustered
        if 'Z_outliers' in self.sequences:
            outlier_ids = self.sequences['Z_outliers']['ID'].values
            outlier_mask = enriched_data['ID'].isin(outlier_ids)
            enriched_data.loc[outlier_mask, 'sequence_label'] = 'unclustered'
            enriched_data.loc[outlier_mask, 'segmentation_level'] = None
        
        # Initialize analysis result columns with NaN
        analysis_columns = {
            # Fault network results
            'rupture_plane_azimuth': np.nan,
            'rupture_plane_dip': np.nan, 
            'rupture_plane_strike': np.nan,
            'normal_vector_x': np.nan,
            'normal_vector_y': np.nan,
            'normal_vector_z': np.nan,
            'nvector_quality_R': np.nan,
            'nvector_quality_N': np.nan,
            'nvector_quality_ratio': np.nan,
            'kent_kappa': np.nan,
            'kent_beta': np.nan,
            'nr_rupt_fits': np.nan,
            
            # Stress analysis results
            'effective_normal_stress': np.nan,
            'shear_stress': np.nan,
            'rake_angle': np.nan,
            'instability_index': np.nan,
            'slip_tendency': np.nan,
            'dilation_tendency': np.nan,
            
            # Fault system clustering results
            'fault_system_id': None,  # Global FS ID (FS0001, FS0002, ...)
            'orientation_cluster': np.nan,
            'spatial_cluster': np.nan,
            
            # Additional metadata
            'analysis_status': 'not_processed',
            'sequence_outlier': False
        }
        
        # Add columns to enriched data
        for col_name, default_value in analysis_columns.items():
            enriched_data[col_name] = default_value
        
        # Fill in analysis results by reading HyFI_results.csv files from sequence directories
        output_dir = Path(self.config.output_directory)
        
        for sequence_name, sequence_result in self.sequence_results.items():
            if 'workflow_results' not in sequence_result:
                continue
            
            # Check if this sequence has a HyFI_results.csv file
            sequence_results_file = output_dir / sequence_name / 'HyFI_results.csv'
            if not sequence_results_file.exists():
                # Mark as processed but without detailed results (e.g., Z_outliers)
                sequence_events = self.sequences.get(sequence_name)
                if sequence_events is not None:
                    sequence_ids = sequence_events['ID'].values
                    mask = enriched_data['ID'].isin(sequence_ids)
                    enriched_data.loc[mask, 'analysis_status'] = 'processed_no_results'
                continue
            
            try:
                # Read the cluster's HyFI results
                cluster_hyfi_results = pd.read_csv(sequence_results_file)
                
                # Get cluster event IDs
                sequence_events = self.sequences.get(sequence_name)
                if sequence_events is None:
                    continue
                
                sequence_ids = sequence_events['ID'].values
                mask = enriched_data['ID'].isin(sequence_ids)
                
                # Update analysis status
                enriched_data.loc[mask, 'analysis_status'] = 'processed'
                
                # Map HyFI results to enriched data by matching IDs
                for _, hyfi_row in cluster_hyfi_results.iterrows():
                    event_id = hyfi_row['ID']
                    event_mask = enriched_data['ID'] == event_id
                    
                    if event_mask.any():
                        # Fault plane orientation (using mean values from HyFI results)
                        enriched_data.loc[event_mask, 'rupture_plane_azimuth'] = hyfi_row.get('rupt_plane_azi', np.nan)
                        enriched_data.loc[event_mask, 'rupture_plane_dip'] = hyfi_row.get('rupt_plane_dip', np.nan)
                        
                        # Calculate strike from azimuth (azimuth - 90 degrees)
                        if not pd.isna(hyfi_row.get('rupt_plane_azi')):
                            strike = (hyfi_row.get('rupt_plane_azi') - 90) % 360
                            enriched_data.loc[event_mask, 'rupture_plane_strike'] = strike
                        
                        # Normal vector components
                        enriched_data.loc[event_mask, 'normal_vector_x'] = hyfi_row.get('nor_x_mean', np.nan)
                        enriched_data.loc[event_mask, 'normal_vector_y'] = hyfi_row.get('nor_y_mean', np.nan)
                        enriched_data.loc[event_mask, 'normal_vector_z'] = hyfi_row.get('nor_z_mean', np.nan)
                        
                        # Quality metrics
                        enriched_data.loc[event_mask, 'nvector_quality_R'] = hyfi_row.get('R', np.nan)
                        enriched_data.loc[event_mask, 'nvector_quality_N'] = hyfi_row.get('N', np.nan)
                        enriched_data.loc[event_mask, 'nvector_quality_ratio'] = hyfi_row.get('R/N', np.nan)
                        enriched_data.loc[event_mask, 'kent_kappa'] = hyfi_row.get('kappa', np.nan)
                        enriched_data.loc[event_mask, 'kent_beta'] = hyfi_row.get('beta', np.nan)
                        enriched_data.loc[event_mask, 'nr_rupt_fits'] = hyfi_row.get('nr_fits', np.nan)
                        
                        # Stress analysis results (if available)
                        enriched_data.loc[event_mask, 'effective_normal_stress'] = hyfi_row.get('Sn_eff', np.nan)
                        enriched_data.loc[event_mask, 'shear_stress'] = hyfi_row.get('Tau', np.nan)
                        enriched_data.loc[event_mask, 'rake_angle'] = hyfi_row.get('rake', np.nan)
                        enriched_data.loc[event_mask, 'instability_index'] = hyfi_row.get('instab', np.nan)
                        enriched_data.loc[event_mask, 'slip_tendency'] = hyfi_row.get('sliptend', np.nan)
                        enriched_data.loc[event_mask, 'dilation_tendency'] = hyfi_row.get('dilatend', np.nan)
                        
                        # Fault system clustering (global FS IDs)
                        enriched_data.loc[event_mask, 'fault_system_id'] = hyfi_row.get('final_cluster_id', None)
                        enriched_data.loc[event_mask, 'orientation_cluster'] = hyfi_row.get('orient_cluster', np.nan)
                        enriched_data.loc[event_mask, 'spatial_cluster'] = hyfi_row.get('spatial_cluster', np.nan)
                
                print(f"  Merged {len(cluster_hyfi_results)} results from {sequence_name}")
                
            except Exception as e:
                print(f"  Warning: Could not read HyFI results for {sequence_name}: {e}")
                # Mark as processed but with error
                sequence_events = self.sequences.get(sequence_name)
                if sequence_events is not None:
                    sequence_ids = sequence_events['ID'].values
                    mask = enriched_data['ID'].isin(sequence_ids)
                    enriched_data.loc[mask, 'analysis_status'] = 'processed_with_error'
        
        # Save enriched CSV
        enriched_file = output_dir / 'HyFI_results_multi.csv'
        enriched_data.to_csv(enriched_file, index=False)
        
        # Store in aggregated results
        self.aggregated_results['enriched_catalog'] = enriched_data
        
        # Print summary
        total_events = len(enriched_data)
        clustered_events = len(enriched_data[enriched_data['sequence_label'] != 'unclustered'])
        processed_events = len(enriched_data[enriched_data['analysis_status'] == 'processed'])
        processed_no_results = len(enriched_data[enriched_data['analysis_status'] == 'processed_no_results'])
        outlier_events = len(enriched_data[enriched_data['sequence_outlier'] == True])
        
        print(f"HyFI results CSV of multi-sequence analysis created: {enriched_file}")
        print(f"  Total events: {total_events}")
        print(f"  Clustered events: {clustered_events} ({clustered_events/total_events:.1%})")
        print(f"  Successfully analyzed: {processed_events} ({processed_events/total_events:.1%})")
        if processed_no_results > 0:
            print(f"  Processed without detailed results: {processed_no_results} ({processed_no_results/total_events:.1%})")
        print(f"  Fault network outliers: {outlier_events} ({outlier_events/total_events:.1%})")
    
    def _export_fault_system_metadata(self):
        """Export comprehensive fault system metadata to CSV."""
        print("Exporting fault system metadata...")
        
        if not self.fault_system_metadata:
            print("  No fault system metadata available to export")
            return
        
        output_dir = Path(self.config.output_directory)
        metadata_file = output_dir / 'active_faults_database.csv'
        
        # Convert to DataFrame
        df_metadata = pd.DataFrame(self.fault_system_metadata)
        
        # Sort by fault_system_id
        df_metadata = df_metadata.sort_values('fault_system_id')
        
        # Add additional computed fields if possible
        if 'mean_strike' in df_metadata.columns and 'mean_dip' in df_metadata.columns:
            df_metadata['fault_orientation'] = df_metadata.apply(
                lambda row: f"Strike: {row['mean_strike']:.1f}°, Dip: {row['mean_dip']:.1f}°" 
                if not pd.isna(row['mean_strike']) and not pd.isna(row['mean_dip']) else None,
                axis=1
            )
        
        # Export to CSV
        df_metadata.to_csv(metadata_file, index=False)
        
        print(f"  Active faults database exported: {metadata_file}")
        print(f"  Total active fault systems: {len(df_metadata)}")
        
        # Print summary by sequence
        if 'sequence_label' in df_metadata.columns:
            sequence_counts = df_metadata['sequence_label'].value_counts()
            print(f"  Fault systems by sequence:")
            for seq, count in sequence_counts.items():
                print(f"    {seq}: {count} fault systems")
        
        # Store in aggregated results
        self.aggregated_results['fault_system_metadata'] = df_metadata
    
    def _merge_and_export_vtp_files(self):
        """Merge VTP files from all clusters and export combined versions."""
        print("Merging and exporting combined VTP files...")
        
        output_dir = Path(self.config.output_directory)
        combined_vtp_dir = output_dir / 'vtp_export_multi'
        combined_vtp_dir.mkdir(exist_ok=True)
        
        # Define the VTP files to merge
        vtp_files_to_merge = {
            'hypocenters_ALL.vtp': 'hypocenters.vtp',
            'faults_ALL.vtp': 'faults_compiled.vtp',
            'rupture_planes_ALL.vtp': 'rupture_planes.vtp',
            'enhanced_pointcloud_ALL.vtp': 'enhanced_pointcloud.vtp',
            'focals_ALL.vtp': 'focals_compiled.vtp',
            'slip_vectors_ALL.vtp': 'slip_vectors.vtp'
        }
        
        for combined_filename, source_filename in vtp_files_to_merge.items():
            try:
                # Collect all VTP files of this type from sequence directories
                vtp_files = []
                cluster_sources = []
                
                for sequence_name in self.sequence_results.keys():
                    if sequence_name == 'noise':
                        continue
                    
                    cluster_vtp_dir = output_dir / sequence_name / 'vtp_export'
                    source_vtp_file = cluster_vtp_dir / source_filename
                    
                    if source_vtp_file.exists():
                        vtp_files.append(source_vtp_file)
                        cluster_sources.append(sequence_name)
                
                # Handle special case for Z_outliers hypocenters
                if source_filename == 'hypocenters.vtp' and 'Z_outliers' in self.sequence_results:
                    z_outliers_vtp_dir = output_dir / 'Z_outliers' / 'vtp_export'
                    z_outliers_file = z_outliers_vtp_dir / 'Z_outliers_hypocenters.vtp'
                    if z_outliers_file.exists():
                        vtp_files.append(z_outliers_file)
                        cluster_sources.append('Z_outliers')
                
                if vtp_files:
                    combined_file = combined_vtp_dir / combined_filename
                    self._merge_vtp_files(vtp_files, combined_file, cluster_sources)
                    print(f"  Created {combined_filename} from {len(vtp_files)} clusters")
                    
                    # Create KML version if export_kml is enabled in visualization settings
                    export_kml = self._should_export_kml()
                    if export_kml:
                        kml_filename = combined_filename.replace('.vtp', '.kml')
                        combined_kml_file = combined_vtp_dir / kml_filename
                        self._export_vtp_to_kml(combined_file, combined_kml_file, cluster_sources)
                        print(f"  Created {kml_filename} from {combined_filename}")
                else:
                    print(f"  No {source_filename} files found to merge")
                    
            except Exception as e:
                print(f"  Warning: Could not merge {source_filename}: {e}")
        
        print(f"Combined VTP files saved to: {combined_vtp_dir}")
    
    def _should_export_kml(self) -> bool:
        """Check if KML export is enabled in visualization settings."""
        try:
            # Check if we have sequence_workflow_template with visualization settings
            if hasattr(self.config, 'sequence_workflow_template') and self.config.sequence_workflow_template:
                viz_config = self.config.sequence_workflow_template.get('workflow_dag', {}).get('visualization', {})
                return viz_config.get('parameters', {}).get('export_kml', True)
            # Fallback to checking individual cluster results for visualization settings
            for sequence_result in self.sequence_results.values():
                if 'config' in sequence_result:
                    dag_config = sequence_result['config']
                    if hasattr(dag_config, 'nodes') and 'visualization' in dag_config.nodes:
                        return dag_config.nodes['visualization'].parameters.get('export_kml', True)
            return True  # Default to True if not specified
        except Exception:
            return True  # Default to True if there's any error
    
    def _merge_vtp_files(self, vtp_files: List[Path], output_file: Path, cluster_sources: List[str]):
        """
        Merge multiple VTP files into a single VTK file.
        
        Parameters
        ----------
        vtp_files : List[Path]
            List of VTP files to merge
        output_file : Path
            Output file path for the merged VTK
        cluster_sources : List[str]
            List of cluster names corresponding to each VTK file
        """
        try:
            import vtk
        except ImportError:
            print("  Warning: VTK library not available, using simple text-based merging")
            self._merge_vtp_files_simple(vtp_files, output_file, cluster_sources)
            return
        
        # Create a new VTK data structure to hold all merged data
        merged_data = vtk.vtkAppendPolyData()
        
        point_count = 0
        
        for i, (vtp_file, sequence_name) in enumerate(zip(vtp_files, cluster_sources)):
            try:
                # Read the VTK file (XML format)
                reader = vtk.vtkXMLPolyDataReader()
                reader.SetFileName(str(vtp_file))
                reader.Update()
                
                polydata = reader.GetOutput()
                
                if polydata.GetNumberOfPoints() > 0:
                    # Add cluster information as point data
                    cluster_array = vtk.vtkStringArray()
                    cluster_array.SetName("cluster_source")
                    cluster_array.SetNumberOfTuples(polydata.GetNumberOfPoints())
                    
                    for j in range(polydata.GetNumberOfPoints()):
                        cluster_array.SetValue(j, sequence_name)
                    
                    polydata.GetPointData().AddArray(cluster_array)
                    
                    # Add cluster ID as numerical array for visualization
                    cluster_id_array = vtk.vtkIntArray()
                    cluster_id_array.SetName("cluster_id")
                    cluster_id_array.SetNumberOfTuples(polydata.GetNumberOfPoints())
                    
                    cluster_id = i  # Simple numerical ID based on order
                    for j in range(polydata.GetNumberOfPoints()):
                        cluster_id_array.SetValue(j, cluster_id)
                    
                    polydata.GetPointData().AddArray(cluster_id_array)
                    
                    # Append to merged data
                    merged_data.AddInputData(polydata)
                    point_count += polydata.GetNumberOfPoints()
                    
            except Exception as e:
                print(f"    Warning: Could not read {vtp_file}: {e}")
                continue
        
        if point_count > 0:
            # Update the merged data
            merged_data.Update()
            
            # Write the merged VTK file (XML format)
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(str(output_file))
            writer.SetInputData(merged_data.GetOutput())
            writer.Write()
            
            print(f"    Merged {point_count} points from {len(vtp_files)} files")
        else:
            print(f"    No valid data found to merge for {output_file.name}")
    
    def _merge_vtp_files_simple(self, vtp_files: List[Path], output_file: Path, cluster_sources: List[str]):
        """
        Simple text-based merging of VTP files when VTK library is not available.
        
        Parameters
        ----------
        vtp_files : List[Path]
            List of VTP files to merge
        output_file : Path
            Output file path for the merged VTK
        cluster_sources : List[str]
            List of cluster names corresponding to each VTK file
        """
        all_points = []
        all_cells = []
        all_point_data = []
        header_written = False
        point_offset = 0
        
        for vtp_file, sequence_name in zip(vtp_files, cluster_sources):
            try:
                with open(vtp_file, 'r') as f:
                    lines = f.readlines()
                
                # Parse the VTK file
                in_points = False
                in_cells = False
                in_point_data = False
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    
                    if line.startswith('POINTS'):
                        in_points = True
                        in_cells = False
                        in_point_data = False
                        num_points = int(line.split()[1])
                        continue
                    elif line.startswith('POLYGONS') or line.startswith('LINES') or line.startswith('VERTICES'):
                        in_points = False
                        in_cells = True
                        in_point_data = False
                        num_cells = int(line.split()[1])
                        continue
                    elif line.startswith('POINT_DATA'):
                        in_points = False
                        in_cells = False
                        in_point_data = True
                        continue
                    elif line.startswith('CELL_DATA'):
                        break  # Stop reading at cell data
                    
                    if in_points and line and not line.startswith('#'):
                        # Read point coordinates
                        coords = line.split()
                        if len(coords) >= 3:
                            all_points.extend(coords[:3])
                    
                    elif in_cells and line and not line.startswith('#'):
                        # Read cell definitions and adjust point indices
                        cell_data = line.split()
                        if len(cell_data) > 0:
                            adjusted_cell = [cell_data[0]]  # Keep the first number (number of points in cell)
                            for j in range(1, len(cell_data)):
                                adjusted_index = int(cell_data[j]) + point_offset
                                adjusted_cell.append(str(adjusted_index))
                            all_cells.append(' '.join(adjusted_cell))
                
                point_offset += num_points
                
            except Exception as e:
                print(f"    Warning: Could not parse {vtp_file}: {e}")
                continue
        
        # Write merged VTK file
        if all_points:
            try:
                with open(output_file, 'w') as f:
                    f.write("# vtk DataFile Version 3.0\n")
                    f.write("Merged VTK file from HyFI multi-sequence analysis\n")
                    f.write("ASCII\n")
                    f.write("DATASET POLYDATA\n")
                    
                    # Write points
                    num_total_points = len(all_points) // 3
                    f.write(f"POINTS {num_total_points} float\n")
                    for i in range(0, len(all_points), 3):
                        f.write(f"{all_points[i]} {all_points[i+1]} {all_points[i+2]}\n")
                    
                    # Write cells if any
                    if all_cells:
                        num_total_cells = len(all_cells)
                        total_cell_size = sum(len(cell.split()) for cell in all_cells)
                        f.write(f"VERTICES {num_total_cells} {total_cell_size}\n")
                        for cell in all_cells:
                            f.write(f"{cell}\n")
                
                print(f"    Merged {num_total_points} points using simple text parsing")
                
            except Exception as e:
                print(f"    Error writing merged VTK file: {e}")
        else:
            print(f"    No points found to merge for {output_file.name}")
    
    def _export_vtp_to_kml(self, vtp_file: Path, kml_file: Path, cluster_sources: List[str]):
        """
        Export a VTK file to KML format for Google Earth visualization.
        
        Parameters
        ----------
        vtp_file : Path
            Input VTK file path
        kml_file : Path
            Output KML file path
        cluster_sources : List[str]
            List of cluster names for styling
        """
        try:
            # Parse the VTK file to extract coordinates
            points = []
            cluster_info = []
            
            with open(vtp_file, 'r') as f:
                lines = f.readlines()
            
            in_points = False
            point_count = 0
            expected_points = 0
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('POINTS'):
                    in_points = True
                    expected_points = int(line.split()[1])
                    continue
                elif line.startswith('POLYGONS') or line.startswith('LINES') or line.startswith('VERTICES'):
                    in_points = False
                    break
                
                if in_points and line and not line.startswith('#'):
                    coords = line.split()
                    # Group coordinates by 3 (x, y, z)
                    for i in range(0, len(coords), 3):
                        if i + 2 < len(coords):
                            x, y, z = float(coords[i]), float(coords[i+1]), float(coords[i+2])
                            points.append((x, y, z))
                            # Assign cluster based on point index (simplified)
                            cluster_idx = point_count % len(cluster_sources) if cluster_sources else 0
                            cluster_info.append(cluster_sources[cluster_idx] if cluster_sources else 'unknown')
                            point_count += 1
                            
                            if point_count >= expected_points:
                                break
                    
                    if point_count >= expected_points:
                        break
            
            # Create KML content
            kml_content = self._generate_kml_content(points, cluster_info, vtp_file.stem)
            
            # Write KML file
            with open(kml_file, 'w', encoding='utf-8') as f:
                f.write(kml_content)
            
            print(f"    Exported {len(points)} points to KML: {kml_file.name}")
            
        except Exception as e:
            print(f"    Warning: Could not export {vtp_file.name} to KML: {e}")
    
    def _generate_kml_content(self, points: List[Tuple[float, float, float]], 
                            cluster_info: List[str], title: str) -> str:
        """
        Generate KML content from points and cluster information.
        
        Parameters
        ----------
        points : List[Tuple[float, float, float]]
            List of (x, y, z) coordinates
        cluster_info : List[str]
            List of cluster names for each point
        title : str
            Title for the KML document
            
        Returns
        -------
        str
            KML content as string
        """
        # Get unique clusters for styling
        unique_clusters = list(set(cluster_info)) if cluster_info else ['default']
        
        # Define colors for different clusters (cycling through a palette)
        colors = [
            'ff0000ff',  # Red
            'ff00ff00',  # Green
            'ffff0000',  # Blue
            'ff00ffff',  # Yellow
            'ffff00ff',  # Magenta
            'ffffff00',  # Cyan
            'ff800080',  # Purple
            'ff008000',  # Dark Green
            'ff000080',  # Dark Red
            'ff808000',  # Olive
        ]
        
        cluster_colors = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
        
        kml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            '<Document>',
            f'<name>{title} - HyFI Multi-Sequence Results</name>',
            '<description>Multi-sequence earthquake fault imaging results from HyFI</description>',
            ''
        ]
        
        # Add styles for each cluster
        for cluster, color in cluster_colors.items():
            kml_lines.extend([
                f'<Style id="cluster_{cluster}">',
                '<IconStyle>',
                f'<color>{color}</color>',
                '<scale>0.8</scale>',
                '<Icon>',
                '<href>http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png</href>',
                '</Icon>',
                '</IconStyle>',
                '</Style>',
                ''
            ])
        
        # Add folder for each cluster
        cluster_folders = {cluster: [] for cluster in unique_clusters}
        
        for i, ((x, y, z), cluster) in enumerate(zip(points, cluster_info)):
            # Convert from local coordinates to lat/lon using configured coordinate system
            input_crs = getattr(self.config, 'coordinate_system', 'EPSG:21781')  # Default to Swiss coordinates
            output_crs = 'EPSG:4326'  # WGS84
            transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)
            lon, lat = transformer.transform(x, y)
            depth = -z  # Convert back to positive depth
            
            placemark = [
                '<Placemark>',
                f'<name>Event {i+1}</name>',
                f'<description>Cluster: {cluster}<br/>Depth: {depth:.1f} m<br/>Coordinates: {x:.1f}, {y:.1f}, {z:.1f}</description>',
                f'<styleUrl>#cluster_{cluster}</styleUrl>',
                '<Point>',
                f'<coordinates>{lon:.6f},{lat:.6f},{depth:.1f}</coordinates>',
                '</Point>',
                '</Placemark>',
                ''
            ]
            
            cluster_folders[cluster].extend(placemark)
        
        # Add folders to KML
        for cluster in unique_clusters:
            if cluster_folders[cluster]:
                kml_lines.extend([
                    '<Folder>',
                    f'<name>Cluster {cluster}</name>',
                    f'<description>Events from cluster {cluster}</description>',
                    ''
                ])
                kml_lines.extend(cluster_folders[cluster])
                kml_lines.extend([
                    '</Folder>',
                    ''
                ])
        
        kml_lines.extend([
            '</Document>',
            '</kml>'
        ])
        
        return '\n'.join(kml_lines)
    
    def _export_to_sql_database(self):
        """Export sequence attributes, fault parameters, and stress analysis to SQL database."""
        print("Exporting results to SQL database...")
        
        # Check if SQL database export is enabled in config
        if not hasattr(self.config, 'sequence_workflow_template') or not self.config.sequence_workflow_template:
            print("  SQL database export not configured, skipping...")
            return
        
        workflow_dag = self.config.sequence_workflow_template.get('workflow_dag', {})
        merge_export_config = workflow_dag.get('step_4_merge_and_export', {})
        sql_config = merge_export_config.get('sql_database', {})
        
        if not sql_config.get('enabled', False):
            print("  SQL database export disabled in configuration, skipping...")
            return
        
        try:
            import sqlite3
        except ImportError:
            print("  Warning: sqlite3 not available, skipping SQL export")
            return
        
        # Get database configuration
        db_type = sql_config.get('database_type', 'sqlite')
        db_path = Path(sql_config.get('database_path', './output_SECOS_VS/hyfi_results.db'))
        
        # Ensure the database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create absolute path if relative
        if not db_path.is_absolute():
            db_path = Path(self.config.output_directory) / db_path
        
        print(f"  Database: {db_path}")
        
        # Connect to database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create tables
        self._create_sql_tables(cursor, sql_config)
        
        # Export enriched catalog (all events with clustering and analysis results)
        if 'enriched_catalog' in self.aggregated_results:
            self._export_enriched_catalog_to_sql(cursor, self.aggregated_results['enriched_catalog'])
        
        # Export cluster statistics
        if 'sequence_statistics' in self.aggregated_results:
            self._export_sequence_statistics_to_sql(cursor, self.aggregated_results['sequence_statistics'])
        
        # Export fault planes
        if 'combined_fault_planes' in self.aggregated_results:
            self._export_fault_planes_to_sql(cursor, self.aggregated_results['combined_fault_planes'])
        
        # Export clustering details
        if 'clustering_details' in self.aggregated_results:
            self._export_clustering_details_to_sql(cursor, self.aggregated_results['clustering_details'])
        
        # Commit and close
        conn.commit()
        conn.close()
        
        print(f"  SQL database export completed: {db_path}")
    
    def _create_sql_tables(self, cursor, sql_config):
        """Create SQL tables for storing results."""
        
        # Table for enriched event catalog
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                event_id INTEGER,
                lat REAL,
                lon REAL,
                depth REAL,
                x REAL,
                y REAL,
                z REAL,
                ex REAL,
                ey REAL,
                ez REAL,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                hour INTEGER,
                minute INTEGER,
                second REAL,
                magnitude REAL,
                cluster_label TEXT,
                segmentation_level TEXT,
                fault_plane_azimuth REAL,
                fault_plane_dip REAL,
                fault_plane_strike REAL,
                normal_vector_x REAL,
                normal_vector_y REAL,
                normal_vector_z REAL,
                clustering_quality_r REAL,
                clustering_quality_n REAL,
                clustering_quality_ratio REAL,
                kent_kappa REAL,
                kent_beta REAL,
                nr_fault_fits INTEGER,
                effective_normal_stress REAL,
                shear_stress REAL,
                rake_angle REAL,
                instability_index REAL,
                slip_tendency REAL,
                dilation_tendency REAL,
                analysis_status TEXT,
                fault_network_outlier INTEGER
            )
        ''')
        
        # Table for cluster statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sequence_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_name TEXT UNIQUE,
                n_events INTEGER,
                n_fault_planes INTEGER,
                execution_status TEXT,
                runtime_seconds REAL,
                nodes_executed TEXT
            )
        ''')
        
        # Table for fault planes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fault_planes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_name TEXT,
                event_id INTEGER,
                azimuth REAL,
                dip REAL,
                strike REAL,
                normal_x REAL,
                normal_y REAL,
                normal_z REAL,
                r_value REAL,
                n_value REAL,
                kappa REAL,
                beta REAL
            )
        ''')
        
        # Table for clustering details
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clustering_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_sequences INTEGER,
                total_events_clustered INTEGER,
                final_outliers INTEGER,
                outlier_ratio REAL,
                segmentation_steps TEXT
            )
        ''')
        
        print("  Created SQL tables: events, sequence_statistics, fault_planes, clustering_summary")
    
    def _export_enriched_catalog_to_sql(self, cursor, enriched_catalog):
        """Export enriched catalog to SQL database."""
        print("  Exporting enriched catalog to SQL...")
        
        # Select relevant columns
        columns_to_export = [
            'ID', 'LAT', 'LON', 'DEPTH', 'X', 'Y', 'Z', 'EX', 'EY', 'EZ',
            'YR', 'MO', 'DY', 'HR', 'MI', 'SC', 'MAG',
            'sequence_label', 'segmentation_level',
            'rupture_plane_azimuth', 'rupture_plane_dip', 'rupture_plane_strike',
            'normal_vector_x', 'normal_vector_y', 'normal_vector_z',
            'nvector_quality_R', 'nvector_quality_N', 'nvector_quality_ratio',
            'kent_kappa', 'kent_beta', 'nr_rupt_fits',
            'effective_normal_stress', 'shear_stress', 'rake_angle',
            'instability_index', 'slip_tendency', 'dilation_tendency',
            'analysis_status', 'sequence_outlier'
        ]
        
        # Filter to only columns that exist
        available_columns = [col for col in columns_to_export if col in enriched_catalog.columns]
        
        # Prepare data for insertion
        for _, row in enriched_catalog.iterrows():
            values = []
            for col in available_columns:
                val = row[col]
                # Handle NaN values
                if pd.isna(val):
                    values.append(None)
                else:
                    values.append(val)
            
            # Build SQL insert statement
            placeholders = ','.join(['?' for _ in available_columns])
            sql_columns = ','.join([col.lower().replace('/', '_') for col in available_columns])
            
            cursor.execute(f'''
                INSERT OR REPLACE INTO events ({sql_columns})
                VALUES ({placeholders})
            ''', values)
        
        print(f"    Exported {len(enriched_catalog)} events to SQL database")
    
    def _export_sequence_statistics_to_sql(self, cursor, sequence_statistics):
        """Export cluster statistics to SQL database."""
        print("  Exporting cluster statistics to SQL...")
        
        for sequence_name, stats in sequence_statistics.items():
            cursor.execute('''
                INSERT OR REPLACE INTO sequence_statistics 
                (sequence_name, n_events, n_fault_planes, execution_status, runtime_seconds, nodes_executed)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                sequence_name,
                stats.get('input_events', 0),
                stats.get('n_fault_planes', 0),
                stats.get('status', 'unknown'),
                stats.get('runtime_seconds', 0),
                ','.join(stats.get('nodes_executed', []))
            ))
        
        print(f"    Exported statistics for {len(sequence_statistics)} clusters")
    
    def _export_fault_planes_to_sql(self, cursor, fault_planes):
        """Export fault planes to SQL database."""
        print("  Exporting fault planes to SQL...")
        
        for _, row in fault_planes.iterrows():
            cursor.execute('''
                INSERT INTO fault_planes 
                (sequence_name, event_id, azimuth, dip, strike, normal_x, normal_y, normal_z, 
                 r_value, n_value, kappa, beta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row.get('source_cluster', 'unknown'),
                row.get('ID', None),
                row.get('rupt_plane_azi', None),
                row.get('rupt_plane_dip', None),
                row.get('rupt_plane_azi', None) - 90 if pd.notna(row.get('rupt_plane_azi')) else None,  # Strike
                row.get('nor_x_mean', None),
                row.get('nor_y_mean', None),
                row.get('nor_z_mean', None),
                row.get('R', None),
                row.get('N', None),
                row.get('kappa', None),
                row.get('beta', None)
            ))
        
        print(f"    Exported {len(fault_planes)} fault planes")
    
    def _export_clustering_details_to_sql(self, cursor, clustering_details):
        """Export clustering details to SQL database."""
        print("  Exporting clustering details to SQL...")
        
        # Format step results as JSON string
        import json
        step_results_str = json.dumps(clustering_details.get('step_results', {}))
        
        cursor.execute('''
            INSERT INTO clustering_summary 
            (total_sequences, total_events_clustered, final_outliers, outlier_ratio, segmentation_steps)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            clustering_details.get('total_sequences', 0),
            clustering_details.get('total_events_clustered', 0),
            clustering_details.get('final_outliers', 0),
            clustering_details.get('outlier_ratio', 0.0),
            step_results_str
        ))
        
        print("    Exported clustering summary")
    
    def _create_multi_sequence_visualizations(self):
        """Create visualizations combining results from all sequences."""
        print("Creating multi-sequence visualizations...")
        
        if 'combined_fault_planes' not in self.aggregated_results:
            print("No fault planes to visualize")
            return
        
        try:
            # Create combined visualization parameters
            viz_params = {
                'out_dir': str(self.config.output_directory),
                'project_title': f"{self.config.project_title}_combined"
            }
            
            combined_faults = self.aggregated_results['combined_fault_planes']
            combined_input = self.aggregated_results.get('combined_input_data')
            enriched_catalog = self.aggregated_results.get('enriched_catalog')
            
            # Create traditional 3D visualization of all clusters
            visualisation.model_3d(
                viz_params, 
                combined_input, 
                pd.DataFrame(),  # Empty outliers for now
                combined_faults
            )
            
            # Create new cluster-colored 3D visualization
            if enriched_catalog is not None:
                try:
                    visualisation.model_3d_multi_sequence(
                        viz_params,
                        enriched_catalog,
                        combined_faults
                    )
                except Exception as e:
                    logger.error(f"Error creating cluster-colored 3D visualization: {e}")
                    print(f"Warning: Could not create cluster-colored visualization: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Create cluster-specific visualizations
            self._create_cluster_comparison_plots()
            
        except Exception as e:
            logger.error(f"Error creating multi-sequence visualizations: {e}")
    
    def _create_cluster_comparison_plots(self):
        """Create plots comparing results across clusters."""
        # This could include:
        # - Fault plane orientations by cluster
        # - Cluster distribution in space and time
        # - Statistical comparisons
        # Implementation would go here
        pass
    
    def _save_multi_sequence_results(self):
        """Save all multi-sequence results."""
        print("Saving multi-sequence results...")
        
        output_dir = Path(self.config.output_directory)
        
        # Save aggregated fault planes
        if 'combined_fault_planes' in self.aggregated_results:
            combined_file = output_dir / 'combined_fault_planes.csv'
            self.aggregated_results['combined_fault_planes'].to_csv(combined_file, index=False)
        
        # Save cluster summary
        summary_file = output_dir / 'segmentation_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Multi-Sequence Analysis Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Total sequences processed: {len(self.sequence_results)}\n")
            f.write(f"Successful analyses: {len(self.aggregated_results.get('successful_sequences', []))}\n")
            f.write(f"Total input events: {len(self.full_catalog)}\n")
            
            if 'combined_fault_planes' in self.aggregated_results:
                f.write(f"Total fault planes identified: {len(self.aggregated_results['combined_fault_planes'])}\n")
            
            f.write(f"\nSequence Details:\n")
            for sequence_name, sequence_data in self.sequences.items():
                f.write(f"{sequence_name}: {len(sequence_data)} events\n")
        
        print(f"Multi-sequence results saved to: {output_dir}")
    
    def get_multi_sequence_summary(self) -> Dict[str, Any]:
        """Get a summary of the multi-sequence analysis."""
        if not self.sequence_results:
            return {"status": "No analysis has been run yet"}
        
        summary = {
            "status": "Multi-sequence analysis completed",
            "total_sequences": len(self.sequences),
            "successful_sequences": len([r for r in self.sequence_results.values() if 'workflow_results' in r]),
            "failed_sequences": len([r for r in self.sequence_results.values() if 'error' in r]),
            "total_input_events": len(self.full_catalog) if self.full_catalog is not None else 0,
            "clustering_method": self.config.clustering_method,
            "output_directory": str(self.config.output_directory)
        }
        
        if self.start_time and self.end_time:
            runtime_seconds = self.end_time - self.start_time
            hours, remainder = divmod(int(runtime_seconds), 3600)
            minutes, seconds = divmod(remainder, 60)
            summary["runtime_seconds"] = runtime_seconds
            summary["runtime_formatted"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        if 'combined_fault_planes' in self.aggregated_results:
            summary["total_fault_planes"] = len(self.aggregated_results['combined_fault_planes'])
        
        return summary
