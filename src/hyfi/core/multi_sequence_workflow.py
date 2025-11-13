#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-sequence workflow for processing full earthquake catalogs.

This module provides functionality to segment earthquake catalogs into clusters
and process each cluster through the standard HyFI workflow.
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
    and applies the standard HyFI workflow to each identified cluster.
    """
    
    def __init__(self, config: MultiSequenceConfig):
        """
        Initialize the multi-sequence workflow.
        
        Parameters
        ----------
        config : MultiSequenceConfig
            Configuration object containing clustering and workflow parameters
        """
        self.config = config
        self.config.validate()
        
        # Results storage
        self.full_catalog = None
        self.clusters = {}
        self.cluster_results = {}
        self.aggregated_results = {}
        
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
            Dictionary containing results from all clusters and aggregated analysis
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
        
        # Step 3: Process each cluster with single-sequence workflow
        self._process_clusters()
        
        # Step 4: Aggregate results across clusters
        self._aggregate_results()
        
        # Step 5: Create multi-sequence visualizations
        self._create_multi_sequence_visualizations()
        
        # Step 6: Create enriched CSV output
        self._create_enriched_csv_output()
        
        # Step 7: Merge and export combined VTP files
        self._merge_and_export_vtp_files()
        
        # Step 8: Save all results
        self._save_multi_sequence_results()
        
        self.end_time = time.time()
        runtime = datetime.timedelta(seconds=(self.end_time - self.start_time))
        
        print('')
        print('Multi-sequence analysis completed!')
        print(f'Total runtime: {runtime}')
        print(f'Processed {len(self.clusters)} clusters')
        
        return {
            'clusters': self.clusters,
            'cluster_results': self.cluster_results,
            'aggregated_results': self.aggregated_results,
            'metadata': {
                'n_clusters': len(self.clusters),
                'total_events': len(self.full_catalog),
                'runtime_seconds': (self.end_time - self.start_time)
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
        """Segment the catalog into clusters using multi-step clustering."""
        print(f"Segmenting catalog using multi-step clustering...")
        print(f"Number of segmentation steps: {len(self.config.segmentation_steps)}")
        
        # Import the updated clustering function
        from ..utils.clustering import multi_step_catalog_clustering
        
        # Apply multi-step clustering
        self.clusters, self.clustering_results = multi_step_catalog_clustering(
            self.full_catalog,
            self.config.segmentation_steps,
            self.config.clustering.final_outlier_handling,
            self.config.clustering.max_outlier_ratio
        )
        
        print(f"Multi-step clustering completed:")
        print(f"  Total clusters identified: {self.clustering_results['total_clusters']}")
        print(f"  Events clustered: {self.clustering_results['total_events_clustered']}")
        print(f"  Final outliers: {self.clustering_results['final_outliers']} ({self.clustering_results['outlier_ratio']:.2%})")
        
        # Print detailed step results
        for step_name, step_result in self.clustering_results['step_results'].items():
            print(f"  Step '{step_name}': {step_result['clusters_found']} clusters, "
                  f"{step_result['events_clustered']} events clustered, "
                  f"{step_result['outliers']} outliers")
        
        # Store results for later analysis
        self.aggregated_results['clustering_details'] = self.clustering_results
    
    def _process_clusters(self):
        """Process each cluster through the DAG-based workflow."""
        print("Processing individual clusters...")
        
        for cluster_name, cluster_data in self.clusters.items():
            if cluster_name == 'noise':
                continue  # Skip noise cluster
            
            print('\n')
            print("=" * 60)
            print(f"Processing {cluster_name} ({len(cluster_data)} events)...")
            print("=" * 60)
            print('\n')

            try:
                # Create cluster-specific DAG configuration
                cluster_dag_config = self._create_cluster_dag_config(cluster_name, cluster_data)
                
                # Run DAG-based workflow
                from .dag_executor import DAGExecutor
                executor = DAGExecutor(cluster_dag_config, cluster_name=cluster_name)
                cluster_results = executor.execute()
                
                # Store results
                self.cluster_results[cluster_name] = {
                    'workflow_results': cluster_results,
                    'summary': executor.get_execution_summary(),
                    'input_events': len(cluster_data),
                    'config': cluster_dag_config
                }
                
                print(f"Completed {cluster_name}: {executor.get_execution_summary()}")
                
            except Exception as e:
                logger.error(f"Failed to process {cluster_name}: {e}")
                self.cluster_results[cluster_name] = {
                    'error': str(e),
                    'input_events': len(cluster_data)
                }
    
    def _create_cluster_config(self, cluster_name: str, cluster_data: pd.DataFrame) -> ProjectConfig:
        """Create a configuration for processing a single cluster."""
        # Create a temporary file for this cluster
        cluster_dir = Path(self.config.output_directory) / cluster_name
        cluster_dir.mkdir(parents=True, exist_ok=True)
        
        cluster_file = cluster_dir / f"{cluster_name}_data.txt"
        cluster_data.to_csv(cluster_file, sep=self.config.catalog_sep, index=False)
        
        # Clone the template configuration
        cluster_config = ProjectConfig(
            project_title=f"{self.config.project_title} - {cluster_name}",
            hypo_file=cluster_file,
            hypo_sep=self.config.catalog_sep,
            out_dir=cluster_dir
        )
        
        # Copy module configurations from template
        cluster_config.fault_network = self.config.template_config.fault_network
        cluster_config.model_validation = self.config.template_config.model_validation
        cluster_config.auto_class = self.config.template_config.auto_class
        cluster_config.stress_analysis = self.config.template_config.stress_analysis
        
        # Ensure focal mechanism file path is absolute (it should already be resolved 
        # in __post_init__, but double-check for safety)
        if (cluster_config.model_validation.validation_bool and 
            cluster_config.model_validation.foc_file):
            foc_path = Path(cluster_config.model_validation.foc_file)
            if not foc_path.is_absolute():
                # Resolve relative to the original project root, not cluster directory
                original_cwd = Path.cwd()
                cluster_config.model_validation.foc_file = (original_cwd / foc_path).resolve()
        
        return cluster_config
    
    def _create_cluster_dag_config(self, cluster_name: str, cluster_data: pd.DataFrame):
        """Create a DAG configuration for processing a single cluster."""
        from ..config.schema import HyFIWorkflowDAG
        
        # Create a temporary file for this cluster
        cluster_dir = Path(self.config.output_directory) / cluster_name
        cluster_dir.mkdir(parents=True, exist_ok=True)
        
        cluster_file = cluster_dir / f"{cluster_name}_data.csv"
        cluster_data.to_csv(cluster_file, sep=self.config.catalog_sep, index=False)
        
        # Get the cluster workflow template from config
        if hasattr(self.config, 'cluster_workflow_template'):
            template = self.config.cluster_workflow_template
        else:
            # Fallback to template_config if using old format
            template = {
                "metadata": {
                    "workflow_name": f"{self.config.project_title} - {cluster_name}",
                    "workflow_version": "1.0.0"
                },
                "global_settings": {
                    "output_directory": str(cluster_dir)
                },
                "workflow_dag": self._convert_template_config_to_dag()
            }
        
        # Create base DAG configuration dictionary
        dag_dict = {
            "metadata": {
                "workflow_name": f"{template['metadata']['workflow_name']} - {cluster_name}",
                "workflow_version": template['metadata'].get('workflow_version', '1.0.0'),
                "created_date": template['metadata'].get('created_date', '2025-10-03T00:00:00')
            },
            "global_settings": {
                "output_directory": str(cluster_dir)
            },
            "workflow_dag": {}
        }
        
        # Copy template DAG settings
        if 'workflow_dag' in template:
            dag_template = template['workflow_dag']
            
            # Set up input_data node with cluster-specific file
            dag_dict['workflow_dag']['input_data'] = {
                "hypocenter_file": str(cluster_file),
                "hypocenter_separator": self.config.catalog_sep,
                "focal_mechanism_file": "",
                "focal_mechanism_separator": ","
            }
            
            # Copy focal mechanism file from template if specified
            if 'input_data' in dag_template:
                template_input = dag_template['input_data']
                if 'focal_mechanism_file' in template_input and template_input['focal_mechanism_file']:
                    dag_dict['workflow_dag']['input_data']['focal_mechanism_file'] = template_input['focal_mechanism_file']
                if 'focal_mechanism_separator' in template_input:
                    dag_dict['workflow_dag']['input_data']['focal_mechanism_separator'] = template_input['focal_mechanism_separator']
            
            # Copy all other DAG nodes from template
            for node_name, node_config in dag_template.items():
                if node_name != 'input_data':  # input_data is already set above
                    dag_dict['workflow_dag'][node_name] = node_config.copy()
        
        # Create HyFIWorkflowDAG from dictionary
        return HyFIWorkflowDAG.from_dict(dag_dict)
    
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
        """Aggregate results across all clusters."""
        print("Aggregating results across clusters...")
        
        successful_clusters = [name for name, result in self.cluster_results.items() 
                             if 'workflow_results' in result]
        
        if not successful_clusters:
            print("No successful cluster analyses to aggregate")
            return
        
        # Aggregate fault networks
        all_fault_planes = []
        all_input_data = []
        cluster_statistics = {}
        
        for cluster_name in successful_clusters:
            result = self.cluster_results[cluster_name]['workflow_results']
            
            # Check if fault_network results exist and have the expected structure
            if 'fault_network' in result and isinstance(result['fault_network'], dict):
                # Check for data_output (fault planes)
                if 'data_output' in result['fault_network']:
                    fault_data = result['fault_network']['data_output']
                    if fault_data is not None and len(fault_data) > 0:
                        # Add cluster identifier
                        fault_data_copy = fault_data.copy()
                        fault_data_copy['source_cluster'] = cluster_name
                        all_fault_planes.append(fault_data_copy)
                
                # Check for data_input (hypocenter data)
                if 'data_input' in result['fault_network']:
                    input_data = result['fault_network']['data_input']
                    if input_data is not None and len(input_data) > 0:
                        input_data_copy = input_data.copy()
                        input_data_copy['source_cluster'] = cluster_name
                        all_input_data.append(input_data_copy)
            
            # For outlier clusters or clusters without fault network data,
            # try to get input data from visualization results
            elif 'visualization' in result and isinstance(result['visualization'], dict):
                if 'input_data' in result['visualization']:
                    input_data = result['visualization']['input_data']
                    if input_data is not None and len(input_data) > 0:
                        input_data_copy = input_data.copy()
                        input_data_copy['source_cluster'] = cluster_name
                        all_input_data.append(input_data_copy)
            
            # Collect statistics
            summary = self.cluster_results[cluster_name]['summary']
            cluster_statistics[cluster_name] = summary
        
        # Combine results
        if all_fault_planes:
            self.aggregated_results['combined_fault_planes'] = pd.concat(all_fault_planes, ignore_index=True)
            print(f"Aggregated {len(self.aggregated_results['combined_fault_planes'])} fault planes from {len(successful_clusters)} clusters")
        
        if all_input_data:
            self.aggregated_results['combined_input_data'] = pd.concat(all_input_data, ignore_index=True)
        
        self.aggregated_results['cluster_statistics'] = cluster_statistics
        self.aggregated_results['successful_clusters'] = successful_clusters
    
    def _create_enriched_csv_output(self):
        """Create an enriched CSV file combining input data with all analysis results."""
        print("Creating enriched CSV output with analysis results...")
        
        # Start with the original catalog as base
        enriched_data = self.full_catalog.copy()
        
        # Add clustering information
        enriched_data['cluster_label'] = 'unclustered'
        enriched_data['segmentation_level'] = None
        
        # Add cluster labels from the clustering results
        for cluster_name, cluster_events in self.clusters.items():
            if cluster_name == 'noise':
                continue
                
            # Find the cluster events in the main catalog by ID
            cluster_ids = cluster_events['ID'].values
            mask = enriched_data['ID'].isin(cluster_ids)
            enriched_data.loc[mask, 'cluster_label'] = cluster_name
            
            # Extract segmentation level from cluster name (A, B, C, etc.)
            if cluster_name and len(cluster_name) > 0:
                segmentation_level = cluster_name[0]  # First character (A, B, C)
                enriched_data.loc[mask, 'segmentation_level'] = segmentation_level
        
        # Mark outliers as unclustered
        if 'Z_outliers' in self.clusters:
            outlier_ids = self.clusters['Z_outliers']['ID'].values
            outlier_mask = enriched_data['ID'].isin(outlier_ids)
            enriched_data.loc[outlier_mask, 'cluster_label'] = 'unclustered'
            enriched_data.loc[outlier_mask, 'segmentation_level'] = None
        
        # Initialize analysis result columns with NaN
        analysis_columns = {
            # Fault network results
            'fault_plane_azimuth': np.nan,
            'fault_plane_dip': np.nan, 
            'fault_plane_strike': np.nan,
            'normal_vector_x': np.nan,
            'normal_vector_y': np.nan,
            'normal_vector_z': np.nan,
            'clustering_quality_R': np.nan,
            'clustering_quality_N': np.nan,
            'clustering_quality_ratio': np.nan,
            'kent_kappa': np.nan,
            'kent_beta': np.nan,
            'nr_fault_fits': np.nan,
            
            # Stress analysis results
            'effective_normal_stress': np.nan,
            'shear_stress': np.nan,
            'rake_angle': np.nan,
            'instability_index': np.nan,
            'slip_tendency': np.nan,
            'dilation_tendency': np.nan,
            
            # Additional metadata
            'analysis_status': 'not_processed',
            'fault_network_outlier': False
        }
        
        # Add columns to enriched data
        for col_name, default_value in analysis_columns.items():
            enriched_data[col_name] = default_value
        
        # Fill in analysis results by reading HyFI_results.csv files from cluster directories
        output_dir = Path(self.config.output_directory)
        
        for cluster_name, cluster_result in self.cluster_results.items():
            if 'workflow_results' not in cluster_result:
                continue
            
            # Check if this cluster has a HyFI_results.csv file
            cluster_results_file = output_dir / cluster_name / 'HyFI_results.csv'
            if not cluster_results_file.exists():
                # Mark as processed but without detailed results (e.g., Z_outliers)
                cluster_events = self.clusters.get(cluster_name)
                if cluster_events is not None:
                    cluster_ids = cluster_events['ID'].values
                    mask = enriched_data['ID'].isin(cluster_ids)
                    enriched_data.loc[mask, 'analysis_status'] = 'processed_no_results'
                continue
            
            try:
                # Read the cluster's HyFI results
                cluster_hyfi_results = pd.read_csv(cluster_results_file)
                
                # Get cluster event IDs
                cluster_events = self.clusters.get(cluster_name)
                if cluster_events is None:
                    continue
                
                cluster_ids = cluster_events['ID'].values
                mask = enriched_data['ID'].isin(cluster_ids)
                
                # Update analysis status
                enriched_data.loc[mask, 'analysis_status'] = 'processed'
                
                # Map HyFI results to enriched data by matching IDs
                for _, hyfi_row in cluster_hyfi_results.iterrows():
                    event_id = hyfi_row['ID']
                    event_mask = enriched_data['ID'] == event_id
                    
                    if event_mask.any():
                        # Fault plane orientation (using mean values from HyFI results)
                        enriched_data.loc[event_mask, 'fault_plane_azimuth'] = hyfi_row.get('mean_azi', np.nan)
                        enriched_data.loc[event_mask, 'fault_plane_dip'] = hyfi_row.get('mean_dip', np.nan)
                        
                        # Calculate strike from azimuth (azimuth - 90 degrees)
                        if not pd.isna(hyfi_row.get('mean_azi')):
                            strike = (hyfi_row.get('mean_azi') - 90) % 360
                            enriched_data.loc[event_mask, 'fault_plane_strike'] = strike
                        
                        # Normal vector components
                        enriched_data.loc[event_mask, 'normal_vector_x'] = hyfi_row.get('nor_x_mean', np.nan)
                        enriched_data.loc[event_mask, 'normal_vector_y'] = hyfi_row.get('nor_y_mean', np.nan)
                        enriched_data.loc[event_mask, 'normal_vector_z'] = hyfi_row.get('nor_z_mean', np.nan)
                        
                        # Quality metrics
                        enriched_data.loc[event_mask, 'clustering_quality_R'] = hyfi_row.get('R', np.nan)
                        enriched_data.loc[event_mask, 'clustering_quality_N'] = hyfi_row.get('N', np.nan)
                        enriched_data.loc[event_mask, 'clustering_quality_ratio'] = hyfi_row.get('R/N', np.nan)
                        enriched_data.loc[event_mask, 'kent_kappa'] = hyfi_row.get('kappa', np.nan)
                        enriched_data.loc[event_mask, 'kent_beta'] = hyfi_row.get('beta', np.nan)
                        enriched_data.loc[event_mask, 'nr_fault_fits'] = hyfi_row.get('nr_fits', np.nan)
                        
                        # Stress analysis results (if available)
                        enriched_data.loc[event_mask, 'effective_normal_stress'] = hyfi_row.get('Sn_eff', np.nan)
                        enriched_data.loc[event_mask, 'shear_stress'] = hyfi_row.get('Tau', np.nan)
                        enriched_data.loc[event_mask, 'rake_angle'] = hyfi_row.get('rake', np.nan)
                        enriched_data.loc[event_mask, 'instability_index'] = hyfi_row.get('I', np.nan)
                        enriched_data.loc[event_mask, 'slip_tendency'] = hyfi_row.get('sliptend', np.nan)
                        enriched_data.loc[event_mask, 'dilation_tendency'] = hyfi_row.get('dilatend', np.nan)
                
                print(f"  Merged {len(cluster_hyfi_results)} results from {cluster_name}")
                
            except Exception as e:
                print(f"  Warning: Could not read HyFI results for {cluster_name}: {e}")
                # Mark as processed but with error
                cluster_events = self.clusters.get(cluster_name)
                if cluster_events is not None:
                    cluster_ids = cluster_events['ID'].values
                    mask = enriched_data['ID'].isin(cluster_ids)
                    enriched_data.loc[mask, 'analysis_status'] = 'processed_with_error'
        
        # Save enriched CSV
        enriched_file = output_dir / 'enriched_catalog_with_analysis_results.csv'
        enriched_data.to_csv(enriched_file, index=False)
        
        # Store in aggregated results
        self.aggregated_results['enriched_catalog'] = enriched_data
        
        # Print summary
        total_events = len(enriched_data)
        clustered_events = len(enriched_data[enriched_data['cluster_label'] != 'unclustered'])
        processed_events = len(enriched_data[enriched_data['analysis_status'] == 'processed'])
        processed_no_results = len(enriched_data[enriched_data['analysis_status'] == 'processed_no_results'])
        outlier_events = len(enriched_data[enriched_data['fault_network_outlier'] == True])
        
        print(f"Enriched CSV created: {enriched_file}")
        print(f"  Total events: {total_events}")
        print(f"  Clustered events: {clustered_events} ({clustered_events/total_events:.1%})")
        print(f"  Successfully analyzed: {processed_events} ({processed_events/total_events:.1%})")
        if processed_no_results > 0:
            print(f"  Processed without detailed results: {processed_no_results} ({processed_no_results/total_events:.1%})")
        print(f"  Fault network outliers: {outlier_events} ({outlier_events/total_events:.1%})")
        print("")
        print("Available columns in enriched CSV:")
        print("  Original data: ID, LAT, LON, DEPTH, X, Y, Z, EX, EY, EZ, YR, MO, DY, HR, MI, SC, MAG, etc.")
        print("  Clustering: cluster_label, segmentation_level")
        print("  Fault plane orientation: fault_plane_azimuth, fault_plane_dip, fault_plane_strike")
        print("  Normal vectors: normal_vector_x, normal_vector_y, normal_vector_z")
        print("  Quality metrics: clustering_quality_R, clustering_quality_N, clustering_quality_ratio")
        print("  Statistical parameters: kent_kappa, kent_beta, nr_fault_fits")
        print("  Stress analysis: effective_normal_stress, shear_stress, rake_angle, instability_index")
        print("  Fault mechanics: slip_tendency, dilation_tendency")
        print("  Status: analysis_status, fault_network_outlier")
    
    def _merge_and_export_vtp_files(self):
        """Merge VTP files from all clusters and export combined versions."""
        print("Merging and exporting combined VTP files...")
        
        output_dir = Path(self.config.output_directory)
        combined_vtp_dir = output_dir / 'vtp_export_combined'
        combined_vtp_dir.mkdir(exist_ok=True)
        
        # Define the VTP files to merge
        vtp_files_to_merge = {
            'hypocenters_ALL.vtp': 'hypocenters.vtp',
            'interpolated_active_faults_ALL.vtp': 'interpolated_active_faults.vtp',
            'rupture_planes_ALL.vtp': 'rupture_planes.vtp',
            'enhanced_pointcloud_ALL.vtp': 'enhanced_pointcloud.vtp',
            'enhanced_focalplanes_ALL.vtp': 'enhanced_focalplanes.vtp'
        }
        
        for combined_filename, source_filename in vtp_files_to_merge.items():
            try:
                # Collect all VTP files of this type from cluster directories
                vtp_files = []
                cluster_sources = []
                
                for cluster_name in self.cluster_results.keys():
                    if cluster_name == 'noise':
                        continue
                    
                    cluster_vtp_dir = output_dir / cluster_name / 'vtp_export'
                    source_vtp_file = cluster_vtp_dir / source_filename
                    
                    if source_vtp_file.exists():
                        vtp_files.append(source_vtp_file)
                        cluster_sources.append(cluster_name)
                
                # Handle special case for Z_outliers hypocenters
                if source_filename == 'hypocenters.vtp' and 'Z_outliers' in self.cluster_results:
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
            # Check if we have cluster_workflow_template with visualization settings
            if hasattr(self.config, 'cluster_workflow_template') and self.config.cluster_workflow_template:
                viz_config = self.config.cluster_workflow_template.get('workflow_dag', {}).get('visualization', {})
                return viz_config.get('parameters', {}).get('export_kml', True)
            # Fallback to checking individual cluster results for visualization settings
            for cluster_result in self.cluster_results.values():
                if 'config' in cluster_result:
                    dag_config = cluster_result['config']
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
        merged_data = vtk.vtpAppendPolyData()
        merged_data.SetInputData(0, vtk.vtpPolyData())  # Initialize with empty data
        
        point_count = 0
        
        for i, (vtp_file, cluster_name) in enumerate(zip(vtp_files, cluster_sources)):
            try:
                # Read the VTK file
                reader = vtk.vtpPolyDataReader()
                reader.SetFileName(str(vtp_file))
                reader.Update()
                
                polydata = reader.GetOutput()
                
                if polydata.GetNumberOfPoints() > 0:
                    # Add cluster information as point data
                    cluster_array = vtk.vtpStringArray()
                    cluster_array.SetName("cluster_source")
                    cluster_array.SetNumberOfTuples(polydata.GetNumberOfPoints())
                    
                    for j in range(polydata.GetNumberOfPoints()):
                        cluster_array.SetValue(j, cluster_name)
                    
                    polydata.GetPointData().AddArray(cluster_array)
                    
                    # Add cluster ID as numerical array for visualization
                    cluster_id_array = vtk.vtpIntArray()
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
            
            # Write the merged VTK file
            writer = vtk.vtpPolyDataWriter()
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
        
        for vtp_file, cluster_name in zip(vtp_files, cluster_sources):
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
    
    def _create_multi_sequence_visualizations(self):
        """Create visualizations combining results from all clusters."""
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
        summary_file = output_dir / 'cluster_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Multi-Sequence Analysis Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Total clusters processed: {len(self.cluster_results)}\n")
            f.write(f"Successful analyses: {len(self.aggregated_results.get('successful_clusters', []))}\n")
            f.write(f"Total input events: {len(self.full_catalog)}\n")
            
            if 'combined_fault_planes' in self.aggregated_results:
                f.write(f"Total fault planes identified: {len(self.aggregated_results['combined_fault_planes'])}\n")
            
            f.write(f"\nCluster Details:\n")
            for cluster_name, cluster_data in self.clusters.items():
                f.write(f"{cluster_name}: {len(cluster_data)} events\n")
        
        print(f"Multi-sequence results saved to: {output_dir}")
    
    def get_multi_sequence_summary(self) -> Dict[str, Any]:
        """Get a summary of the multi-sequence analysis."""
        if not self.cluster_results:
            return {"status": "No analysis has been run yet"}
        
        summary = {
            "status": "Multi-sequence analysis completed",
            "total_clusters": len(self.clusters),
            "successful_clusters": len([r for r in self.cluster_results.values() if 'workflow_results' in r]),
            "failed_clusters": len([r for r in self.cluster_results.values() if 'error' in r]),
            "total_input_events": len(self.full_catalog) if self.full_catalog is not None else 0,
            "clustering_method": self.config.clustering_method,
            "output_directory": str(self.config.output_directory)
        }
        
        if self.start_time and self.end_time:
            summary["runtime_seconds"] = self.end_time - self.start_time
        
        if 'combined_fault_planes' in self.aggregated_results:
            summary["total_fault_planes"] = len(self.aggregated_results['combined_fault_planes'])
        
        return summary
