#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main workflow class for hypocenter-based fault imaging.

This module provides a high-level API for running the complete fault imaging
workflow with proper state management and error handling.
"""

import datetime
import time
from typing import Tuple, Optional
import pandas as pd

from ..config.parameters import ProjectConfig
from . import fault_network
from . import model_validation  
from . import auto_class
from . import stress_analysis
from ..visualization import visualisation
from ..utils import utilities


class FaultImagingWorkflow:
    """
    Main workflow class for hypocenter-based 3D fault imaging.
    
    This class manages the complete workflow including:
    - Fault network reconstruction
    - Model validation with focal mechanisms
    - Automatic classification
    - Stress analysis
    - Visualization and data saving
    """
    
    def __init__(self, config: ProjectConfig):
        """
        Initialize the workflow with configuration.
        
        Parameters
        ----------
        config : ProjectConfig
            Configuration object containing all parameters
        """
        self.config = config
        self.config.validate()  # Validate configuration on initialization
        
        # Results storage
        self.results = {}
        self.data_input = None
        self.data_input_outliers = None
        self.data_output = None
        self.df_per_X = None
        self.df_per_Y = None  
        self.df_per_Z = None
        self.S2_trend = None
        self.S2_plunge = None
        
        # Timing
        self.start_time = None
        self.end_time = None
    
    def run_full_analysis(self) -> dict:
        """
        Run the complete fault imaging workflow.
        
        Returns
        -------
        dict
            Dictionary containing all results from the analysis
        """
        print('')
        print('###   HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS   ###')
        print('Calculation started...')
        print('')
        
        self.start_time = time.time()
        
        # Step 1: Fault network reconstruction
        self._run_fault_network_reconstruction()
        
        # Step 2: Model validation (if enabled)
        if self.config.model_validation.validation_bool:
            self._run_model_validation()
        
        # Step 3: Automatic classification (if enabled)
        if self.config.auto_class.autoclass_bool:
            self._run_auto_classification()
        
        # Step 4: Stress analysis (if enabled)
        if self.config.stress_analysis.stress_bool:
            self._run_stress_analysis()
        
        # Step 5: Visualization
        self._run_visualization()
        
        # Step 6: Save results
        self._save_results()
        
        self.end_time = time.time()
        runtime = datetime.timedelta(seconds=(self.end_time - self.start_time))
        
        print('')
        print('')
        print('Calculation done!')
        print('Model runtime: ', str(runtime))
        
        return self.results
    
    def _run_fault_network_reconstruction(self):
        """Run fault network reconstruction module."""
        print("Running fault network reconstruction...")
        
        # Convert config to dictionary for backward compatibility
        input_params = self.config.to_dict()
        
        (self.data_output, self.df_per_X, 
         self.df_per_Y, self.df_per_Z) = fault_network.faultnetwork3D(input_params)
        
        # The function now returns the enriched dataframe as data_output, 
        # we need to extract data_input from it
        self.data_input = self.data_output.copy()
        
        # Extract outliers from data_output using clust_labels == -1
        if 'clust_labels' in self.data_output.columns:
            self.data_input_outliers = self.data_output[self.data_output['clust_labels'] == -1].copy()
        else:
            self.data_input_outliers = pd.DataFrame()  # Empty DataFrame if no outliers
        
        self.results['fault_network'] = {
            'data_input': self.data_input,
            'data_input_outliers': self.data_input_outliers,
            'data_output': self.data_output,
            'df_per_X': self.df_per_X,
            'df_per_Y': self.df_per_Y,
            'df_per_Z': self.df_per_Z
        }
        
        print(f"Fault network reconstruction completed. Found {len(self.data_output)} fault planes.")
    
    def _run_model_validation(self):
        """Run model validation module."""
        print("Running model validation...")
        
        input_params = self.config.to_dict()
        
        self.data_input, self.data_output = model_validation.focal_validation(
            input_params, self.data_input, self.data_output
        )
        
        self.results['model_validation'] = {
            'data_input': self.data_input,
            'data_output': self.data_output
        }
        
        print("Model validation completed.")
    
    def _run_auto_classification(self):
        """Run automatic classification module."""
        print("Running automatic classification...")
        
        input_params = self.config.to_dict()
        
        self.data_output = auto_class.auto_classification(input_params, self.data_output, self.data_input)
        
        self.results['auto_classification'] = {
            'data_output': self.data_output
        }
        
        print("Automatic classification completed.")
    
    def _run_stress_analysis(self):
        """Run stress analysis module."""
        print("Running stress analysis...")
        
        input_params = self.config.to_dict()
        
        self.data_output, self.S2_trend, self.S2_plunge = stress_analysis.fault_stress(
            input_params, self.data_output
        )
        
        self.results['stress_analysis'] = {
            'data_output': self.data_output,
            'S2_trend': self.S2_trend,
            'S2_plunge': self.S2_plunge
        }
        
        print("Stress analysis completed.")
    
    def _run_visualization(self):
        """Run visualization module."""
        print("Creating visualizations...")
        
        input_params = self.config.to_dict()
        
        # Standard 3D visualization
        visualisation.model_3d(input_params, self.data_input, self.data_output)
        
        # Check if interpolation is enabled
        viz_params = input_params.get('visualization', {}).get('parameters', {})
        if viz_params.get('enable_plane_interpolation', False):
            print("Running Poisson surface reconstruction for fault plane interpolation...")
            
            try:
                # Create interpolated fault planes
                # Combine data_input and data_output into a single dataframe for the new function
                df_combined = self.data_output.copy()  # Start with data_output (has fault planes)
                # Add any missing columns from data_input if needed
                for col in ['X', 'Y', 'Z', 'ID', 'ML', 'Date']:
                    if col not in df_combined.columns and col in self.data_input.columns:
                        df_combined[col] = self.data_input[col]
                
                combined_mesh, individual_meshes, point_cloud, fault_disc_meshes = visualisation.create_interpolated_fault_planes(
                    df_combined, viz_params
                )
                
                # Export to VTK if requested
                if viz_params.get('export_vtp', True) and combined_mesh is not None:
                    visualisation.export_interpolated_planes_vtp(
                        combined_mesh, individual_meshes, point_cloud, 
                        input_params.get('out_dir', './output'), fault_disc_meshes, df_combined,
                        use_focal_constraints=input_params.get('use_focal_constraints', False),
                        export_obj=viz_params.get('export_obj', False)
                    )
                
                print("✓ Fault plane interpolation completed successfully")
                
            except Exception as e:
                print(f"Warning: Fault plane interpolation failed: {e}")
                print("Continuing with standard visualization...")
        
        self.results['visualization'] = 'completed'
        
        print("Visualization completed.")
    
    def _save_results(self):
        """Save results to output directory."""
        print("Saving results...")
        
        input_params = self.config.to_dict()
        
        utilities.save_data(input_params, self.data_input, self.data_input_outliers,
                           self.data_output, self.df_per_X, self.df_per_Y, self.df_per_Z)
        
        print(f"Results saved to {self.config.out_dir}")
    
    def run_fault_network_only(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                             pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run only the fault network reconstruction module.
        
        Returns
        -------
        tuple
            (data_input, data_input_outliers, data_output, df_per_X, df_per_Y, df_per_Z)
        """
        self._run_fault_network_reconstruction()
        return (self.data_input, self.data_input_outliers, self.data_output,
                self.df_per_X, self.df_per_Y, self.df_per_Z)
    
    def get_results_summary(self) -> dict:
        """
        Get a summary of the analysis results.
        
        Returns
        -------
        dict
            Summary statistics and key results
        """
        if not self.results:
            return {"status": "No analysis has been run yet"}
        
        summary = {
            "status": "Analysis completed",
            "modules_run": list(self.results.keys()),
            "runtime_seconds": None
        }
        
        if self.start_time and self.end_time:
            summary["runtime_seconds"] = self.end_time - self.start_time
        
        if self.data_input is not None:
            summary["input_events"] = len(self.data_input)
        
        if self.data_output is not None:
            summary["fault_planes"] = len(self.data_output)
        
        if self.data_input_outliers is not None:
            summary["outlier_events"] = len(self.data_input_outliers)
        
        return summary
