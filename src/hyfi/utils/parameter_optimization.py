#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS: Parameter Optimization Module

Automatic optimization of fault network parameters (search radius and search time window)
based on catalog characteristics and focal mechanism validation when available.

Please cite: Truttmann et al. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps.

@author: Sandro Truttmann
@contact: sandro.truttmann@gmail.com
@license: GPL-3.0
@date: September 2025
@version: 0.2.0
"""

import numpy as np
import pandas as pd
import logging
import os
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.optimize import minimize_scalar, differential_evolution
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from scipy.interpolate import griddata


try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import HyFI modules
from ..core import fault_network, model_validation
from . import utilities

logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """
    Automatic optimization of fault network parameters (r_nn and dt_nn).
    
    This class provides methods to automatically determine optimal search radius
    and search time window parameters based on catalog characteristics and 
    focal mechanism validation when available.
    """
    
    def __init__(self, data_input, focal_mechanisms=None, method='grid_search', 
                 custom_r_nn_range=None, custom_dt_nn_range=None, verbose=True,
                 use_adaptive_weights=True, n_matched_focals=None, original_input_params=None):
        """
        Initialize the parameter optimizer.
        
        Parameters
        ----------
        data_input : pd.DataFrame
            Hypocenter catalog with columns: ID, X, Y, Z, Date, etc.
        focal_mechanisms : pd.DataFrame, optional
            Focal mechanism data with Strike1, Dip1, Strike2, Dip2, A (active plane)
        method : str
            Optimization method ('grid_search', 'optuna', 'heuristic')
        custom_r_nn_range : tuple, optional
            Custom search radius range (min_meters, max_meters). If None, uses automatic calculation.
        custom_dt_nn_range : tuple, optional
            Custom time window range (min_hours, max_hours). If None, uses automatic calculation.
        verbose : bool
            Whether to show verbose output during optimization runs (default: True)
        use_adaptive_weights : bool
            Whether to use adaptive weighting based on dataset characteristics (default: True).
            When True, objective function weights adjust based on number of focal mechanisms
            and dataset density. When False, uses fixed weights (original behavior).
        n_matched_focals : int, optional
            Number of focal mechanisms that match hypocenters in this dataset.
            If None, will be calculated internally.
        original_input_params : dict, optional
            Original input_params dictionary to use for optimization (avoids creating temp files).
            If None, will create temporary files.
        """
        self.data_input = data_input.copy()
        self.focal_mechanisms = focal_mechanisms
        self.method = method
        self.custom_r_nn_range = custom_r_nn_range
        self.custom_dt_nn_range = custom_dt_nn_range
        self.verbose = verbose
        self.use_adaptive_weights = use_adaptive_weights
        self.n_matched_focals = n_matched_focals
        self.original_input_params = original_input_params
        self.optimization_results = {}
        
        # Validate input data
        self._validate_input_data()
        
        # Analyze catalog characteristics
        self.catalog_stats = self._analyze_catalog_characteristics()
        
        # Log custom ranges if provided
        if custom_r_nn_range:
            print(f"Custom search radius range: {custom_r_nn_range[0]:.1f} - {custom_r_nn_range[1]:.1f} m")
        if custom_dt_nn_range:
            print(f"Custom time window range: {custom_dt_nn_range[0]:.1f} - {custom_dt_nn_range[1]:.1f} h")
        
        # Log adaptive weighting status
        if use_adaptive_weights:
            n_focals_loaded = len(focal_mechanisms) if focal_mechanisms is not None else 0
            print(f"Using adaptive objective function weights")
            print(f"    Focal mechanisms loaded: {n_focals_loaded}")
            print(f"    Hypocenters in dataset: {len(data_input)}")
            if n_matched_focals is not None:
                print(f"    Focal mechanisms matching hypocenters: {n_matched_focals}")
                if n_matched_focals > 0:
                    print(f"    Adaptive weights will be based on {n_matched_focals} matched focal mechanisms")
            elif n_focals_loaded > 0:
                print(f"    Note: Adaptive weights will be based on focal mechanisms that match hypocenters")
        else:
            print("Using fixed objective function weights")

    def _validate_input_data(self):
        """Validate input data format and completeness."""
        required_columns = ['ID', 'X', 'Y', 'Z', 'Date']
        missing_columns = [col for col in required_columns if col not in self.data_input.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for NaN values in critical columns
        for col in ['X', 'Y', 'Z']:
            if self.data_input[col].isna().any():
                logger.warning(f"NaN values found in column {col}")
        
        # Ensure Date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data_input['Date']):
            try:
                self.data_input['Date'] = pd.to_datetime(self.data_input['Date'])
            except:
                raise ValueError("Cannot convert Date column to datetime format")
    
    def _analyze_catalog_characteristics(self):
        """
        Analyze spatial and temporal characteristics of the earthquake catalog.
        
        Returns
        -------
        dict
            Dictionary containing catalog statistics for parameter optimization
        """
        print("\nAnalyzing catalog characteristics...")
        
        # Spatial analysis
        coordinates = self.data_input[['X', 'Y', 'Z']].values
        
        # Calculate pairwise distances
        distances = pdist(coordinates)
        
        # Calculate nearest neighbor distances for each event
        nn_distances = []
        neigh = NearestNeighbors(n_neighbors=2, algorithm='auto')  # n_neighbors=2 to exclude self
        neigh.fit(coordinates)
        
        for i, coord in enumerate(coordinates):
            distances_i, indices_i = neigh.kneighbors([coord])
            nn_distances.append(distances_i[0][1])  # Second nearest (first is self)
        
        nn_distances = np.array(nn_distances)
        
        # Temporal analysis
        dates = pd.to_datetime(self.data_input['Date']).sort_values()
        time_diffs = np.diff(dates).astype('timedelta64[h]').astype(float)
        
        # Calculate catalog duration and event rate
        catalog_duration_days = (dates.max() - dates.min()).days
        event_rate_per_day = len(self.data_input) / catalog_duration_days if catalog_duration_days > 0 else 0
        
        # Magnitude analysis (if available)
        mag_stats = {}
        for mag_col in ['ML', 'Mw', 'Mag']:
            if mag_col in self.data_input.columns:
                mags = self.data_input[mag_col].dropna()
                if len(mags) > 0:
                    mag_stats[mag_col] = {
                        'min': mags.min(),
                        'max': mags.max(),
                        'mean': mags.mean(),
                        'std': mags.std()
                    }
                    break
        
        stats = {
            'n_events': len(self.data_input),
            'spatial': {
                # 'nn_distances': nn_distances,
                'nn_distance_stats': {
                    'min': np.min(nn_distances),
                    'max': np.max(nn_distances),
                    'mean': np.mean(nn_distances),
                    'median': np.median(nn_distances),
                    'p10': np.percentile(nn_distances, 10),
                    'p25': np.percentile(nn_distances, 25),
                    'p75': np.percentile(nn_distances, 75),
                    'p90': np.percentile(nn_distances, 90),
                    'p95': np.percentile(nn_distances, 95),
                    'p99': np.percentile(nn_distances, 99)

                },
                'distance_range': {
                    'x_range': self.data_input['X'].max() - self.data_input['X'].min(),
                    'y_range': self.data_input['Y'].max() - self.data_input['Y'].min(),
                    'z_range': self.data_input['Z'].max() - self.data_input['Z'].min()
                }
            },
            'temporal': {
                'catalog_duration_days': catalog_duration_days,
                'event_rate_per_day': event_rate_per_day,
                # 'time_differences_hours': time_diffs,
                'time_diff_stats': {
                    'min': np.min(time_diffs) if len(time_diffs) > 0 else 0,
                    'max': np.max(time_diffs) if len(time_diffs) > 0 else 0,
                    'mean': np.mean(time_diffs) if len(time_diffs) > 0 else 0,
                    'median': np.median(time_diffs) if len(time_diffs) > 0 else 0,
                    'p10': np.percentile(time_diffs, 10) if len(time_diffs) > 0 else 0,
                    'p25': np.percentile(time_diffs, 25) if len(time_diffs) > 0 else 0,
                    'p75': np.percentile(time_diffs, 75) if len(time_diffs) > 0 else 0,
                    'p90': np.percentile(time_diffs, 90) if len(time_diffs) > 0 else 0
                }
            },
            'magnitude': mag_stats
        }

        print(f"    - Spatial: median NN distance = {stats['spatial']['nn_distance_stats']['median']:.1f} m")
        print(f"    - Temporal: median time diff = {stats['temporal']['time_diff_stats']['median']:.1f} h")
        print(f"    - X range: {stats['spatial']['distance_range']['x_range']:.1f} m")
        print(f"    - Y range: {stats['spatial']['distance_range']['y_range']:.1f} m")
        print(f"    - Z range: {stats['spatial']['distance_range']['z_range']:.1f} m")
        print(f"    - Duration: {catalog_duration_days:.1f} days")

        return stats
    
    def _define_parameter_ranges(self):
        """
        Define parameter ranges based on custom ranges or catalog characteristics.
        
        Returns
        -------
        tuple
            (r_nn_range, dt_nn_range) tuples with (min, max) values
        """
        # Use custom ranges if provided
        if self.custom_r_nn_range:
            r_nn_min, r_nn_max = self.custom_r_nn_range
            print(f"    Using custom search radius range: {r_nn_min:.1f} - {r_nn_max:.1f} m")
        else:
            # Auto-calculate search radius range based on catalog characteristics
            spatial_stats = self.catalog_stats['spatial']['nn_distance_stats']
            
            # Use percentiles of nearest neighbor distances with some buffer
            r_nn_min = max(spatial_stats['median'], 10)  # Minimum 10m
            r_nn_max = min(spatial_stats['max'] * 10, 5000)  # Maximum 5km
            
            # Ensure reasonable range
            if r_nn_max <= r_nn_min:
                r_nn_min = 50
                r_nn_max = 5000
            
            print(f"Auto-calculated search radius range: {r_nn_min:.1f} - {r_nn_max:.1f} m")
        
        if self.custom_dt_nn_range:
            dt_nn_min, dt_nn_max = self.custom_dt_nn_range
            print(f"    Using custom time window range: {dt_nn_min:.1f} - {dt_nn_max:.1f} h")
        else:
            # Auto-calculate time window range based on catalog characteristics
            temporal_stats = self.catalog_stats['temporal']['time_diff_stats']
            
            # Use temporal clustering characteristics with more reasonable minimums
            # Many earthquake catalogs have daily resolution, so start from 1h minimum
            dt_nn_min = max(temporal_stats['p75'], 1)  # Minimum 1 hour
            dt_nn_max = min(temporal_stats['max'] * 10, 876000)

            # Ensure reasonable range for fault network analysis
            if dt_nn_max <= dt_nn_min:
                dt_nn_min = 1  # 1 hour
                dt_nn_max = 876000  # 10 years
            
            print(f"Auto-calculated time window range: {dt_nn_min:.1f} - {dt_nn_max:.1f} h")
                
        return (r_nn_min, r_nn_max), (dt_nn_min, dt_nn_max)
    
    def _objective_function(self, params, return_details=False):
        """
        Objective function for parameter optimization.
        
        Parameters
        ----------
        params : tuple or list
            (r_nn, dt_nn) parameter values
        return_details : bool
            Whether to return detailed results
            
        Returns
        -------
        float or dict
            Objective value (lower is better) or detailed results
        """
        r_nn, dt_nn = params
        
        try:
            # Instead of creating temporary files, use the original input_params
            # and just override the parameters we're optimizing
            input_params = self.original_input_params.copy()
            input_params['r_nn'] = r_nn
            input_params['dt_nn'] = dt_nn
            input_params['n_mc'] = 1  # Use single MC run for optimization speed
            
            # Run fault network reconstruction with original files
            # Always suppress output during optimization (even if self.verbose=True)
            stdout_capture = StringIO()
            with redirect_stdout(stdout_capture):
                (data_output, per_X, per_Y, per_Z) = fault_network.faultnetwork3D(input_params)
            
            # Calculate basic metrics
            n_events = len(data_output)
            n_planes = len(data_output['rupt_plane_azi'].dropna())
            plane_recovery_rate = n_planes / n_events if n_events > 0 else 0
            
            # Quality metrics
            quality_metrics = self._calculate_quality_metrics(data_output)
            
            # Focal mechanism validation (if available)
            focal_metrics = {}
            if self.focal_mechanisms is not None:
                focal_metrics = self._calculate_focal_validation_metrics(
                    input_params, data_output, data_output)
            
            # Combined objective score - use adaptive or fixed weighting
            if self.use_adaptive_weights:
                objective_score = self._calculate_combined_objective_adaptive(
                    plane_recovery_rate, quality_metrics, focal_metrics, n_events)
            else:
                objective_score = self._calculate_combined_objective(
                    plane_recovery_rate, quality_metrics, focal_metrics)
            
            if return_details:
                return {
                    'objective_score': objective_score,
                    'n_events': n_events,
                    'n_planes': n_planes,
                    'plane_recovery_rate': plane_recovery_rate,
                    'quality_metrics': quality_metrics,
                    'focal_metrics': focal_metrics,
                    'data_output': data_output
                }
            
            return objective_score
            
        except Exception as e:
            logger.warning(f"Error in objective function with params ({r_nn}, {dt_nn}): {e}")
            if return_details:
                return {
                    'objective_score': 1e6,
                    'n_events': 0,
                    'n_planes': 0,
                    'plane_recovery_rate': 0,
                    'quality_metrics': {},
                    'focal_metrics': {},
                    'data_output': None,
                    'error': str(e)
                }
            return 1e6  # Large penalty for failed evaluations
    
    def _calculate_quality_metrics(self, data_output):
        """
        Calculate quality metrics from fault network results.
        
        For single MC runs, uses lambda2/3 ratio from PCA plane fitting as the primary
        quality metric - this measures how well events define a planar structure.
        
        Lambda2/3 ratio interpretation:
        - < 5: Rejected by fault network module (poor planar fit)
        - 5-20: Accepted but basic quality
        - 20-50: Good planar fit quality
        - 50-100+: Excellent to exceptional planar fit quality
        """
        metrics = {}
        
        # Lambda2/3 ratio - primary quality metric for single MC runs
        if 'lambda_2_3' in data_output.columns:
            lambda23_values = data_output['lambda_2_3'].dropna()
            if len(lambda23_values) > 0:
                metrics['mean_lambda23_ratio'] = lambda23_values.mean()
                metrics['median_lambda23_ratio'] = lambda23_values.median()
                # Higher lambda2/3 ratios indicate better planar fits
                # Based on fault network acceptance: >5 is minimum accepted, >20 is good, >50 is excellent
                metrics['accepted_planar_fit_fraction'] = (lambda23_values >= 5.0).sum() / len(lambda23_values)
                metrics['good_planar_fit_fraction'] = (lambda23_values >= 20.0).sum() / len(lambda23_values)
                metrics['excellent_planar_fit_fraction'] = (lambda23_values >= 50.0).sum() / len(lambda23_values)
                metrics['exceptional_planar_fit_fraction'] = (lambda23_values >= 100.0).sum() / len(lambda23_values)
                
        return metrics
    
    def _calculate_focal_validation_metrics(self, input_params, data_input, data_output):
        """Calculate focal mechanism validation metrics."""
        metrics = {}
        
        try:
            # Run focal mechanism validation only if focal mechanisms are available
            if self.focal_mechanisms is not None:
                # Always suppress output during optimization
                stdout_capture = StringIO()
                with redirect_stdout(stdout_capture):
                    data_output_val = model_validation.focal_validation(
                        data_output, input_params)
                
                if 'epsilon' in data_output_val.columns:
                    epsilon_values = data_output_val['epsilon'].dropna()
                    if len(epsilon_values) > 0:
                        metrics['mean_angular_diff'] = epsilon_values.mean()
                        metrics['median_angular_diff'] = epsilon_values.median()
                        metrics['good_fit_fraction'] = (epsilon_values < 30).sum() / len(epsilon_values)
                        metrics['excellent_fit_fraction'] = (epsilon_values < 15).sum() / len(epsilon_values)
                        metrics['n_focal_comparisons'] = len(epsilon_values)
                        metrics['n_compared'] = len(epsilon_values)  # For adaptive weighting
                        
                        # Use active plane information if available
                        if 'A' in self.focal_mechanisms.columns:
                            active_plane_metrics = self._calculate_active_plane_metrics(
                                data_output_val, epsilon_values)
                            metrics.update(active_plane_metrics)
            
        except Exception as e:
            logger.warning(f"Error in focal validation: {e}")
            metrics['mean_angular_diff'] = 90  # Poor fit penalty
            metrics['n_focal_comparisons'] = 0
        
        return metrics
    
    def _calculate_active_plane_metrics(self, data_output, epsilon_values):
        """Calculate metrics specific to active plane selection."""
        metrics = {}
        
        try:
            # Check if focal mechanisms data is available
            if self.focal_mechanisms is None:
                logger.debug("No focal mechanisms data available for active plane metrics")
                return metrics
            
            # Check available columns
            logger.debug(f"Focal mechanisms columns: {list(self.focal_mechanisms.columns)}")
            logger.debug(f"Data output columns: {list(data_output.columns)}")
            
            # Check if preferred focal plane matches active plane designation
            if 'pref_foc' in data_output.columns and 'A' in self.focal_mechanisms.columns:
                # Merge with focal mechanism data to get active plane info
                # Rename 'A' column in focal mechanisms to avoid conflicts during merge
                focal_subset = self.focal_mechanisms[['ID', 'A']].rename(columns={'A': 'active_plane'})
                merged_data = data_output.merge(
                    focal_subset, 
                    on='ID', 
                    how='inner'
                )
                
                logger.debug(f"Merged data shape: {merged_data.shape}")
                
                if len(merged_data) > 0:
                    # ONLY calculate match rate for pre-specified planes (A=1 or A=2)
                    # A=0 events don't have a "ground truth" - they use geometric selection
                    prespecified_mask = merged_data['active_plane'].isin([1, 2])
                    prespecified_events = merged_data[prespecified_mask]
                    
                    if len(prespecified_events) > 0:
                        # Calculate how often pref_foc matches the pre-specified A value
                        active_matches = (prespecified_events['pref_foc'] == prespecified_events['active_plane']).sum()
                        metrics['active_plane_match_rate'] = active_matches / len(prespecified_events)
                        
                        logger.debug(f"Pre-specified plane matches: {active_matches}/{len(prespecified_events)} = {metrics['active_plane_match_rate']:.3f}")
                        logger.debug(f"(Excluded {len(merged_data) - len(prespecified_events)} events with A=0 from match rate)")
                    else:
                        logger.debug("No pre-specified planes (A=1 or A=2) to validate")
                        metrics['active_plane_match_rate'] = 1.0  # Perfect if no pre-specified planes to check
                    
                    # Track newly determined planes (A=0)
                    A0_mask = merged_data['active_plane'] == 0
                    nr_A0_determined = A0_mask.sum()
                    if nr_A0_determined > 0:
                        metrics['newly_determined_planes'] = nr_A0_determined
                        logger.debug(f"Newly determined planes (A=0): {nr_A0_determined}")
                    
                    # Angular differences - use ALL events with pref_foc (including A=0)
                    events_with_pref = merged_data[merged_data['pref_foc'].notna()]
                    if len(events_with_pref) > 0 and 'epsilon' in events_with_pref.columns:
                        metrics['mean_angular_diff_all'] = events_with_pref['epsilon'].mean()
                        logger.debug(f"Mean angular difference (all with pref_foc): {metrics['mean_angular_diff_all']:.2f}°")
                        
                        # Separate stats for pre-specified vs newly determined
                        if len(prespecified_events) > 0 and 'epsilon' in prespecified_events.columns:
                            metrics['active_plane_angular_diff'] = prespecified_events['epsilon'].mean()
                            logger.debug(f"Pre-specified plane angular difference: {metrics['active_plane_angular_diff']:.2f}°")
                        
                        if nr_A0_determined > 0:
                            A0_events = merged_data[A0_mask & merged_data['pref_foc'].notna()]
                            if len(A0_events) > 0 and 'epsilon' in A0_events.columns:
                                metrics['newly_determined_angular_diff'] = A0_events['epsilon'].mean()
                                logger.debug(f"Newly determined (A=0) angular difference: {metrics['newly_determined_angular_diff']:.2f}°")
                else:
                    logger.debug("No matching events found between data_output and focal mechanisms")
            else:
                missing_cols = []
                if 'pref_foc' not in data_output.columns:
                    missing_cols.append("'pref_foc' in data_output")
                if self.focal_mechanisms is not None and 'A' not in self.focal_mechanisms.columns:
                    missing_cols.append("'A' in focal_mechanisms")
                logger.debug(f"Cannot calculate active plane metrics: missing {', '.join(missing_cols)}")
        
        except Exception as e:
            logger.warning(f"Error calculating active plane metrics: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        return metrics
    
    def _calculate_combined_objective(self, plane_recovery_rate, quality_metrics, focal_metrics):
        """
        Calculate combined objective score (lower is better).
        
        The objective function balances (optimized for single MC runs):
        1. Focal mechanism fit (60% weight - lower angular differences are better) - MOST IMPORTANT
        2. Plane recovery rate (25% weight - higher is better) - number of planes found
        3. Quality metrics: lambda2/3 ratio (15% weight - higher is better)
        
        Lambda2/3 ratio scaling:
        - < 5: Below acceptance threshold (fault network module rejects these) → score = 0
        - = 5: Already very good quality → score = 0.7
        - 5-20: Gentle improvement to maximum → score = 0.7 + (ratio-5)/50
        - ≥ 20: Maxed out to prevent domination → score = 1.0
        
        When focal mechanisms are not available, weights are redistributed:
        1. Plane recovery rate (70% weight) - number of planes becomes most important
        2. Quality metrics: lambda2/3 ratio (30% weight)
        """
        score = 0
        
        # Check if focal mechanism data is available
        has_focal_data = (focal_metrics and 'mean_angular_diff' in focal_metrics)
        
        if has_focal_data:
            # Full 3-component objective function with focal mechanisms prioritized
            
            # Focal mechanism component (weight: 0.6) - MOST IMPORTANT
            angular_score = min(focal_metrics['mean_angular_diff'] / 90.0, 1.0)
            score += 0.6 * angular_score

            # Bonus for active plane matching (additional 5% weight)
            if 'active_plane_match_rate' in focal_metrics:
                active_bonus = 1 - focal_metrics['active_plane_match_rate']
                score += 0.05 * active_bonus
            else:
                # No active plane data penalty
                score += 0.05 * 1.0
            
            # Plane recovery component (weight: 0.25) - Second most important
            recovery_score = 1 - plane_recovery_rate  # Convert to minimization
            score += 0.25 * recovery_score
            
            # Quality component based on lambda2/3 ratio (weight: 0.15) - Least important when focals available
            if 'mean_lambda23_ratio' in quality_metrics:
                lambda23_ratio = quality_metrics['mean_lambda23_ratio']
                if lambda23_ratio >= 5.0:  # Accepted ratios
                    # Gentle scaling: ratio 5 = 0.7, ratio 20 = 1.0 (maxed out)
                    # This recognizes that 5 is already very good quality
                    lambda23_normalized = min(0.7 + (lambda23_ratio - 5.0) / 50.0, 1.0)
                else:
                    lambda23_normalized = 0.0  # Below acceptance threshold
                lambda23_score = 1 - lambda23_normalized  # Convert to minimization
                score += 0.15 * lambda23_score
            else:
                # Penalize missing quality metrics
                score += 0.15 * 1.0
                
        else:
            # No focal mechanism data - plane recovery becomes most important
            
            # Plane recovery component (weight: 0.7) - Most important when no focals
            recovery_score = 1 - plane_recovery_rate
            score += 0.7 * recovery_score
            
            # Quality component based on lambda2/3 ratio (weight: 0.3)
            if 'mean_lambda23_ratio' in quality_metrics:
                lambda23_ratio = quality_metrics['mean_lambda23_ratio']
                if lambda23_ratio >= 5.0:  # Accepted ratios
                    # Gentle scaling: ratio 5 = 0.7, ratio 20 = 1.0 (maxed out)
                    # This recognizes that 5 is already very good quality
                    lambda23_normalized = min(0.7 + (lambda23_ratio - 5.0) / 50.0, 1.0)
                else:
                    lambda23_normalized = 0.0  # Below acceptance threshold
                lambda23_score = 1 - lambda23_normalized  # Convert to minimization
                score += 0.3 * lambda23_score
            else:
                # Penalize missing quality metrics
                score += 0.3 * 1.0
        
        return score
    
    def _calculate_adaptive_weights(self, n_focals, n_events):
        """
        Calculate adaptive weights based on data characteristics.
        
        Adjusts objective function weights based on:
        - Number of focal mechanisms available (confidence in validation)
        - Total number of events (dataset density)
        
        Parameters
        ----------
        n_focals : int
            Number of focal mechanisms available
        n_events : int
            Total number of events in catalog
            
        Returns
        -------
        dict
            Dictionary with weights for 'angular', 'active', 'recovery', 'quality'
        """
        if n_focals == 0:
            # No focals: emphasize recovery and quality
            return {
                'angular': 0.0,
                'active': 0.0,
                'recovery': 0.70,
                'quality': 0.30
            }
        
        elif n_focals <= 5:
            # Few focals: reduce confidence in focal fit, increase recovery importance
            # Statistical uncertainty is high with few samples
            return {
                'angular': 0.30,  # Reduced from 0.60
                'active': 0.03,   # Reduced from 0.05
                'recovery': 0.50, # Increased from 0.25
                'quality': 0.20   # Increased from 0.15
            }
        
        elif n_focals <= 20:
            # Moderate focals: balanced approach
            # Transitional regime between low and high confidence
            return {
                'angular': 0.45,
                'active': 0.04,
                'recovery': 0.35,
                'quality': 0.18
            }
        
        else:
            # Many focals (>20): trust focal data more
            # High statistical confidence, use original weights
            return {
                'angular': 0.60,  # Original weights
                'active': 0.05,
                'recovery': 0.25,
                'quality': 0.15
            }
    
    def _calculate_confidence_weighted_focal_score(self, focal_metrics, n_focals):
        """
        Calculate focal score with confidence weighting based on sample size.
        
        For datasets with few focal mechanisms, the mean angular difference
        has high uncertainty. This method blends the observed score with a
        neutral score based on statistical confidence.
        
        Parameters
        ----------
        focal_metrics : dict
            Focal mechanism validation metrics
        n_focals : int
            Number of focal mechanisms
            
        Returns
        -------
        float
            Confidence-weighted angular score (0-1, lower is better)
        """
        angular_diff = focal_metrics.get('mean_angular_diff', 45.0)
        
        # Confidence factor based on sample size
        # Uses sqrt(n) to account for standard error scaling
        # Reaches full confidence at n=20
        confidence = min(np.sqrt(n_focals) / np.sqrt(20), 1.0)
        
        # For low confidence, blend with neutral score (45° = 0.5)
        neutral_score = 45.0 / 90.0
        angular_normalized = min(angular_diff / 90.0, 1.0)
        
        # Weighted blend: high confidence → use observed, low confidence → blend with neutral
        weighted_score = confidence * angular_normalized + (1 - confidence) * neutral_score
        
        return weighted_score
    
    def _calculate_density_adjusted_recovery(self, plane_recovery_rate, n_events):
        """
        Adjust recovery score based on dataset density.
        
        Sparse datasets naturally have lower recovery rates due to data limitations.
        This method normalizes recovery expectations based on catalog size.
        
        Parameters
        ----------
        plane_recovery_rate : float
            Raw plane recovery rate (0-1)
        n_events : int
            Total number of events in catalog
            
        Returns
        -------
        float
            Density-adjusted recovery score (0-1, higher is better)
        """
        if n_events < 100:
            # Sparse datasets: lower expectations
            # E.g., 50% recovery from 50 events might be acceptable
            expected_recovery = 0.5 + 0.3 * (n_events / 100)
        elif n_events < 500:
            # Medium datasets: moderate expectations
            expected_recovery = 0.8
        else:
            # Dense datasets: should achieve high recovery
            expected_recovery = 0.9
        
        # Normalize actual recovery against expectations
        # Cap at 1.0 to avoid rewarding over-clustering
        adjusted = plane_recovery_rate / expected_recovery
        return min(adjusted, 1.0)
    
    def _calculate_combined_objective_adaptive(self, plane_recovery_rate, quality_metrics, 
                                              focal_metrics, n_events):
        """
        Calculate combined objective score with adaptive weighting (lower is better).
        
        This enhanced version adjusts weights and scores based on:
        - Number of focal mechanisms (statistical confidence)
        - Dataset density (recovery expectations)
        - Sample size uncertainty
        
        Parameters
        ----------
        plane_recovery_rate : float
            Fraction of events successfully assigned to planes (0-1)
        quality_metrics : dict
            Quality metrics including mean_lambda23_ratio
        focal_metrics : dict
            Focal mechanism validation metrics
        n_events : int
            Total number of events in catalog
            
        Returns
        -------
        float
            Combined objective score (0-1, lower is better)
        """
        score = 0
        
        # Determine number of focal mechanisms that match hypocenters in this dataset
        n_focals = 0
        has_focal_data = False
        if focal_metrics and 'mean_angular_diff' in focal_metrics:
            has_focal_data = True
            # Use the pre-calculated matched count if available
            if self.n_matched_focals is not None:
                n_focals = self.n_matched_focals
            elif 'n_compared' in focal_metrics:
                # Fallback to n_compared from metrics
                n_focals = focal_metrics['n_compared']
            else:
                # Last resort: use n_focal_comparisons
                n_focals = focal_metrics.get('n_focal_comparisons', 0)
        
        # Get adaptive weights based on data characteristics
        weights = self._calculate_adaptive_weights(n_focals, n_events)
        
        # 1. Focal mechanism component (if available)
        if has_focal_data and weights['angular'] > 0:
            # Use confidence-weighted scoring for focal mechanisms
            angular_score = self._calculate_confidence_weighted_focal_score(focal_metrics, n_focals)
            score += weights['angular'] * angular_score
            
            # Bonus for active plane matching
            if 'active_plane_match_rate' in focal_metrics:
                active_bonus = 1 - focal_metrics['active_plane_match_rate']
                score += weights['active'] * active_bonus
            else:
                # No active plane data penalty
                score += weights['active'] * 1.0
        
        # 2. Plane recovery component with density adjustment
        adjusted_recovery = self._calculate_density_adjusted_recovery(plane_recovery_rate, n_events)
        recovery_score = 1 - adjusted_recovery  # Convert to minimization
        score += weights['recovery'] * recovery_score
        
        # 3. Quality component based on lambda2/3 ratio
        if 'mean_lambda23_ratio' in quality_metrics:
            lambda23_ratio = quality_metrics['mean_lambda23_ratio']
            if lambda23_ratio >= 5.0:  # Accepted ratios
                # Gentle scaling: ratio 5 = 0.7, ratio 20 = 1.0 (maxed out)
                lambda23_normalized = min(0.7 + (lambda23_ratio - 5.0) / 50.0, 1.0)
            else:
                lambda23_normalized = 0.0  # Below acceptance threshold
            lambda23_score = 1 - lambda23_normalized  # Convert to minimization
            score += weights['quality'] * lambda23_score
        else:
            # Penalize missing quality metrics
            score += weights['quality'] * 1.0
        
        return score
    
    def optimize_grid_search(self, n_points=25, plot_results=False, save_plot_path=None, verbose_optimization=False):
        """
        Perform grid search optimization.
        
        Parameters
        ----------
        n_points : int
            Number of points per dimension (total evaluations = n_points^2)
        plot_results : bool
            Whether to generate 2D visualization of results
        save_plot_path : str, optional
            Path to save the plot (if plot_results=True)
        verbose_optimization : bool
            Whether to show verbose output during individual optimization runs (default: False)
            
        Returns
        -------
        dict
            Optimization results
        """
        print(f"    Using {n_points}^2 = {n_points**2} evaluations")
        
        # Temporarily override verbosity setting for optimization runs
        original_verbose = self.verbose
        self.verbose = verbose_optimization
        
        try:
            # Define parameter ranges
            (r_nn_min, r_nn_max), (dt_nn_min, dt_nn_max) = self._define_parameter_ranges()
            
            # Create parameter grids (logarithmic spacing for better coverage)
            r_nn_values = np.logspace(np.log10(r_nn_min), np.log10(r_nn_max), n_points)
            dt_nn_values = np.logspace(np.log10(dt_nn_min), np.log10(dt_nn_max), n_points)
            
            best_score = float('inf')
            best_params = None
            best_details = None
            
            results = []
            
            total_evaluations = len(r_nn_values) * len(dt_nn_values)
            evaluation_count = 0
            
            for i, r_nn in enumerate(r_nn_values):
                for j, dt_nn in enumerate(dt_nn_values):
                    evaluation_count += 1
                    
                    if evaluation_count % 10 == 0:
                        print(f"        Progress: {evaluation_count}/{total_evaluations} evaluations completed")
                    
                    # Evaluate parameter combination
                    details = self._objective_function((r_nn, dt_nn), return_details=True)
                    score = details['objective_score']
                    
                    results.append({
                        'r_nn': r_nn,
                        'dt_nn': dt_nn,
                        'score': score,
                        **details
                    })
                    
                    # Track best result
                    if score < best_score:
                        best_score = score
                        best_params = (r_nn, dt_nn)
                        best_details = details
        
        finally:
            # Restore original verbosity setting
            self.verbose = original_verbose
        
        # Store results
        self.optimization_results = {
            'method': 'grid_search',
            'best_params': {
                'r_nn': best_params[0],
                'dt_nn': best_params[1]
            },
            'best_score': best_score,
            'best_details': best_details,
            'all_results': results,
            'parameter_ranges': {
                'r_nn_range': (r_nn_min, r_nn_max),
                'dt_nn_range': (dt_nn_min, dt_nn_max)
            }
        }

        print(f"        Grid search completed.")

        # Generate plot if requested
        if plot_results:
            try:
                self.plot_grid_search_results(save_path=save_plot_path, show_plot=False)
            except Exception as e:
                logger.warning(f"Failed to generate grid search plot: {e}")
                import traceback
                logger.debug(f"Full plotting error: {traceback.format_exc()}")
        
        return self.optimization_results
    
    def plot_grid_search_results(self, save_path=None, show_plot=True):
        """
        Plot 2D grid search results showing objective function values across parameter space.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the plot. If None, plot is not saved.
        show_plot : bool
            Whether to display the plot
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if not self.optimization_results or self.optimization_results['method'] != 'grid_search':
            raise ValueError("Grid search results not available. Run optimize_grid_search() first.")
        
        results = self.optimization_results['all_results']
        
        # Extract parameter values and scores
        r_nn_values = np.array([r['r_nn'] for r in results])
        dt_nn_values = np.array([r['dt_nn'] for r in results])
        scores = np.array([r['score'] for r in results])
        
        # Get unique parameter values (should be regularly spaced from grid)
        unique_r_nn = np.unique(r_nn_values)
        unique_dt_nn = np.unique(dt_nn_values)
        
        # Create 2D grid for visualization
        score_grid = np.full((len(unique_dt_nn), len(unique_r_nn)), np.nan)
        
        for i, result in enumerate(results):
            r_idx = np.where(unique_r_nn == result['r_nn'])[0][0]
            dt_idx = np.where(unique_dt_nn == result['dt_nn'])[0][0]
            score_grid[dt_idx, r_idx] = result['score']
        
        # Create the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
        fig.suptitle('Grid Search Parameter Optimization Results', fontsize=16, fontweight='bold')
        
        # Create meshgrid for proper coordinate mapping with logarithmic spacing
        R_nn, DT_nn = np.meshgrid(unique_r_nn, unique_dt_nn)
        
        # 1. Main heatmap: Objective scores (lower = better, so blue should be best)
        im1 = ax1.pcolormesh(R_nn, DT_nn, score_grid, cmap='RdYlBu_r', shading='gouraud')
        
        # Add scatter points for all evaluated parameter combinations
        ax1.scatter(r_nn_values, dt_nn_values, 
                   c='black', s=10, marker='o', edgecolor='white', linewidth=0.5, 
                   alpha=0.8, label='Evaluated points')
        
        # Mark best parameter with debugging info
        best_params = self.optimization_results['best_params']
        best_score = self.optimization_results['best_score']
        ax1.scatter(best_params['r_nn'], best_params['dt_nn'], 
                   color='red', s=500, marker='*', edgecolor='white', linewidth=2,
                   label=f"Best: r_nn={best_params['r_nn']:.1f}m, dt_nn={best_params['dt_nn']:.1f}h, score={best_score:.3f}")
        
        # Debug: Add text showing min/max scores to verify colormap
        min_score = np.nanmin(score_grid)
        max_score = np.nanmax(score_grid)
        ax1.text(0.02, 0.98, f'Score range: {min_score:.3f} (best) to {max_score:.3f} (worst)', 
                transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
        
        ax1.set_xlabel('Search Radius (m)', fontweight='bold')
        ax1.set_ylabel('Time Window (h)', fontweight='bold')
        ax1.set_title('Objective Function Values', fontweight='bold')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Objective Score (lower = better)', fontweight='bold')
        
        # 2. Plane recovery rate
        recovery_rates = np.array([r['plane_recovery_rate'] for r in results])
        recovery_grid = np.full((len(unique_dt_nn), len(unique_r_nn)), np.nan)
        
        for i, result in enumerate(results):
            r_idx = np.where(unique_r_nn == result['r_nn'])[0][0]
            dt_idx = np.where(unique_dt_nn == result['dt_nn'])[0][0]
            recovery_grid[dt_idx, r_idx] = result['plane_recovery_rate']
        
        im2 = ax2.pcolormesh(R_nn, DT_nn, recovery_grid, cmap='viridis', shading='gouraud')

        # Add scatter points for all evaluated parameter combinations
        ax2.scatter(r_nn_values, dt_nn_values, 
                   c='black', s=10, marker='o', edgecolor='white', linewidth=0.5, 
                   alpha=0.8, label='Evaluated points')
        
        ax2.scatter(best_params['r_nn'], best_params['dt_nn'], 
                   color='red', s=500, marker='*', edgecolor='white', linewidth=2)
        ax2.set_xlabel('Search Radius (m)', fontweight='bold')
        ax2.set_ylabel('Time Window (h)', fontweight='bold')
        ax2.set_title('Plane Recovery Rate', fontweight='bold')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Recovery Rate', fontweight='bold')
    
        
        # 3. Quality metrics
        lambda23_scores = []
        for result in results:
            if 'quality_metrics' in result and 'mean_lambda23_ratio' in result['quality_metrics']:
                lambda23_scores.append(result['quality_metrics']['mean_lambda23_ratio'])
            else:
                lambda23_scores.append(np.nan)

        lambda23_scores = np.array(lambda23_scores)
        lambda23_grid = np.full((len(unique_dt_nn), len(unique_r_nn)), np.nan)
        for i, result in enumerate(results):
            r_idx = np.where(unique_r_nn == result['r_nn'])[0][0]
            dt_idx = np.where(unique_dt_nn == result['dt_nn'])[0][0]
            lambda23_grid[dt_idx, r_idx] = lambda23_scores[i]

        im3 = ax3.pcolormesh(R_nn, DT_nn, lambda23_grid, cmap='plasma', shading='gouraud')
        # Add scatter points for all evaluated parameter combinations
        ax3.scatter(r_nn_values, dt_nn_values, 
                   c='black', s=10, marker='o', edgecolor='white', linewidth=0.5, 
                   alpha=0.8, label='Evaluated points')
        ax3.scatter(best_params['r_nn'], best_params['dt_nn'], 
                   color='red', s=500, marker='*', edgecolor='white', linewidth=2)
        ax3.set_xlabel('Search Radius (m)', fontweight='bold')
        ax3.set_ylabel('Time Window (h)', fontweight='bold')
        ax3.set_title('Mean Lambda23 Ratio', fontweight='bold')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('Lambda23 Ratio', fontweight='bold')

        # 4. Focal mechanism metrics (if available)
        focal_scores = []
        has_focal_data = False
        
        for result in results:
            if 'focal_metrics' in result and 'mean_angular_diff' in result['focal_metrics']:
                focal_scores.append(result['focal_metrics']['mean_angular_diff'])
                has_focal_data = True
            else:
                focal_scores.append(np.nan)
        
        if has_focal_data:
            focal_scores = np.array(focal_scores)
            focal_grid = np.full((len(unique_dt_nn), len(unique_r_nn)), np.nan)
            
            for i, result in enumerate(results):
                r_idx = np.where(unique_r_nn == result['r_nn'])[0][0]
                dt_idx = np.where(unique_dt_nn == result['dt_nn'])[0][0]
                focal_grid[dt_idx, r_idx] = focal_scores[i]
            
            im4 = ax4.pcolormesh(R_nn, DT_nn, focal_grid, cmap='RdYlGn_r', shading='gouraud')
            
            # Add scatter points for all evaluated parameter combinations
            ax4.scatter(r_nn_values, dt_nn_values, 
                    c='black', s=10, marker='o', edgecolor='white', linewidth=0.5, 
                    alpha=0.8, label='Evaluated points')

            ax4.scatter(best_params['r_nn'], best_params['dt_nn'], 
                       color='red', s=500, marker='*', edgecolor='white', linewidth=2)
            ax4.set_title('Mean Angular Difference (°)', fontweight='bold')

            cbar4 = plt.colorbar(im4, ax=ax4)
            cbar4.set_label('Angular Difference (°)', fontweight='bold')
        else:
            # Show quality metrics instead
            kappa_scores = []
            for result in results:
                if 'quality_metrics' in result and 'mean_kappa' in result['quality_metrics']:
                    kappa_scores.append(result['quality_metrics']['mean_kappa'])
                else:
                    kappa_scores.append(np.nan)
            
            kappa_scores = np.array(kappa_scores)
            kappa_grid = np.full((len(unique_dt_nn), len(unique_r_nn)), np.nan)
            
            for i, result in enumerate(results):
                r_idx = np.where(unique_r_nn == result['r_nn'])[0][0]
                dt_idx = np.where(unique_dt_nn == result['dt_nn'])[0][0]
                kappa_grid[dt_idx, r_idx] = kappa_scores[i]
            
            im4 = ax4.pcolormesh(R_nn, DT_nn, kappa_grid, cmap='viridis', shading='gouraud')
            
            # Add scatter points for all evaluated parameter combinations
            ax4.scatter(r_nn_values, dt_nn_values, 
                    c='black', s=10, marker='o', edgecolor='white', linewidth=0.5, 
                    alpha=0.8, label='Evaluated points')

            ax4.scatter(best_params['r_nn'], best_params['dt_nn'], 
                       color='red', s=500, marker='*', edgecolor='white', linewidth=2)
            ax4.set_title('Mean Kappa Values', fontweight='bold')

            cbar4 = plt.colorbar(im4, ax=ax4)
            cbar4.set_label('Kappa', fontweight='bold')

        ax4.set_xlabel('Search Radius (m)', fontweight='bold')
        ax4.set_ylabel('Time Window (h)', fontweight='bold')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # Add summary text
        best_score = self.optimization_results['best_score']
        best_details = self.optimization_results['best_details']
        
        summary_text = f"""Grid Search Summary:
                        Best Score: {best_score:.4f}
                        Best Parameters: r_nn={best_params['r_nn']:.1f}m, dt_nn={best_params['dt_nn']:.1f}h
                        Fault Planes: {best_details['n_planes']}
                        Recovery Rate: {best_details['plane_recovery_rate']:.3f}"""
        
        fig.text(0.02, 0.02, summary_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save plot if requested
        if save_path and save_path.strip():
            try:
                # Ensure directory exists
                save_dir = os.path.dirname(save_path)
                if save_dir:  # Only create directory if path contains a directory
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            except Exception as e:
                logger.warning(f"Failed to save plot: {e}")
                import traceback
                logger.debug(f"Full error: {traceback.format_exc()}")
        elif save_path is not None:
            logger.warning(f"Invalid save_path provided: '{save_path}'")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def optimize_heuristic(self):
        """
        Fast heuristic parameter estimation based on catalog statistics.
        
        Returns
        -------
        dict
            Optimization results
        """
        print("    Using heuristic parameter estimation")
        
        spatial_stats = self.catalog_stats['spatial']['nn_distance_stats']
        temporal_stats = self.catalog_stats['temporal']['time_diff_stats']
        
        # Heuristic rules based on empirical observations
        # Search radius: use median nearest neighbor distance with scaling factor
        r_nn_heuristic = spatial_stats['p99']
        # r_nn_heuristic = max(min(r_nn_heuristic, 500), 1000)  # Clamp to reasonable range
        
        # Time window: use temporal clustering characteristics
        # For dense catalogs, use shorter time windows; for sparse catalogs, use longer
        event_density = self.catalog_stats['temporal']['event_rate_per_day']
        
        if event_density > 1:  # Dense catalog (>1 event/day)
            dt_nn_heuristic = temporal_stats['p75']  # 75th percentile
        elif event_density > 0.1:  # Moderate catalog
            dt_nn_heuristic = temporal_stats['p90']  # 90th percentile
        else:  # Sparse catalog
            dt_nn_heuristic = min(temporal_stats['max'], 8760 * 3)  # Up to 3 years
        
        # Clamp to reasonable range
        dt_nn_heuristic = max(min(dt_nn_heuristic, 8760 * 5), 24)  # 1 day to 5 years
        
        # Evaluate heuristic parameters
        try:
            details = self._objective_function((r_nn_heuristic, dt_nn_heuristic), return_details=True)
            if isinstance(details, (int, float)):
                # If we got a score instead of details, create a minimal details dict
                details = {
                    'objective_score': details,
                    'n_events': len(self.data_input),
                    'n_planes': 0,
                    'plane_recovery_rate': 0.0,
                    'quality_metrics': {},
                    'focal_metrics': {}
                }
        except Exception as e:
            logger.error(f"Failed to evaluate heuristic parameters: {e}")
            # Return default results if evaluation fails
            details = {
                'objective_score': 1.0,
                'n_events': len(self.data_input),
                'n_planes': 0,
                'plane_recovery_rate': 0.0,
                'quality_metrics': {},
                'focal_metrics': {}
            }
        
        self.optimization_results = {
            'method': 'heuristic',
            'best_params': {
                'r_nn': r_nn_heuristic,
                'dt_nn': dt_nn_heuristic
            },
            'best_score': details['objective_score'],
            'best_details': details,
            'reasoning': {
                'r_nn_reasoning': f"1.5x median NN distance ({spatial_stats['median']:.1f}m)",
                'dt_nn_reasoning': f"Based on event density ({event_density:.3f} events/day)"
            }
        }
        
        print(f"Heuristic estimation: r_nn={r_nn_heuristic:.1f}m, dt_nn={dt_nn_heuristic:.1f}h")
        
        return self.optimization_results

    
    def optimize_optuna(self, n_trials=50, sampler='tpe', n_startup_trials=10,
                       plot_results=False, save_plot_path=None, verbose_optimization=False, 
                       random_state=None, study_name=None, early_stopping_rounds=None,
                       early_stopping_threshold=1e-4):
        """
        Perform optimization using Optuna framework with modern hyperparameter optimization.
        
        Optuna provides state-of-the-art hyperparameter optimization with excellent visualization,
        pruning capabilities, and multiple sampling algorithms.
        
        Parameters
        ----------
        n_trials : int
            Total number of optimization trials (default: 50)
        sampler : str
            Sampling algorithm to use (default: 'tpe')
            Options: 'tpe' (Tree-structured Parzen Estimator - recommended),
                     'cmaes' (CMA-ES),
                     'random' (Random sampling)
        n_startup_trials : int
            Number of random trials before starting sampler-specific optimization (default: 10)
        plot_results : bool
            Whether to generate optimization history and parameter importance plots
        save_plot_path : str, optional
            Path to save the plot (if plot_results=True)
        verbose_optimization : bool
            Whether to show verbose output during individual trials (default: False)
        random_state : int, optional
            Random seed for reproducibility
        study_name : str, optional
            Name for the Optuna study (for tracking/logging)
        early_stopping_rounds : int, optional
            Stop optimization if no improvement for this many consecutive trials (default: None, no early stopping)
            Recommended: 10-20 for n_trials=50, 20-30 for n_trials=100+
        early_stopping_threshold : float, optional
            Minimum improvement to be considered significant (default: 1e-4)
            If best score improves by less than this, it's not counted as improvement
            
        Returns
        -------
        dict
            Optimization results containing best parameters and study object
            
        Raises
        ------
        ImportError
            If Optuna is not installed
        """
        
        if not HAS_OPTUNA:
            raise ImportError(
                "Optuna is required for this optimization method. "
                "Install it with: pip install optuna"
            )
        
        print(f"    Using Optuna optimization with {n_trials} trials")
        print(f"    Sampler: {sampler}")
        print(f"    Startup trials: {n_startup_trials}")
        if early_stopping_rounds:
            print(f"    Early stopping: enabled ({early_stopping_rounds} rounds, threshold={early_stopping_threshold})", flush=True)
        
        # Temporarily override verbosity setting for optimization runs
        original_verbose = self.verbose
        self.verbose = verbose_optimization
        
        # Define parameter ranges
        (r_nn_min, r_nn_max), (dt_nn_min, dt_nn_max) = self._define_parameter_ranges()
        
        # Store evaluation history
        results_history = []
        trial_count = [0]  # Use list to modify in nested function
        
        try:
            # Create sampler based on selection
            if sampler.lower() == 'tpe':
                sampler_obj = TPESampler(n_startup_trials=n_startup_trials, seed=random_state)
            elif sampler.lower() == 'cmaes':
                sampler_obj = CmaEsSampler(n_startup_trials=n_startup_trials, seed=random_state)
            elif sampler.lower() == 'random':
                sampler_obj = RandomSampler(seed=random_state)
            else:
                raise ValueError(f"Unknown sampler: {sampler}. Choose from: 'tpe', 'cmaes', 'random'")
            
            # Create Optuna study
            study = optuna.create_study(
                direction='minimize',
                sampler=sampler_obj,
                study_name=study_name or f'fault_network_optimization_{sampler}',
                load_if_exists=False
            )
            
            # Set logging verbosity
            if not verbose_optimization:
                optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            # Define objective function for Optuna
            def objective(trial):
                # Suggest parameters (log scale for better exploration)
                r_nn = trial.suggest_float('r_nn', r_nn_min, r_nn_max, log=True)
                dt_nn = trial.suggest_float('dt_nn', dt_nn_min, dt_nn_max, log=True)
                
                trial_count[0] += 1
                
                if trial_count[0] % 5 == 0:
                    print(f"        Progress: {trial_count[0]}/{n_trials} trials completed")
                
                # Evaluate parameter combination
                details = self._objective_function((r_nn, dt_nn), return_details=True)
                score = details['objective_score']
                
                # Store result
                results_history.append({
                    'trial_number': trial_count[0],
                    'r_nn': r_nn,
                    'dt_nn': dt_nn,
                    'score': score,
                    **details
                })
                
                # Log intermediate values for Optuna's analysis
                trial.set_user_attr('n_planes', details.get('n_planes', 0))
                trial.set_user_attr('plane_recovery_rate', details.get('plane_recovery_rate', 0))
                
                return score
            
            # Create early stopping callback if requested
            callbacks = []
            
            if early_stopping_rounds is not None and early_stopping_rounds > 0:
                
                class EarlyStoppingCallback:
                    """Custom callback for early stopping based on no improvement."""
                    def __init__(self, patience, threshold, n_startup_trials):
                        self.patience = patience
                        self.threshold = threshold
                        self.n_startup_trials = n_startup_trials
                        self.best_score = None
                        self.no_improvement_count = 0
                        
                    def __call__(self, study, trial):
                        # Skip early stopping during startup phase (random trials)
                        if trial.number < self.n_startup_trials:
                            return
                        
                        # Get current best value
                        if study.best_value is None:
                            return
                        
                        # Initialize on first call after startup phase
                        if self.best_score is None:
                            self.best_score = study.best_value
                            self.no_improvement_count = 0
                            print(f"        ✓ Early stopping activated after startup (trial {trial.number}) - tracking best score: {self.best_score:.6f}", flush=True)
                            return
                        
                        # Calculate improvement based on optimization direction
                        # For minimization: improvement means new best is lower
                        # For maximization: improvement means new best is higher
                        if study.direction == optuna.study.StudyDirection.MINIMIZE:
                            improvement = self.best_score - study.best_value
                        else:
                            improvement = study.best_value - self.best_score
                        
                        # Check if there's significant improvement
                        if improvement > self.threshold:
                            # Significant improvement found
                            old_best = self.best_score
                            self.best_score = study.best_value
                            self.no_improvement_count = 0
                            print(f"        ✓ Trial {trial.number}: IMPROVED from {old_best:.6f} to {study.best_value:.6f} (Δ={improvement:.6f}), counter RESET", flush=True)
                        else:
                            # No significant improvement
                            self.no_improvement_count += 1
                            # Print every 5 trials or when close to stopping
                            if self.no_improvement_count % 5 == 0 or self.no_improvement_count >= self.patience - 2:
                                print(f"        ⚠ Trial {trial.number}: No improvement (Δ={improvement:.6f} <= {self.threshold}), "
                                      f"counter: {self.no_improvement_count}/{self.patience}", flush=True)
                        
                        # Stop if no improvement for patience trials
                        if self.no_improvement_count >= self.patience:
                            print(f"        🛑 EARLY STOPPING TRIGGERED after {self.patience} trials without improvement!", flush=True)
                            print(f"        Best score: {self.best_score:.6f} at trial {study.best_trial.number}", flush=True)
                            study.stop()
                
                callback_instance = EarlyStoppingCallback(early_stopping_rounds, early_stopping_threshold, n_startup_trials)
                callbacks.append(callback_instance)
            else:
                print(f"        ✗ Early stopping DISABLED (rounds={early_stopping_rounds})", flush=True)
            
            # Run optimization with callbacks
            study.optimize(objective, n_trials=n_trials, callbacks=callbacks, show_progress_bar=False)
            
            # Extract best parameters
            best_params = study.best_params
            best_r_nn = best_params['r_nn']
            best_dt_nn = best_params['dt_nn']
            best_score = study.best_value
            
            # Find details for best trial
            best_details = None
            for res in results_history:
                if np.isclose(res['r_nn'], best_r_nn, rtol=1e-3) and \
                   np.isclose(res['dt_nn'], best_dt_nn, rtol=1e-3):
                    best_details = {k: v for k, v in res.items() 
                                   if k not in ['trial_number', 'r_nn', 'dt_nn', 'score']}
                    break
            
            if best_details is None:
                # Fallback: evaluate best parameters if not found in history
                best_details = self._objective_function((best_r_nn, best_dt_nn), return_details=True)
                best_score = best_details['objective_score']
        
        finally:
            # Restore original verbosity setting
            self.verbose = original_verbose
            # Restore Optuna logging
            optuna.logging.set_verbosity(optuna.logging.INFO)
        
        # Store results
        self.optimization_results = {
            'method': 'optuna',
            'best_params': {
                'r_nn': best_r_nn,
                'dt_nn': best_dt_nn
            },
            'best_score': best_score,
            'best_details': best_details,
            'all_results': results_history,
            'optuna_study': study,  # Store full Optuna study for advanced analysis
            'parameter_ranges': {
                'r_nn_range': (r_nn_min, r_nn_max),
                'dt_nn_range': (dt_nn_min, dt_nn_max)
            },
            'optimization_settings': {
                'n_trials': n_trials,
                'sampler': sampler,
                'n_startup_trials': n_startup_trials
            }
        }
        
        print(f"        Optuna optimization completed.")
        print(f"        Best found: r_nn={best_r_nn:.1f}m, dt_nn={best_dt_nn:.1f}h, score={best_score:.4f}")
        print(f"        Best trial number: {study.best_trial.number + 1}/{n_trials}")
        
        # Generate plots if requested
        if plot_results:
            try:
                self.plot_optuna_results(save_path=save_plot_path, show_plot=False)
            except Exception as e:
                logger.warning(f"Failed to generate Optuna optimization plots: {e}")
                import traceback
                logger.debug(f"Full plotting error: {traceback.format_exc()}")
        
        return self.optimization_results
    
    def plot_optuna_results(self, save_path=None, show_plot=True):
        """
        Plot Optuna optimization results with comprehensive diagnostics.
        
        Uses Optuna's built-in visualization plus custom analysis plots.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the plot. If None, plot is not saved.
        show_plot : bool
            Whether to display the plot
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if not self.optimization_results or self.optimization_results['method'] != 'optuna':
            raise ValueError("Optuna optimization results not available. Run optimize_optuna() first.")
                
        study = self.optimization_results['optuna_study']
        results = self.optimization_results['all_results']
        
        # Create figure with subplots: 3 rows x 3 columns
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        best_params = self.optimization_results['best_params']
        best_score = self.optimization_results['best_score']
        n_startup = self.optimization_results['optimization_settings']['n_startup_trials']
        best_trial_number = study.best_trial.number + 1  # +1 for 1-based indexing
        
        # Extract trial data
        trial_numbers = np.array([r['trial_number'] for r in results])
        r_nn_values = np.array([r['r_nn'] for r in results])
        dt_nn_values = np.array([r['dt_nn'] for r in results])
        scores = np.array([r['score'] for r in results])
        
        # ROW 1: Evolution of objective values (left, double width) + Parameter space exploration (right)
        
        # 1. Evolution of objective values (top left, double width)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(trial_numbers, scores, 'o-', alpha=0.6, markersize=4, color='midnightblue', label='Trial value')
        
        # Plot best value evolution
        best_so_far = np.minimum.accumulate(scores)
        ax1.plot(trial_numbers, best_so_far, 'r-', linewidth=2.5, label='Best value', alpha=0.9)
        
        # Mark startup phase and best trial
        ax1.axvline(x=n_startup, color='gray', linestyle='--', alpha=0.6, label='End of random startup')
        ax1.axvline(x=best_trial_number, color='gold', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Best trial #{best_trial_number}')
        ax1.scatter([best_trial_number], [best_score], 
                   c='gold', s=300, marker='*', edgecolors='darkred', linewidth=2, zorder=10)
        
        ax1.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Objective Value (lower is better)', fontsize=11, fontweight='bold')
        ax1.set_title('Evolution of Objective Values', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Parameter space exploration (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        
        # Create interpolation grid for background
        try:
            r_min, r_max = r_nn_values.min(), r_nn_values.max()
            dt_min, dt_max = dt_nn_values.min(), dt_nn_values.max()
            
            # Add some padding to the grid
            r_padding = (r_max - r_min) * 0.05
            dt_padding = (dt_max - dt_min) * 0.05
            
            # Create grid
            grid_r = np.linspace(r_min - r_padding, r_max + r_padding, 100)
            grid_dt = np.linspace(dt_min - dt_padding, dt_max + dt_padding, 100)
            grid_r_mesh, grid_dt_mesh = np.meshgrid(grid_r, grid_dt)
            
            # Interpolate scores onto grid
            points = np.column_stack([r_nn_values, dt_nn_values])
            grid_scores = griddata(points, scores, (grid_r_mesh, grid_dt_mesh), method='cubic')
            
            # Plot interpolated background with pale colors (no lines, only fills)
            contourf = ax2.contourf(grid_r_mesh, grid_dt_mesh, grid_scores, 
                                   levels=20, cmap='RdYlBu_r', alpha=0.3, zorder=1, antialiased=True)
            
        except:
            # If interpolation fails, continue without background
            pass
        
        # Plot trial points on top
        scatter = ax2.scatter(r_nn_values, dt_nn_values, c=scores, s=60, 
                             cmap='RdYlBu_r', alpha=0.7, edgecolors='black', linewidth=0.5, zorder=3)
        ax2.scatter(best_params['r_nn'], best_params['dt_nn'], 
                   c='gold', s=400, marker='*', edgecolor='darkred', linewidth=2.5,
                   label=f'Best: r={best_params["r_nn"]:.0f}m, dt={best_params["dt_nn"]:.0f}h', zorder=10)
        
        # Mark startup trials
        ax2.scatter(r_nn_values[:n_startup], dt_nn_values[:n_startup], 
                   c='gray', s=120, marker='x', linewidth=2.5, label='Startup trials', zorder=5, alpha=0.8)
        
        ax2.set_xlabel('Search Radius (m)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Time Window (hours)', fontsize=11, fontweight='bold')
        ax2.set_title('Parameter Space Exploration', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        plt.colorbar(scatter, ax=ax2, label='Objective Score')
        ax2.grid(True, alpha=0.3)
        
        # ROW 2: r_nn evolution (left) + dt_nn evolution (center) + Summary (right)
        
        # 3. Search radius evolution (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(trial_numbers, r_nn_values, 'o-', alpha=0.6, markersize=4, color='steelblue')
        ax3.axhline(y=best_params['r_nn'], color='red', linestyle='--', linewidth=2, 
                   label=f'Best: {best_params["r_nn"]:.1f}m')
        ax3.axvline(x=n_startup, color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(x=best_trial_number, color='gold', linestyle='--', linewidth=2, alpha=0.8)
        ax3.scatter([best_trial_number], [best_params['r_nn']], 
                   c='gold', s=200, marker='*', edgecolors='darkred', linewidth=2, zorder=10)
        ax3.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Search Radius (m)', fontsize=11, fontweight='bold')
        ax3.set_title('Evolution of r_nn', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. Time window evolution (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(trial_numbers, dt_nn_values, 'o-', alpha=0.6, markersize=4, color='steelblue')
        ax4.axhline(y=best_params['dt_nn'], color='red', linestyle='--', linewidth=2, 
                   label=f'Best: {best_params["dt_nn"]:.1f}h')
        ax4.axvline(x=n_startup, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(x=best_trial_number, color='gold', linestyle='--', linewidth=2, alpha=0.8)
        ax4.scatter([best_trial_number], [best_params['dt_nn']], 
                   c='gold', s=200, marker='*', edgecolors='darkred', linewidth=2, zorder=10)
        ax4.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Time Window (hours)', fontsize=11, fontweight='bold')
        ax4.set_title('Evolution of dt_nn', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. Summary statistics (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        # Calculate improvement statistics
        startup_best = np.min(scores[:n_startup]) if len(scores[:n_startup]) > 0 else best_score
        improvement_pct = ((startup_best - best_score) / startup_best * 100) if startup_best > 0 else 0
        
        sampler_name = self.optimization_results['optimization_settings']['sampler'].upper()
        
        summary_text = f"""
        Optuna Optimization Summary
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        Best Parameters:
          • Search Radius: {best_params['r_nn']:.1f} m
          • Time Window: {best_params['dt_nn']:.1f} hours
        
        Performance:
          • Best Score: {best_score:.4f}
          • Fault Planes: {self.optimization_results['best_details']['n_planes']}
          • Recovery Rate: {self.optimization_results['best_details']['plane_recovery_rate']*100:.1f}%
        
        Optimization:
          • Sampler: {sampler_name}
          • Total Trials: {len(results)}
          • Best Trial: #{study.best_trial.number + 1}
          • Improvement: {improvement_pct:.1f}%
        
        Efficiency:
          • vs Grid (25²): {625/len(results):.1f}x fewer
        """
        
        ax5.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # ROW 3: Objective function components
        # Extract data for objective function components - check both flat structure and nested quality_metrics
        
        # 6. Recovery rate evolution (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        recovery_rates = np.array([r.get('plane_recovery_rate', 0) for r in results])
        ax6.plot(trial_numbers, recovery_rates * 100, 'o-', alpha=0.6, markersize=4, color='mediumseagreen')
        ax6.axhline(y=self.optimization_results['best_details']['plane_recovery_rate'] * 100, 
                   color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Best: {self.optimization_results["best_details"]["plane_recovery_rate"]*100:.1f}%')
        ax6.axvline(x=n_startup, color='gray', linestyle='--', alpha=0.5)
        ax6.axvline(x=best_trial_number, color='gold', linestyle='--', linewidth=2, alpha=0.8)
        ax6.scatter([best_trial_number], [self.optimization_results['best_details']['plane_recovery_rate'] * 100], 
                   c='gold', s=200, marker='*', edgecolors='darkred', linewidth=2, zorder=10)
        ax6.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Recovery Rate (%)', fontsize=11, fontweight='bold')
        ax6.set_title('Plane Recovery Rate', fontsize=12, fontweight='bold')
        ax6.legend(loc='best', fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # 7. Lambda2/3 ratio evolution (bottom center)
        ax7 = fig.add_subplot(gs[2, 1])
        # Try both flat structure and nested quality_metrics
        lambda_ratios = []
        for r in results:
            if 'mean_lambda23_ratio' in r:
                lambda_ratios.append(r['mean_lambda23_ratio'])
            elif 'quality_metrics' in r and 'mean_lambda23_ratio' in r['quality_metrics']:
                lambda_ratios.append(r['quality_metrics']['mean_lambda23_ratio'])
            else:
                lambda_ratios.append(np.nan)
        lambda_ratios = np.array(lambda_ratios)
        
        if not np.all(np.isnan(lambda_ratios)):
            ax7.plot(trial_numbers, lambda_ratios, 'o-', alpha=0.6, markersize=4, color='mediumseagreen')
            # Get best lambda from best_details
            best_lambda = None
            if 'mean_lambda23_ratio' in self.optimization_results['best_details']:
                best_lambda = self.optimization_results['best_details']['mean_lambda23_ratio']
            elif 'quality_metrics' in self.optimization_results['best_details']:
                best_lambda = self.optimization_results['best_details']['quality_metrics'].get('mean_lambda23_ratio', None)
            
            if best_lambda is not None and not np.isnan(best_lambda):
                ax7.axhline(y=best_lambda, color='red', linestyle='--', linewidth=2, alpha=0.7,
                           label=f'Best: {best_lambda:.2f}')
            ax7.axvline(x=n_startup, color='gray', linestyle='--', alpha=0.5)
            ax7.axvline(x=best_trial_number, color='gold', linestyle='--', linewidth=2, alpha=0.8)
            ax7.scatter([best_trial_number], [best_lambda], 
                       c='gold', s=200, marker='*', edgecolors='darkred', linewidth=2, zorder=10)
            ax7.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
            ax7.set_ylabel('Mean λ₂/λ₃ Ratio', fontsize=11, fontweight='bold')
            ax7.set_title('Planarity (λ₂/λ₃)', fontsize=12, fontweight='bold')
            ax7.legend(loc='best', fontsize=9)
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'λ₂/λ₃ ratio not available\n(Check quality_metrics in results)', 
                    ha='center', va='center', fontsize=10, transform=ax7.transAxes)
            ax7.axis('off')
        
        # 8. Focal mechanism mismatch evolution (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        # Try both flat structure and nested focal_metrics
        focal_mismatches = []
        for r in results:
            if 'focal_metrics' in r and 'mean_angular_diff' in r['focal_metrics']:
                focal_mismatches.append(r['focal_metrics']['mean_angular_diff'])
            elif 'mean_angular_diff' in r:
                focal_mismatches.append(r['mean_angular_diff'])
            else:
                focal_mismatches.append(np.nan)
        focal_mismatches = np.array(focal_mismatches)
        
        if not np.all(np.isnan(focal_mismatches)):
            ax8.plot(trial_numbers, focal_mismatches, 'o-', alpha=0.6, markersize=4, color='mediumseagreen')
            # Get best focal from best_details
            best_focal = None
            if 'focal_metrics' in self.optimization_results['best_details']:
                best_focal = self.optimization_results['best_details']['focal_metrics'].get('mean_angular_diff', None)
            elif 'mean_angular_diff' in self.optimization_results['best_details']:
                best_focal = self.optimization_results['best_details']['mean_angular_diff']
            
            if best_focal is not None and not np.isnan(best_focal):
                ax8.axhline(y=best_focal, color='red', linestyle='--', linewidth=2, alpha=0.7,
                           label=f'Best: {best_focal:.1f}°')
            ax8.axvline(x=n_startup, color='gray', linestyle='--', alpha=0.5)
            ax8.axvline(x=best_trial_number, color='gold', linestyle='--', linewidth=2, alpha=0.8)
            ax8.scatter([best_trial_number], [best_focal], 
                       c='gold', s=200, marker='*', edgecolors='darkred', linewidth=2, zorder=10)
            ax8.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
            ax8.set_ylabel('Mean Angular Difference (°)', fontsize=11, fontweight='bold')
            ax8.set_title('Focal Mechanism Mismatch', fontsize=12, fontweight='bold')
            ax8.legend(loc='best', fontsize=9)
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'Focal mechanism data not available', 
                    ha='center', va='center', fontsize=10, transform=ax8.transAxes)
            ax8.axis('off')
        
        plt.suptitle(f'Optuna Optimization Results (Sampler: {sampler_name})', 
                    fontsize=15, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Optuna optimization plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def optimize_pareto(self, n_trials=100, sampler='nsga2', n_startup_trials=20,
                       plot_results=False, save_plot_path=None, verbose_optimization=False, 
                       random_state=None, study_name=None, population_size=50):
        """
        Perform Pareto multi-objective optimization using Optuna.
        
        Instead of combining objectives with fixed weights, this method treats fault network
        optimization as a true multi-objective problem and finds the Pareto front - the set
        of solutions where improving one objective requires degrading another.
        
        This allows you to explore trade-offs between:
        - Focal mechanism fit quality (if available)
        - Plane recovery rate (completeness)
        - Planarity quality (lambda2/3 ratio)
        
        Parameters
        ----------
        n_trials : int
            Total number of optimization trials (default: 100)
            Recommendation: Use 2x more trials than single-objective
        sampler : str
            Multi-objective sampling algorithm (default: 'nsga2')
            Options: 'nsga2' (Non-dominated Sorting Genetic Algorithm II - recommended),
                     'nsga3' (NSGA-III, better for >3 objectives),
                     'random' (Random sampling for baseline)
        n_startup_trials : int
            Number of random trials before starting evolutionary optimization (default: 20)
        population_size : int
            Population size for evolutionary algorithms (default: 50)
        plot_results : bool
            Whether to generate Pareto front visualization
        save_plot_path : str, optional
            Path to save the plot (if plot_results=True)
        verbose_optimization : bool
            Whether to show verbose output during individual trials (default: False)
        random_state : int, optional
            Random seed for reproducibility
        study_name : str, optional
            Name for the Optuna study (for tracking/logging)
            
        Returns
        -------
        dict
            Optimization results containing:
            - 'pareto_front': List of all Pareto-optimal solutions
            - 'best_balanced': Solution with best balance across objectives
            - 'best_focal': Solution with best focal mechanism fit (if available)
            - 'best_recovery': Solution with highest plane recovery
            - 'best_planarity': Solution with best lambda2/3 ratio (planarity)
            - 'optuna_study': Full Optuna study for advanced analysis
            
        Notes
        -----
        The Pareto front provides multiple optimal solutions. You can choose based on priorities:
        - Publications: Select solution with best focal mechanism fit
        - Exploratory analysis: Select solution with highest recovery
        - High-planarity subset: Select solution with best lambda2/3 ratio (planarity)
        
        Raises
        ------
        ImportError
            If Optuna is not installed
        """
        
        print(f"    Using Pareto multi-objective optimization with {n_trials} trials")
        print(f"    Sampler: {sampler}")
        
        # Temporarily override verbosity setting for optimization runs
        original_verbose = self.verbose
        self.verbose = verbose_optimization
        
        # Define parameter ranges
        (r_nn_min, r_nn_max), (dt_nn_min, dt_nn_max) = self._define_parameter_ranges()
        
        # Store evaluation history
        results_history = []
        trial_count = [0]
        
        # Determine number of objectives based on data availability
        has_focal_data = self.focal_mechanisms is not None
        
        try:
            # Create sampler based on selection
            if sampler.lower() == 'nsga2':
                from optuna.samplers import NSGAIISampler
                sampler_obj = NSGAIISampler(
                    population_size=population_size,
                    seed=random_state
                )
            elif sampler.lower() == 'nsga3':
                from optuna.samplers import NSGAIIISampler
                sampler_obj = NSGAIIISampler(
                    population_size=population_size,
                    seed=random_state
                )
            elif sampler.lower() == 'random':
                sampler_obj = RandomSampler(seed=random_state)
            else:
                raise ValueError(f"Unknown sampler: {sampler}. Choose from: 'nsga2', 'nsga3', 'random'")
            
            # Define objective directions (all minimization)
            if has_focal_data:
                directions = ['minimize', 'minimize', 'minimize']  # focal, recovery_loss, quality_loss
                objective_names = ['Focal Mechanism Fit', 'Recovery Loss', 'Quality Loss']
                print(f"    Objectives: Focal mechanism fit, Recovery rate, Lambda2/3 quality")
            else:
                directions = ['minimize', 'minimize']  # recovery_loss, quality_loss
                objective_names = ['Recovery Loss', 'Quality Loss']
                print(f"    Objectives: Recovery rate, Lambda2/3 quality")
            
            # Create multi-objective study
            study = optuna.create_study(
                directions=directions,
                sampler=sampler_obj,
                study_name=study_name or f'fault_network_pareto_{sampler}',
                load_if_exists=False
            )
            
            # Set logging verbosity
            if not verbose_optimization:
                optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            # Define multi-objective function
            def multiobjective(trial):
                # Suggest parameters (log scale for better exploration)
                r_nn = trial.suggest_float('r_nn', r_nn_min, r_nn_max, log=True)
                dt_nn = trial.suggest_float('dt_nn', dt_nn_min, dt_nn_max, log=True)
                
                trial_count[0] += 1
                
                if trial_count[0] % 10 == 0:
                    print(f"        Progress: {trial_count[0]}/{n_trials} trials completed")
                
                # Evaluate parameter combination - get detailed results
                details = self._objective_function((r_nn, dt_nn), return_details=True)
                
                # Extract individual objective components
                plane_recovery_rate = details.get('plane_recovery_rate', 0)
                quality_metrics = details.get('quality_metrics', {})
                focal_metrics = details.get('focal_metrics', {})
                
                # Calculate individual objectives (all for minimization)
                
                # Objective 1: Focal mechanism fit (if available)
                if has_focal_data:
                    if 'mean_angular_diff' in focal_metrics:
                        focal_objective = focal_metrics['mean_angular_diff'] / 90.0  # Normalize to [0,1]
                    else:
                        # If focal mechanism evaluation fails, use maximum penalty (worst fit)
                        focal_objective = 1.0
                
                # Objective 2: Recovery loss (minimize loss = maximize recovery)
                recovery_loss = 1 - plane_recovery_rate
                
                # Objective 3: Quality loss (minimize loss = maximize quality)
                lambda23_ratio = quality_metrics.get('mean_lambda23_ratio', 0)
                if lambda23_ratio >= 5.0:
                    # Normalize quality: 5 -> 0.7, 20 -> 1.0
                    quality_normalized = min(0.7 + (lambda23_ratio - 5.0) / 50.0, 1.0)
                else:
                    quality_normalized = 0.0
                quality_loss = 1 - quality_normalized
                
                # Store detailed results
                result_dict = {
                    'trial_number': trial_count[0],
                    'r_nn': r_nn,
                    'dt_nn': dt_nn,
                    'recovery_loss': recovery_loss,
                    'quality_loss': quality_loss,
                    'plane_recovery_rate': plane_recovery_rate,
                    'n_planes': details.get('n_planes', 0),
                    'mean_lambda23_ratio': lambda23_ratio
                }
                
                if has_focal_data:
                    result_dict['focal_objective'] = focal_objective
                    if 'mean_angular_diff' in focal_metrics:
                        result_dict['mean_angular_diff'] = focal_metrics['mean_angular_diff']
                
                results_history.append(result_dict)
                
                # Set user attributes for analysis
                trial.set_user_attr('n_planes', details.get('n_planes', 0))
                trial.set_user_attr('plane_recovery_rate', plane_recovery_rate)
                trial.set_user_attr('mean_lambda23_ratio', lambda23_ratio)
                if has_focal_data:
                    if 'mean_angular_diff' in focal_metrics:
                        trial.set_user_attr('mean_angular_diff', focal_metrics['mean_angular_diff'])
                
                # Return objectives based on data availability
                # IMPORTANT: Always return the same number of objectives that the study expects
                if has_focal_data:
                    return focal_objective, recovery_loss, quality_loss
                else:
                    return recovery_loss, quality_loss
            
            # Run optimization
            study.optimize(multiobjective, n_trials=n_trials, show_progress_bar=False)
            
            # Extract Pareto front
            pareto_trials = study.best_trials  # All non-dominated solutions
            
            print(f"        Pareto optimization completed.")
            print(f"        Found {len(pareto_trials)} Pareto-optimal solutions")
            
            # Extract Pareto front solutions
            pareto_solutions = []
            for trial in pareto_trials:
                params = trial.params
                
                # Find full details from history
                matching = [r for r in results_history 
                           if np.isclose(r['r_nn'], params['r_nn'], rtol=1e-3) and
                              np.isclose(r['dt_nn'], params['dt_nn'], rtol=1e-3)]
                
                if matching:
                    solution = matching[0].copy()
                    solution['trial_number'] = trial.number + 1
                    solution['objectives'] = trial.values
                    pareto_solutions.append(solution)
            
            # Select representative solutions from Pareto front
            best_solutions = self._select_pareto_representatives(pareto_solutions, has_focal_data)
        
        finally:
            # Restore original verbosity setting
            self.verbose = original_verbose
            optuna.logging.set_verbosity(optuna.logging.INFO)
        
        # Store results
        self.optimization_results = {
            'method': 'pareto',
            'pareto_front': pareto_solutions,
            'best_balanced': best_solutions['balanced'],
            'best_focal': best_solutions.get('focal'),
            'best_recovery': best_solutions['recovery'],
            'best_planarity': best_solutions['quality'],
            'all_results': results_history,
            'optuna_study': study,
            'parameter_ranges': {
                'r_nn_range': (r_nn_min, r_nn_max),
                'dt_nn_range': (dt_nn_min, dt_nn_max)
            },
            'optimization_settings': {
                'n_trials': n_trials,
                'sampler': sampler,
                'n_startup_trials': n_startup_trials,
                'population_size': population_size,
                'has_focal_data': has_focal_data,
                'objective_names': objective_names
            }
        }
        
        # Print summary of representative solutions
        print(f"\n        Representative Pareto Solutions:")
        print(f"        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        for key, name in [('balanced', 'Best Balanced'), ('focal', 'Best Focal Fit'), 
                          ('recovery', 'Best Recovery'), ('quality', 'Best Planarity')]:
            sol = best_solutions.get(key)
            if sol:
                print(f"        {name}:")
                print(f"          r_nn={sol['r_nn']:.1f}m, dt_nn={sol['dt_nn']:.1f}h")
                if has_focal_data and 'mean_angular_diff' in sol:
                    print(f"          Focal fit: {sol['mean_angular_diff']:.1f}°, ", end='')
                print(f"Recovery: {sol['plane_recovery_rate']*100:.1f}%, λ2/3: {sol['mean_lambda23_ratio']:.1f}")
        
        # Generate plots if requested
        if plot_results:
            try:
                self.plot_pareto_results(save_path=save_plot_path, show_plot=False)
            except Exception as e:
                logger.warning(f"Failed to generate Pareto optimization plots: {e}")
                import traceback
                logger.debug(f"Full plotting error: {traceback.format_exc()}")
        
        return self.optimization_results
    
    def _select_pareto_representatives(self, pareto_solutions, has_focal_data):
        """
        Select representative solutions from Pareto front.
        
        Returns solutions that optimize different criteria:
        - Balanced: Best compromise across all objectives
        - Focal: Best focal mechanism fit (if available)
        - Recovery: Highest plane recovery rate
        - Quality: Best lambda2/3 ratio
        """
        if not pareto_solutions:
            return {}
        
        representatives = {}
        
        # Best balanced: Use knee point detection or minimum sum of normalized objectives
        if has_focal_data:
            # Normalize objectives to [0, 1]
            focal_vals = [s['focal_objective'] for s in pareto_solutions if s['focal_objective'] is not None]
            recovery_vals = [s['recovery_loss'] for s in pareto_solutions]
            quality_vals = [s['quality_loss'] for s in pareto_solutions]
            
            if focal_vals:
                focal_min, focal_max = min(focal_vals), max(focal_vals)
                focal_range = focal_max - focal_min if focal_max > focal_min else 1.0
            else:
                focal_min, focal_range = 0, 1.0
            
            recovery_min, recovery_max = min(recovery_vals), max(recovery_vals)
            recovery_range = recovery_max - recovery_min if recovery_max > recovery_min else 1.0
            
            quality_min, quality_max = min(quality_vals), max(quality_vals)
            quality_range = quality_max - quality_min if quality_max > quality_min else 1.0
            
            # Find balanced solution (minimum normalized distance from ideal)
            best_balanced_score = float('inf')
            for sol in pareto_solutions:
                focal_norm = ((sol.get('focal_objective') or 0) - focal_min) / focal_range if focal_vals else 0
                recovery_norm = (sol['recovery_loss'] - recovery_min) / recovery_range
                quality_norm = (sol['quality_loss'] - quality_min) / quality_range
                
                # Euclidean distance from ideal point (0, 0, 0)
                score = np.sqrt(focal_norm**2 + recovery_norm**2 + quality_norm**2)
                
                if score < best_balanced_score:
                    best_balanced_score = score
                    representatives['balanced'] = sol
            
            # Best focal fit
            representatives['focal'] = min(
                [s for s in pareto_solutions if s.get('focal_objective') is not None],
                key=lambda x: x['focal_objective']
            )
        else:
            # Without focal data, use 2D Pareto front
            recovery_vals = [s['recovery_loss'] for s in pareto_solutions]
            quality_vals = [s['quality_loss'] for s in pareto_solutions]
            
            recovery_min, recovery_max = min(recovery_vals), max(recovery_vals)
            recovery_range = recovery_max - recovery_min if recovery_max > recovery_min else 1.0
            
            quality_min, quality_max = min(quality_vals), max(quality_vals)
            quality_range = quality_max - quality_min if quality_max > quality_min else 1.0
            
            # Balanced solution
            best_balanced_score = float('inf')
            for sol in pareto_solutions:
                recovery_norm = (sol['recovery_loss'] - recovery_min) / recovery_range
                quality_norm = (sol['quality_loss'] - quality_min) / quality_range
                
                score = np.sqrt(recovery_norm**2 + quality_norm**2)
                
                if score < best_balanced_score:
                    best_balanced_score = score
                    representatives['balanced'] = sol
        
        # Best recovery
        representatives['recovery'] = min(pareto_solutions, key=lambda x: x['recovery_loss'])
        
        # Best quality
        representatives['quality'] = min(pareto_solutions, key=lambda x: x['quality_loss'])
        
        return representatives
    
    def plot_pareto_results(self, save_path=None, show_plot=True):
        """
        Plot Pareto multi-objective optimization results.
        
        Visualizes the Pareto front showing trade-offs between objectives.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the plot. If None, plot is not saved.
        show_plot : bool
            Whether to display the plot
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if not self.optimization_results or self.optimization_results['method'] != 'pareto':
            raise ValueError("Pareto optimization results not available. Run optimize_pareto() first.")
        
        pareto_front = self.optimization_results['pareto_front']
        all_results = self.optimization_results['all_results']
        settings = self.optimization_results['optimization_settings']
        has_focal_data = settings['has_focal_data']
        
        # Determine figure layout based on number of objectives
        if has_focal_data:
            # 3 objectives: portrait format with 3D plot on top
            fig = plt.figure(figsize=(12, 16))
            # Reduced spacing for better page usage
            gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.30, height_ratios=[1.2, 1, 1],
                                 top=0.98, bottom=0.02, left=0.05, right=0.95)
        else:
            # 2 objectives: simpler visualization
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.25,
                                 top=0.98, bottom=0.05, left=0.05, right=0.95)
        
        # Extract data for plotting
        all_r_nn = [r['r_nn'] for r in all_results]
        all_dt_nn = [r['dt_nn'] for r in all_results]
        all_recovery = [r['plane_recovery_rate'] for r in all_results]
        all_lambda23 = [r['mean_lambda23_ratio'] for r in all_results]
        
        pareto_r_nn = [s['r_nn'] for s in pareto_front]
        pareto_dt_nn = [s['dt_nn'] for s in pareto_front]
        pareto_recovery = [s['plane_recovery_rate'] for s in pareto_front]
        pareto_lambda23 = [s['mean_lambda23_ratio'] for s in pareto_front]
        
        # Representative solutions
        best_balanced = self.optimization_results['best_balanced']
        best_recovery = self.optimization_results['best_recovery']
        best_planarity = self.optimization_results['best_planarity']
        best_focal = self.optimization_results.get('best_focal')
        
        if has_focal_data:
            all_angular = [r.get('mean_angular_diff', 90) for r in all_results]
            pareto_angular = [s.get('mean_angular_diff', 90) for s in pareto_front]
            
            # 1. 3D Pareto Front (top row, spanning all columns) - LARGE
            ax1 = fig.add_subplot(gs[0, :], projection='3d')
            
            # Plot all trials
            ax1.scatter(all_angular, [r*100 for r in all_recovery], all_lambda23,
                       c='lightgray', s=20, alpha=0.3, label='All trials')
            
            # Plot Pareto front
            ax1.scatter(pareto_angular, [r*100 for r in pareto_recovery], pareto_lambda23,
                       c='blue', s=100, alpha=0.7, edgecolors='black', linewidth=1, 
                       label='Pareto front')
            
            # Highlight representatives
            if best_focal:
                ax1.scatter([best_focal.get('mean_angular_diff', 90)], 
                           [best_focal['plane_recovery_rate']*100],
                           [best_focal['mean_lambda23_ratio']],
                           c='red', s=300, marker='*', edgecolors='darkred', linewidth=2,
                           label='Best focal fit', zorder=10)
            
            ax1.scatter([best_recovery.get('mean_angular_diff', 90)], 
                       [best_recovery['plane_recovery_rate']*100],
                       [best_recovery['mean_lambda23_ratio']],
                       c='green', s=300, marker='D', edgecolors='darkgreen', linewidth=2,
                       label='Best recovery', zorder=10)
            
            ax1.scatter([best_planarity.get('mean_angular_diff', 90)], 
                       [best_planarity['plane_recovery_rate']*100],
                       [best_planarity['mean_lambda23_ratio']],
                       c='purple', s=300, marker='^', edgecolors='indigo', linewidth=2,
                       label='Best planarity', zorder=10)
            
            ax1.scatter([best_balanced.get('mean_angular_diff', 90)], 
                       [best_balanced['plane_recovery_rate']*100],
                       [best_balanced['mean_lambda23_ratio']],
                       c='orange', s=350, marker='s', edgecolors='darkorange', linewidth=2,
                       label='Best balanced', zorder=10)
            
            ax1.set_xlabel('Focal Fit (°)\n(lower better)', fontsize=11, labelpad=10)
            ax1.set_ylabel('Recovery (%)\n(higher better)', fontsize=11, labelpad=10)
            ax1.set_zlabel('λ2/3 Ratio\n(higher better)', fontsize=11, labelpad=10)
            ax1.set_title('3D Pareto Front: Multi-Objective Trade-offs', fontsize=14, fontweight='bold', pad=15)
            # Place legend to the right of the 3D plot
            ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=9, framealpha=0.9)
            ax1.grid(True, alpha=0.3)
            
            # 2. SECOND ROW: Three 2D trade-off plots
            
            # Focal vs Recovery trade-off (row 1, col 0)
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.scatter(all_angular, [r*100 for r in all_recovery], c='lightgray', s=20, alpha=0.3)
            ax2.scatter(pareto_angular, [r*100 for r in pareto_recovery], 
                       c='blue', s=80, alpha=0.7, edgecolors='black', linewidth=1)
            
            # Highlight representatives
            for sol, color, marker, label in [
                (best_focal, 'red', '*', 'Best focal'),
                (best_recovery, 'green', 'D', 'Best recovery'),
                (best_balanced, 'orange', 's', 'Best balanced')
            ]:
                if sol:
                    ax2.scatter([sol.get('mean_angular_diff', 90)], 
                               [sol['plane_recovery_rate']*100],
                               c=color, s=200, marker=marker, edgecolors='black', 
                               linewidth=1.5, label=label, zorder=10)
            
            ax2.set_xlabel('Focal Mechanism Fit (°)', fontsize=10)
            ax2.set_ylabel('Plane Recovery Rate (%)', fontsize=10)
            ax2.set_title('Focal Fit vs Recovery Trade-off', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Focal vs Quality trade-off (row 1, col 1)
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.scatter(all_angular, all_lambda23, c='lightgray', s=20, alpha=0.3)
            ax3.scatter(pareto_angular, pareto_lambda23,
                       c='blue', s=80, alpha=0.7, edgecolors='black', linewidth=1)
            
            for sol, color, marker in [(best_focal, 'red', '*'), (best_planarity, 'purple', '^'),
                                       (best_balanced, 'orange', 's')]:
                if sol:
                    ax3.scatter([sol.get('mean_angular_diff', 90)], [sol['mean_lambda23_ratio']],
                               c=color, s=200, marker=marker, edgecolors='black', 
                               linewidth=1.5, zorder=10)
            
            ax3.set_xlabel('Focal Mechanism Fit (°)', fontsize=10)
            ax3.set_ylabel('Lambda2/3 Ratio', fontsize=10)
            ax3.set_title('Focal Fit vs Quality Trade-off', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Recovery vs Quality (row 1, col 2)
            ax4 = fig.add_subplot(gs[1, 2])
            ax4.scatter([r*100 for r in all_recovery], all_lambda23,
                       c='lightgray', s=20, alpha=0.3)
            ax4.scatter([r*100 for r in pareto_recovery], pareto_lambda23,
                       c='blue', s=80, alpha=0.7, edgecolors='black', linewidth=1)
            
            for sol, color, marker in [(best_recovery, 'green', 'D'), (best_planarity, 'purple', '^'),
                                       (best_balanced, 'orange', 's')]:
                ax4.scatter([sol['plane_recovery_rate']*100], [sol['mean_lambda23_ratio']],
                           c=color, s=200, marker=marker, edgecolors='black', 
                           linewidth=1.5, zorder=10)
            
            ax4.set_xlabel('Plane Recovery Rate (%)', fontsize=10)
            ax4.set_ylabel('Lambda2/3 Ratio', fontsize=10)
            ax4.set_title('Recovery vs Quality Trade-off', fontsize=11, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # THIRD ROW: Three additional plots
            
            # Parameter space with Pareto front (row 2, col 0)
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.scatter(all_r_nn, all_dt_nn, c='lightgray', s=20, alpha=0.3, label='All trials')
            ax5.scatter(pareto_r_nn, pareto_dt_nn, c='blue', s=80, alpha=0.7, 
                       edgecolors='black', linewidth=1, label='Pareto front')
            
            # Highlight representatives
            for sol, color, marker, label in [
                (best_balanced, 'orange', 's', 'Balanced'),
                (best_recovery, 'green', 'D', 'Recovery'),
                (best_planarity, 'purple', '^', 'Planarity')
            ]:
                ax5.scatter([sol['r_nn']], [sol['dt_nn']], c=color, s=200, marker=marker,
                           edgecolors='black', linewidth=1.5, label=label, zorder=10)
            
            if best_focal:
                ax5.scatter([best_focal['r_nn']], [best_focal['dt_nn']], c='red', s=200, 
                           marker='*', edgecolors='black', linewidth=1.5, label='Focal', zorder=10)
            
            ax5.set_xlabel('Search Radius (m)', fontsize=10)
            ax5.set_ylabel('Time Window (hours)', fontsize=10)
            ax5.set_title('Parameter Space: Pareto Solutions', fontsize=11, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            # Representative Solutions Summary (row 2, col 1)
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.axis('off')
            
            # Build summary text with emphasis on balanced
            summary_lines = ["Representative Solutions", "=" * 28, ""]
            
            # Best Balanced (emphasized)
            summary_lines.append("★ BEST BALANCED ★")
            summary_lines.append(f"  r_nn: {best_balanced['r_nn']:.1f} m")
            summary_lines.append(f"  dt_nn: {best_balanced['dt_nn']:.1f} h")
            summary_lines.append(f"  Recovery: {best_balanced['plane_recovery_rate']*100:.1f}%")
            if 'mean_angular_diff' in best_balanced:
                summary_lines.append(f"  Focal: {best_balanced['mean_angular_diff']:.1f}°")
            summary_lines.append(f"  λ2/3: {best_balanced['mean_lambda23_ratio']:.1f}")
            summary_lines.append("")
            
            # Best Recovery
            summary_lines.append("Best Recovery:")
            summary_lines.append(f"  r_nn: {best_recovery['r_nn']:.1f} m")
            summary_lines.append(f"  dt_nn: {best_recovery['dt_nn']:.1f} h")
            summary_lines.append(f"  Recovery: {best_recovery['plane_recovery_rate']*100:.1f}%")
            summary_lines.append("")
            
            # Best Planarity
            summary_lines.append("Best Planarity:")
            summary_lines.append(f"  r_nn: {best_planarity['r_nn']:.1f} m")
            summary_lines.append(f"  dt_nn: {best_planarity['dt_nn']:.1f} h")
            summary_lines.append(f"  λ2/3: {best_planarity['mean_lambda23_ratio']:.1f}")
            
            # Add Best Focal if available
            if best_focal:
                summary_lines.append("")
                summary_lines.append("Best Focal:")
                summary_lines.append(f"  r_nn: {best_focal['r_nn']:.1f} m")
                summary_lines.append(f"  dt_nn: {best_focal['dt_nn']:.1f} h")
                if 'mean_angular_diff' in best_focal:
                    summary_lines.append(f"  Focal: {best_focal['mean_angular_diff']:.1f}°")
            
            summary_text = "\n".join(summary_lines)
            ax6.text(0.5, 0.5, summary_text, ha='center', va='center', 
                    fontsize=9, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange', linewidth=2))
            
            # Distribution of objectives in Pareto front (row 2, col 2)
            ax7 = fig.add_subplot(gs[2, 2])
            
            if pareto_angular:
                ax7.hist(pareto_angular, bins=10, alpha=0.6, color='red', label='Focal fit (°)')
                ax7_twin = ax7.twinx()
                ax7_twin.hist([r*100 for r in pareto_recovery], bins=10, alpha=0.6, 
                             color='green', label='Recovery (%)')
                ax7.set_xlabel('Value', fontsize=10)
                ax7.set_ylabel('Frequency (Focal)', fontsize=9, color='red')
                ax7_twin.set_ylabel('Frequency (Recovery)', fontsize=9, color='green')
                ax7.set_title('Objective Distributions in Pareto Front', fontsize=11, fontweight='bold')
            
            ax7.grid(True, alpha=0.3, axis='y')
            
        else:
            # 2 objectives: Recovery vs Quality
            # 1. 2D Pareto Front (top left, double width)
            ax1 = fig.add_subplot(gs[0, :2])
            
            ax1.scatter([r*100 for r in all_recovery], all_lambda23,
                       c='lightgray', s=30, alpha=0.4, label='All trials')
            ax1.scatter([r*100 for r in pareto_recovery], pareto_lambda23,
                       c='blue', s=150, alpha=0.7, edgecolors='black', linewidth=1.5, 
                       label='Pareto front', zorder=5)
            
            # Highlight representatives
            ax1.scatter([best_recovery['plane_recovery_rate']*100], 
                       [best_recovery['mean_lambda23_ratio']],
                       c='green', s=350, marker='D', edgecolors='darkgreen', linewidth=2,
                       label='Best recovery', zorder=10)
            
            ax1.scatter([best_planarity['plane_recovery_rate']*100], 
                       [best_planarity['mean_lambda23_ratio']],
                       c='purple', s=350, marker='^', edgecolors='indigo', linewidth=2,
                       label='Best planarity', zorder=10)
            
            ax1.scatter([best_balanced['plane_recovery_rate']*100], 
                       [best_balanced['mean_lambda23_ratio']],
                       c='orange', s=400, marker='s', edgecolors='darkorange', linewidth=2,
                       label='Best balanced', zorder=10)
            
            ax1.set_xlabel('Plane Recovery Rate (%)', fontsize=12)
            ax1.set_ylabel('Lambda2/3 Ratio (Quality)', fontsize=12)
            ax1.set_title('Pareto Front: Recovery vs Quality Trade-off', fontsize=13, fontweight='bold')
            # Place legend to the right
            ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.9)
            ax1.grid(True, alpha=0.3)
            
            # Parameter space with Pareto front (top right for 2-obj)
            ax2 = fig.add_subplot(gs[0, 2])
            
            ax2.scatter(all_r_nn, all_dt_nn, c='lightgray', s=20, alpha=0.3, label='All trials')
            ax2.scatter(pareto_r_nn, pareto_dt_nn, c='blue', s=80, alpha=0.7, 
                       edgecolors='black', linewidth=1, label='Pareto front')
            
            # Highlight representatives
            for sol, color, marker, label in [
                (best_balanced, 'orange', 's', 'Balanced'),
                (best_recovery, 'green', 'D', 'Recovery'),
                (best_planarity, 'purple', '^', 'Planarity')
            ]:
                ax2.scatter([sol['r_nn']], [sol['dt_nn']], c=color, s=200, marker=marker,
                           edgecolors='black', linewidth=1.5, label=label, zorder=10)
            
            ax2.set_xlabel('Search Radius (m)', fontsize=11)
            ax2.set_ylabel('Time Window (hours)', fontsize=11)
            ax2.set_title('Parameter Space: Pareto Solutions', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Recovery vs Quality for 2-obj (middle left)
            ax3 = fig.add_subplot(gs[1, 0])
            
            ax3.scatter([r*100 for r in all_recovery], all_lambda23,
                       c='lightgray', s=20, alpha=0.3)
            ax3.scatter([r*100 for r in pareto_recovery], pareto_lambda23,
                       c='blue', s=80, alpha=0.7, edgecolors='black', linewidth=1)
            
            for sol, color, marker in [(best_recovery, 'green', 'D'), (best_planarity, 'purple', '^'),
                                       (best_balanced, 'orange', 's')]:
                ax3.scatter([sol['plane_recovery_rate']*100], [sol['mean_lambda23_ratio']],
                           c=color, s=200, marker=marker, edgecolors='black', 
                           linewidth=1.5, zorder=10)
            
            ax3.set_xlabel('Plane Recovery Rate (%)', fontsize=11)
            ax3.set_ylabel('Lambda2/3 Ratio', fontsize=11)
            ax3.set_title('Recovery vs Quality Detail', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Representative Solutions Summary (bottom left for 2-obj)
            ax4 = fig.add_subplot(gs[2, 0])
            ax4.axis('off')
            
            # Build summary text with emphasis on balanced
            summary_lines = ["Representative Solutions", "=" * 28, ""]
            
            # Best Balanced (emphasized)
            summary_lines.append("★ BEST BALANCED ★")
            summary_lines.append(f"  r_nn: {best_balanced['r_nn']:.1f} m")
            summary_lines.append(f"  dt_nn: {best_balanced['dt_nn']:.1f} h")
            summary_lines.append(f"  Recovery: {best_balanced['plane_recovery_rate']*100:.1f}%")
            summary_lines.append(f"  λ2/3: {best_balanced['mean_lambda23_ratio']:.1f}")
            summary_lines.append("")
            
            # Best Recovery
            summary_lines.append("Best Recovery:")
            summary_lines.append(f"  r_nn: {best_recovery['r_nn']:.1f} m")
            summary_lines.append(f"  dt_nn: {best_recovery['dt_nn']:.1f} h")
            summary_lines.append(f"  Recovery: {best_recovery['plane_recovery_rate']*100:.1f}%")
            summary_lines.append("")
            
            # Best Planarity
            summary_lines.append("Best Planarity:")
            summary_lines.append(f"  r_nn: {best_planarity['r_nn']:.1f} m")
            summary_lines.append(f"  dt_nn: {best_planarity['dt_nn']:.1f} h")
            summary_lines.append(f"  λ2/3: {best_planarity['mean_lambda23_ratio']:.1f}")
            
            summary_text = "\n".join(summary_lines)
            ax4.text(0.5, 0.5, summary_text, ha='center', va='center', 
                    fontsize=10, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange', linewidth=2))
            
            # Distribution of objectives in Pareto front (bottom center for 2-obj)
            ax5 = fig.add_subplot(gs[2, 1])
            ax5.hist([r*100 for r in pareto_recovery], bins=10, alpha=0.6, color='green')
            ax5.set_xlabel('Plane Recovery Rate (%)', fontsize=11)
            ax5.set_ylabel('Frequency', fontsize=11)
            ax5.set_title('Recovery in Pareto Front', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            
            # Summary statistics (bottom right for 2-obj)
            ax6 = fig.add_subplot(gs[2, 2])
            ax6.axis('off')
            
            summary_text = f"""
Pareto Multi-Objective Optimization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pareto Front: {len(pareto_front)} solutions
Total Trials: {settings['n_trials']}
Sampler: {settings['sampler'].upper()}

Best Balanced Solution:
  r_nn: {best_balanced['r_nn']:.1f} m
  dt_nn: {best_balanced['dt_nn']:.1f} h
  Recovery: {best_balanced['plane_recovery_rate']*100:.1f}%
  λ2/3: {best_balanced['mean_lambda23_ratio']:.1f}
"""
            
            ax6.text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # Remove suptitle and use tight layout for better space usage
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Pareto optimization plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def optimize(self, method=None, **kwargs):
        """
        Main optimization interface.
        
        Parameters
        ----------
        method : str, optional
            Optimization method ('grid_search', 'optuna', 'pareto', 'heuristic')
        **kwargs
            Method-specific parameters
            
        Returns
        -------
        dict
            Optimization results
        """
        if method is None:
            method = self.method
        
        start_time = datetime.now()
        print(f"\nStarting parameter optimization using {method} method...")

        if method == 'grid_search':
            results = self.optimize_grid_search(**kwargs)
        elif method == 'optuna':
            results = self.optimize_optuna(**kwargs)
        elif method == 'pareto':
            results = self.optimize_pareto(**kwargs)
        elif method == 'heuristic':
            results = self.optimize_heuristic(**kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}. "
                           f"Choose from: 'grid_search', 'optuna', 'pareto', 'heuristic'")
        
        end_time = datetime.now()
        optimization_time = end_time - start_time
        results['optimization_time'] = optimization_time.total_seconds()
        # Convert to minutes and seconds for display
        minutes, seconds = divmod(optimization_time.total_seconds(), 60)
        optimization_time = f"{int(minutes)}m {int(seconds)}s"

        print(f"\nOptimization completed in {optimization_time}")

        return results
    
    def get_recommended_parameters(self):
        """
        Get recommended parameters from optimization results.
        
        Returns
        -------
        dict
            Recommended parameters with confidence estimates
        """
        if not self.optimization_results:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        results = self.optimization_results
        
        # Handle Pareto multi-objective optimization results differently
        if results['method'] == 'pareto':
            # Use best_balanced solution as the recommended parameters
            best_solution = results['best_balanced']
            
            recommendation = {
                'search_radius_meters': best_solution['r_nn'],
                'search_time_window_hours': best_solution['dt_nn'],
                'method_used': results['method'],
                'confidence_score': best_solution['plane_recovery_rate'],  # Use recovery rate as confidence
                'calculated fault planes': best_solution['n_planes'],
                'plane_recovery_rate': best_solution['plane_recovery_rate']
            }
            
            # Add focal mechanism metrics if available
            if 'mean_angular_diff' in best_solution:
                recommendation['expected_angular_difference'] = best_solution['mean_angular_diff']
            
            # Add quality metrics
            if 'mean_lambda23_ratio' in best_solution:
                recommendation['mean_lambda23_ratio'] = best_solution['mean_lambda23_ratio']
            
        else:
            # Handle single-objective optimization results (grid_search, optuna, heuristic)
            recommendation = {
                'search_radius_meters': results['best_params']['r_nn'],
                'search_time_window_hours': results['best_params']['dt_nn'],
                'method_used': results['method'],
                'confidence_score': 1 - results['best_score'],  # Convert to confidence (0-1)
                'calculated fault planes': results['best_details']['n_planes'],
                'plane_recovery_rate': results['best_details']['plane_recovery_rate']
            }
            
            # Add focal mechanism metrics if available
            if 'focal_metrics' in results['best_details']:
                focal_metrics = results['best_details']['focal_metrics']
                if 'mean_angular_diff' in focal_metrics:
                    recommendation['expected_angular_difference'] = focal_metrics['mean_angular_diff']
                if 'active_plane_match_rate' in focal_metrics:
                    recommendation['active_plane_accuracy'] = focal_metrics['active_plane_match_rate']
        
        return recommendation


def optimize_fault_network_parameters(data_input, focal_mechanisms=None, method='grid_search', 
                                     custom_r_nn_range=None, custom_dt_nn_range=None, verbose=False, 
                                     use_adaptive_weights=True, n_matched_focals=None, **kwargs):
    """
    Convenience function for parameter optimization.
    
    Parameters
    ----------
    data_input : pd.DataFrame
        Hypocenter catalog
    focal_mechanisms : pd.DataFrame, optional
        Focal mechanism data with Strike1, Dip1, Strike2, Dip2, A columns
    method : str
        Optimization method ('grid_search', 'optuna', 'heuristic')
    custom_r_nn_range : tuple, optional
        Custom search radius range (min_meters, max_meters)
    custom_dt_nn_range : tuple, optional
        Custom time window range (min_hours, max_hours)
    verbose : bool
        Whether to show verbose output during optimization runs (default: False)
    use_adaptive_weights : bool
        Whether to use adaptive weighting based on dataset characteristics (default: True)
    n_matched_focals : int, optional
        Number of focal mechanisms that match hypocenters in dataset
    **kwargs
        Method-specific parameters
        
    Returns
    -------
    dict
        Recommended parameters
    """
    # Calculate matched focals if not provided
    if n_matched_focals is None and focal_mechanisms is not None and 'ID' in data_input.columns:
        if 'ID' in focal_mechanisms.columns:
            n_matched_focals = len(focal_mechanisms[focal_mechanisms['ID'].isin(data_input['ID'])])
        else:
            n_matched_focals = None
    
    optimizer = ParameterOptimizer(data_input, focal_mechanisms, method, 
                                   custom_r_nn_range, custom_dt_nn_range, verbose=verbose,
                                   use_adaptive_weights=use_adaptive_weights,
                                   n_matched_focals=n_matched_focals)
    optimizer.optimize(method, **kwargs)
    return optimizer.get_recommended_parameters()
