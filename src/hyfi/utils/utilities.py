#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS: General utilities and helper function

Please cite: Truttmann et al. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps.

@author: Sandro Truttmann
@contact: sandro.truttmann@gmail.com
@license: GPL-3.0
@date: April 2023
@version: 0.1.1
"""

# Import modules
import numpy as np
import numba
import pandas as pd
import os
import json
import shutil
from pathlib import Path


def setup_output_directory(output_dir):
    """
    Create output directory structure if it doesn't exist.
    
    Parameters
    ----------
    output_dir : str or Path
        Path to the output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def dag_params_to_legacy_params(dag, node_id):
    """
    Convert DAG node parameters to legacy input_params format.
    
    Parameters
    ----------
    dag : HyFIWorkflowDAG
        DAG configuration object
    node_id : str
        ID of the current node
        
    Returns
    -------
    dict
        Legacy input_params dictionary
    """
    # Base parameters
    input_params = {
        'out_dir': dag.output_directory,
        'hypo_file': dag.nodes['input_data'].hypocenter_file,
        'hypo_sep': dag.nodes['input_data'].hypocenter_separator,
        'project_title': getattr(dag, 'workflow_name', 'Fault Imaging Analysis'),
    }
    
    # Add focal mechanism data if available
    if dag.nodes['input_data'].focal_mechanism_file:
        input_params.update({
            'foc_file': dag.nodes['input_data'].focal_mechanism_file,
            'foc_sep': dag.nodes['input_data'].focal_mechanism_separator,
        })
    
    # Add node-specific parameters
    if node_id == 'fault_network':
        params = dag.nodes['fault_network'].parameters
        
        # Core network parameters
        core_network = params.get('core_network', {})
        input_params.update({
            'n_mc': core_network.get('monte_carlo_simulations', params.get('monte_carlo_simulations', 1000)),
            'r_nn': core_network.get('search_radius_meters', params.get('search_radius_meters', 100.0)),
            'dt_nn': core_network.get('search_time_window_hours', params.get('search_time_window_hours', 9999999)),
            'mag_type': core_network.get('magnitude_type', params.get('magnitude_type', 'ML')),
        })
        
        # Outlier detection parameters
        outlier_detection = params.get('outlier_detection', {})
        input_params.update({
            'remove_outliers': outlier_detection.get('remove_outliers', params.get('remove_outliers', False)),
            'outlier_method': outlier_detection.get('outlier_method', params.get('outlier_method', 'DBSCAN')),
            'lof_n_neighbors': outlier_detection.get('lof_n_neighbors', params.get('lof_n_neighbors', None)),
            'lof_contamination': outlier_detection.get('lof_contamination', params.get('lof_contamination', 'auto')),
            'if_n_estimators': outlier_detection.get('if_n_estimators', params.get('if_n_estimators', 100)),
            'if_max_samples': outlier_detection.get('if_max_samples', params.get('if_max_samples', 'auto')),
            'if_contamination': outlier_detection.get('if_contamination', params.get('if_contamination', 0.05)),
            'if_random_state': outlier_detection.get('if_random_state', params.get('if_random_state', 42)),
        })
        
        # Focal mechanism constraints
        focal_constraints = params.get('focal_mechanism_constraints', {})
        input_params.update({
            'use_focal_constraints': focal_constraints.get('use_focal_constraints', params.get('use_focal_constraints', False)),
        })
        
        # Automatic parameter optimization
        auto_opt = params.get('automatic_parameter_optimization', {})
        input_params.update({
            'auto_optimize_parameters': auto_opt.get('auto_optimize_parameters', params.get('auto_optimize_parameters', False)),
            'optimization_method': auto_opt.get('optimization_method', params.get('optimization_method', 'optuna')),
            'optimization_random_state': auto_opt.get('optimization_random_state', params.get('optimization_random_state', 42)),
            'optimization_n_trials': auto_opt.get('optimization_n_trials', params.get('optimization_n_trials', 50)),
            'optimization_grid_points': auto_opt.get('optimization_grid_points', params.get('optimization_grid_points', 25)),
            'optimization_plot_results': auto_opt.get('optimization_plot_results', params.get('optimization_plot_results', False)),
            'optimization_r_nn_range': auto_opt.get('optimization_r_nn_range', params.get('optimization_r_nn_range', None)),
            'optimization_dt_nn_range': auto_opt.get('optimization_dt_nn_range', params.get('optimization_dt_nn_range', None)),
            'optimization_n_startup_trials': auto_opt.get('optimization_n_startup_trials', params.get('optimization_n_startup_trials', 10)),
            'optimization_pareto_sampler': auto_opt.get('optimization_pareto_sampler', params.get('optimization_pareto_sampler', 'nsga2')),
            'optimization_pareto_population': auto_opt.get('optimization_pareto_population', params.get('optimization_pareto_population', 50)),
            'optimization_sampler': auto_opt.get('optimization_sampler', params.get('optimization_sampler', 'tpe')),
        })
        
        # Node enable/disable flags
        input_params.update({
            'validation_bool': getattr(dag.nodes.get('model_validation'), 'enabled', True),
            'autoclass_bool': getattr(dag.nodes.get('auto_classification'), 'enabled', True),
            'stress_bool': getattr(dag.nodes.get('stress_analysis'), 'enabled', True),
        })
        
        # Add validation parameters
        if 'model_validation' in dag.nodes:
            mv_params = dag.nodes['model_validation'].parameters
            input_params.update({
                'foc_mag_check': mv_params.get('check_magnitude_consistency', True),
                'foc_loc_check': mv_params.get('check_location_consistency', True),
                'foc_max_mag_diff': mv_params.get('maximum_magnitude_difference', 0.2),
                'foc_max_dist_km': mv_params.get('maximum_distance_km', 1.0),
            })
            
        # Add classification parameters
        if 'auto_classification' in dag.nodes:
            ac_params = dag.nodes['auto_classification'].parameters
            
            # Orientation clustering parameters
            orientation_clustering = ac_params.get('orientation_clustering', {})
            input_params.update({
                'auto_determine_clusters': orientation_clustering.get('auto_determine_clusters', ac_params.get('auto_determine_clusters', True)),
                'max_clusters': orientation_clustering.get('max_clusters', ac_params.get('max_clusters', 8)),
                'n_clusters': orientation_clustering.get('number_of_clusters', ac_params.get('number_of_clusters', 2)),
                'algorithm': orientation_clustering.get('clustering_algorithm', ac_params.get('clustering_algorithm', 'vmf_soft')),
                'rotation': orientation_clustering.get('rotate_poles_before_analysis', ac_params.get('rotate_poles_before_analysis', True)),
                'convergence_tolerance': orientation_clustering.get('convergence_tolerance', ac_params.get('convergence_tolerance', None)),
                'maximum_iterations': orientation_clustering.get('maximum_iterations', ac_params.get('maximum_iterations', 300)),
            })
            
            # Spatial sub-clustering parameters (consolidated structure)
            spatial_subclustering = ac_params.get('spatial_sub_clustering', {})
            input_params.update({
                'enable_spatial_clustering': spatial_subclustering.get('enable_spatial_clustering', ac_params.get('enable_spatial_clustering', True)),
                'spatial_clustering_method': spatial_subclustering.get('spatial_clustering_method', ac_params.get('spatial_clustering_method', 'dbscan')),
                'min_points_per_subcluster': spatial_subclustering.get('min_events_per_cluster', ac_params.get('min_events_per_cluster', 10)),
                'min_events_per_cluster': spatial_subclustering.get('min_events_per_cluster', ac_params.get('min_events_per_cluster', 10)),
                'use_fault_plane_points_for_clustering': spatial_subclustering.get('use_fault_plane_points_for_clustering', ac_params.get('use_fault_plane_points_for_clustering', False)),
                'fault_plane_point_density_meters': spatial_subclustering.get('fault_plane_point_density_meters', ac_params.get('fault_plane_point_density_meters', 10.0)),
                'fault_plane_radius_interval_meters': spatial_subclustering.get('fault_plane_radius_interval_meters', ac_params.get('fault_plane_radius_interval_meters', 10.0)),
                'fault_plane_clustering_eps_meters': spatial_subclustering.get('fault_plane_clustering_eps_meters', ac_params.get('fault_plane_clustering_eps_meters', 200.0)),
                'fault_plane_clustering_min_samples': spatial_subclustering.get('fault_plane_clustering_min_samples', ac_params.get('fault_plane_clustering_min_samples', 5)),
                # Anisotropic clustering parameters
                'use_anisotropic_eps': spatial_subclustering.get('use_anisotropic_eps', ac_params.get('use_anisotropic_eps', False)),
                'in_plane_eps_meters': spatial_subclustering.get('in_plane_eps_meters', ac_params.get('in_plane_eps_meters', 500.0)),
                'out_of_plane_eps_meters': spatial_subclustering.get('out_of_plane_eps_meters', ac_params.get('out_of_plane_eps_meters', 50.0)),
                'anisotropic_min_samples': spatial_subclustering.get('anisotropic_min_samples', ac_params.get('anisotropic_min_samples', 5)),
            })
            
        # Add stress analysis parameters  
        if 'stress_analysis' in dag.nodes:
            sa_params = dag.nodes['stress_analysis'].parameters
            stress_field = sa_params.get('stress_field', {})
            mech_props = sa_params.get('mechanical_properties', {})
            input_params.update({
                'S1_trend': stress_field.get('sigma1_trend_degrees', 301),
                'S1_plunge': stress_field.get('sigma1_plunge_degrees', 23),
                'S3_trend': stress_field.get('sigma3_trend_degrees', 43),
                'S3_plunge': stress_field.get('sigma3_plunge_degrees', 26),
                'stress_R': stress_field.get('stress_shape_ratio', 0.35),
                'PP': mech_props.get('pore_pressure_mpa', 0.0),
                'fric_coeff': mech_props.get('friction_coefficient', 0.75),
                'use_shapefile_stress': stress_field.get('use_shapefile', False),
                'stress_field_shapefile': stress_field.get('shapefile_path', None),
            })
        
        # Add visualization parameters
        if 'visualization' in dag.nodes:
            vis_params = dag.nodes['visualization'].parameters
            
            # Basic visualization parameters
            basic_vis = vis_params.get('basic_visualization', {})
            input_params.update({
                'generate_3d_model': basic_vis.get('generate_3d_model', vis_params.get('generate_3d_model', True)),
                'generate_stereonet': basic_vis.get('generate_stereonet', vis_params.get('generate_stereonet', True)),
            })
            
            # Fault surface interpolation parameters
            interpolation = vis_params.get('fault_surface_interpolation', {})
            input_params.update({
                'enable_plane_interpolation': interpolation.get('enable_plane_interpolation', vis_params.get('enable_plane_interpolation', True)),
                'poisson_depth': interpolation.get('poisson_depth', vis_params.get('poisson_depth', 3)),
                'density_threshold': interpolation.get('density_threshold', vis_params.get('density_threshold', 0.4)),
                'max_distance_factor': interpolation.get('max_distance_factor', vis_params.get('max_distance_factor', 1.5)),
                'min_fault_planes_for_interpolation': interpolation.get('min_fault_planes_for_interpolation', vis_params.get('min_fault_planes_for_interpolation', 10)),
            })
            
            # 3D export parameters
            vtk_export = vis_params.get('3d_export', {})
            input_params.update({
                'export_vtp': vtk_export.get('export_vtp', vis_params.get('export_vtp', True)),
            })
    
    return input_params

def trendplunge_to_vector(trend, plunge):
    """
    Convert from trend/plunge to x/y/z coordinate system.

    Parameters
    ----------
    trend : DataFrame
        Trend.
    plunge : DataFrame
        Plunge.

    Returns
    -------
    XYZ coordinates.

    """
    # Convert to radians
    trend = np.radians(trend)
    plunge = np.radians(plunge)
    # Calculate the normal unit vector components
    nor_x = np.sin(0.5 * np.pi - plunge) * np.sin(trend)
    nor_y = np.sin(0.5 * np.pi - plunge) * np.cos(trend)
    nor_z = - np.cos(0.5 * np.pi - plunge)
    
    return(nor_x, nor_y, nor_z)


@numba.njit
def rake_to_trendplunge(plane_strike, plane_dip, rake):
    """
    Convert from Strike-Dip-Rake to lineation trend and plunge.

    Parameters
    ----------
    plane_strike : int
        Plane strike (right-hand rule (RHR)).
    plane_dip : int
        Plane dip.
    rake : int
        Rake in RHR format (0-180° measured from RHR strike direction).

    Returns
    -------
    Trend and plunge.

    """

    # Convert degrees to radians
    S = np.deg2rad(plane_strike)
    D = np.deg2rad(plane_dip)
    R = np.deg2rad(rake)
    
    # Calculate beta in dependence of rake
    beta = abs(np.arctan(np.tan(R) * np.cos(D)))
    beta = np.pi - beta if R > (np.pi / 2) else beta
    
    # Calculate lineation trend and plunge
    trend = S + beta
    plunge = np.arcsin(np.sin(D) * np.sin(R))
    
    # Convert to degrees and round
    trend = int(round(np.degrees(trend))) % 360
    plunge = int(round(np.degrees(plunge)))

    return(trend, plunge)


@numba.njit
def plane_azidip_to_normal(azi, dip):
    """
    Convert plane azimuth and dip (spherical) to normal vector (cartesian).

    Parameters
    ----------
    azi : int
        Plane azimuth in degrees.
    dip : int
        Plane dip in degrees.

    Returns
    -------
    Normal unit vector to the plane in cartesian coordinates.

    """
    # Calculate the orientation of the pole to the plane
    pole_azi = np.radians(azi) + np.pi % 360
    pole_dip = 0.5 * np.pi - np.radians(dip)

    # Calculate the normal unit vector components
    nor_x = np.sin(0.5 * np.pi - pole_dip) * np.sin(pole_azi)
    nor_y = np.sin(0.5 * np.pi - pole_dip) * np.cos(pole_azi)
    nor_z = - np.cos(0.5 * np.pi - pole_dip)

    return(nor_x, nor_y, nor_z)


@numba.njit
def plane_normal_to_azidip(nor_x, nor_y, nor_z):
    """
    Convert plane normal vector (cartesian) to azimuth and dip (cartesian).

    Parameters
    ----------
    nor_x : float
        X component of normal unit vector.
    nor_y : float
        Y component of normal unit vector.
    nor_z : float
        Z component of normal unit vector.

    Returns
    -------
    Plane azimuth and dip .

    """
    # Calulate the plane orientation from the normal vector
    pole_azi = np.arctan2(nor_x, nor_y)
    azi = int(round(np.degrees(pole_azi - np.pi))) % 360

    # Calculate the plane dip from the normal vector
    pole_dip = np.pi - np.arccos(nor_z)
    dip = int(round(np.degrees(pole_dip)))

    return(azi, dip)


def circular_mean_azimuth(azimuths):
    """
    Calculate circular mean of azimuth angles (0-360°).
    
    Uses vector averaging to properly handle circular nature of angles.
    For example, the mean of 5° and 355° is 0°, not 180°.
    
    Parameters
    ----------
    azimuths : array-like
        Array of azimuth values in degrees (0-360)
        
    Returns
    -------
    float or None
        Circular mean azimuth in degrees (0-360), or None if input is empty/all NaN
    """
    azimuths = np.array(azimuths)
    # Remove NaN values
    azimuths = azimuths[~np.isnan(azimuths)]
    
    if len(azimuths) == 0:
        return None
    
    # Convert to radians
    angles_rad = np.radians(azimuths)
    
    # Calculate mean of unit vectors
    mean_x = np.mean(np.sin(angles_rad))
    mean_y = np.mean(np.cos(angles_rad))
    
    # Convert back to angle
    mean_angle = np.degrees(np.arctan2(mean_x, mean_y))
    
    # Ensure result is in [0, 360) range
    return float(mean_angle % 360)


def circular_mean_orientation_from_azimuth_dip(azimuths, dips):
    """
    Calculate mean orientation by averaging normal vectors, then converting back.
    
    This is the proper way to average fault plane orientations, as it accounts
    for the 3D geometry and avoids circular statistics issues.
    
    Parameters
    ----------
    azimuths : array-like
        Array of azimuth values in degrees (0-360)
    dips : array-like
        Array of dip values in degrees (0-90)
        
    Returns
    -------
    tuple of (float, float) or (None, None)
        Mean azimuth and dip in degrees, or (None, None) if input is empty/all NaN
    """
    azimuths = np.array(azimuths)
    dips = np.array(dips)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(azimuths) | np.isnan(dips))
    azimuths = azimuths[valid_mask]
    dips = dips[valid_mask]
    
    if len(azimuths) == 0:
        return None, None
    
    # Convert azimuth/dip to normal vectors
    normals = []
    for azi, dip in zip(azimuths, dips):
        # Convert to radians
        azi_rad = np.radians(azi)
        dip_rad = np.radians(dip)
        
        # Calculate normal vector (pointing upward)
        # Dip direction is the azimuth, dip angle is measured from horizontal
        nx = np.sin(dip_rad) * np.sin(azi_rad)
        ny = np.sin(dip_rad) * np.cos(azi_rad)
        nz = np.cos(dip_rad)
        
        normals.append([nx, ny, nz])
    
    # Average the normal vectors
    mean_normal = np.mean(normals, axis=0)
    
    # Normalize
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    
    # Convert back to azimuth/dip
    mean_azi, mean_dip = plane_normal_to_azidip(mean_normal[0], mean_normal[1], mean_normal[2])
    
    return float(mean_azi), float(mean_dip)


def angle_between(v1, v2):
    """
    Return the angle in degrees between two vectors.

    Parameters
    ----------
    v1 : list
        Vector with XYZ components of vector 1.
    v2 : list
        Vector with XYZ components of vector 2.

    Returns
    -------
    Angle between the two vectors (in degrees) .

    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle = np.degrees(angle)

    return angle


def save_data(input_params, data_input, data_output, per_X, per_Y, per_Z, config_source_file=None):
    """
    Save data in txt-files.

    Parameters
    ----------
    input_params : DataFrame
        Input parameters.
    data_input : DataFrame
        Input data (includes outliers marked with clust_labels = -1).
    data_output : DataFrame
        Output data.
    per_X : DataFrame
        X coordinates of MC datasets.
    per_Y : DataFrame
        Y coordinates of MC datasets.
    per_Z : DataFrame
        Z coordinates of MC datasets.
    config_source_file : str, optional
        Path to the original configuration file to copy to output.

    Returns
    -------
    None.

    """
    
    # Unpack input parameters from dictionary
    for key, value in input_params.items():
        globals()[key] = value

    # Create output folder
    out_path = input_params['out_dir']
    os.makedirs(out_path, exist_ok=True)
        
    # Save only the meaningful merged data (renamed)
    # Merge data_input and data_output
    # First, ensure no duplicate columns between data_input and data_output
    # Remove any columns from data_output that already exist in data_input (except 'ID')
    duplicate_cols = [col for col in data_output.columns if col in data_input.columns and col != 'ID']
    if duplicate_cols:
        print(f"Removing duplicate columns from data_output before merge: {duplicate_cols}")
        data_output_clean = data_output.drop(columns=duplicate_cols)
    else:
        data_output_clean = data_output
    
    df_merge = pd.merge(data_input, data_output_clean, on='ID', how='left')
    df_merge.to_csv(out_path + '/HyFI_results.csv', sep=';', date_format='%Y-%m-%dT%H:%M:%S.%f')
    
    # Save the input data outliers if they exist
    if 'DBSCAN_outliers' in input_params:
        DBSCAN_outliers = input_params['DBSCAN_outliers']
    else:
        DBSCAN_outliers = False

    # Note: Outliers are now included in data_input with clust_labels = -1
    # No separate outliers file is saved
    
    # Copy the original DAG configuration file to the output directory
    if config_source_file:
        try:
            source_path = Path(config_source_file)
            if source_path.exists():
                dest_path = Path(out_path) / f"{source_path.name}"
                shutil.copy2(source_path, dest_path)
        except Exception as e:
            print(f"Warning: Could not copy configuration file: {e}")

    return


def reduced_stress_tens(fric_coeff, stress_R):
    """
    Normalize the slip tendency according to Lisle & Srivastava (2004).

    Parameters
    ----------
    fric_coeff : float
        Coeffifient of friction.
    stress_R : float
        Stress shape ratio R.

    Returns
    -------
    Normalized stress magnitudes.

    """
    # Calculate stress magnitudes for normalized slip and dilation tendency if
    # defined
    k = 1
    rho = np.arctan(fric_coeff)
    S1_mag = 0.5 * k * ((1 / np.sin(rho)) + 1)
    S2_mag = S1_mag - k * (1 - stress_R)
    S3_mag = S1_mag - k
    PP = 0
    
    return(S1_mag, S2_mag, S3_mag, PP)


def calculate_plane_corners(center_x, center_y, center_z, strike, dip, plane_size):
    """
    Calculate corners of a fault plane for 3D visualization.
    
    Parameters
    ----------
    center_x, center_y, center_z : float
        Center coordinates of the fault plane
    strike : float
        Strike angle in degrees (0-360)
    dip : float
        Dip angle in degrees (0-90)
    plane_size : float
        Size of the fault plane (half-width/height) in meters
        
    Returns
    -------
    list of tuples
        Four corner coordinates [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3), (x4,y4,z4)]
        or None if calculation fails
    """
    try:
        # Convert angles to radians
        strike_rad = np.radians(strike)
        dip_rad = np.radians(dip)
        
        # Define unit vectors
        # Strike vector (horizontal, along strike direction)
        strike_vec = np.array([np.cos(strike_rad), np.sin(strike_rad), 0])
        
        # Dip vector (down-dip direction)
        dip_vec = np.array([
            -np.sin(strike_rad) * np.cos(dip_rad),
            np.cos(strike_rad) * np.cos(dip_rad), 
            -np.sin(dip_rad)
        ])
        
        # Calculate four corners of the fault plane
        corners = []
        
        # Corner offsets relative to center
        offsets = [
            (-plane_size, -plane_size),  # Bottom left
            (plane_size, -plane_size),   # Bottom right  
            (plane_size, plane_size),    # Top right
            (-plane_size, plane_size)    # Top left
        ]
        
        for strike_offset, dip_offset in offsets:
            corner = np.array([center_x, center_y, center_z]) + \
                    strike_offset * strike_vec + \
                    dip_offset * dip_vec
            corners.append((corner[0], corner[1], corner[2]))
        
        return corners
        
    except Exception as e:
        print(f"Warning: Could not calculate plane corners: {e}")
        return None


def fault_network_with_optimization(input_params):
    """
    Execute fault network reconstruction with automatic parameter optimization if enabled.
    
    This function serves as a wrapper around the original faultnetwork3D function,
    adding automatic parameter optimization capabilities when requested.
    
    Parameters
    ----------
    input_params : dict
        Input parameters dictionary. May contain optimization settings:
        - auto_optimize_parameters: bool, enable automatic optimization
        - optimization_method: str, optimization method ('grid_search', 'heuristic')
        - optimization_grid_points: int, grid points for grid search
        
    Returns
    -------
    tuple
        Same as faultnetwork3D: (data_input, data_output, df_per_X, df_per_Y, df_per_Z)
    """
    from ..core import fault_network, model_validation
    from .parameter_optimization import ParameterOptimizer
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Check if parameter optimization is requested
    auto_optimize = input_params.get('auto_optimize_parameters', False)
    r_nn_auto = input_params.get('r_nn') == 'auto'
    dt_nn_auto = input_params.get('dt_nn') == 'auto'
    
    if auto_optimize or r_nn_auto or dt_nn_auto:
        print("\n")
        print("="*50)
        print("AUTOMATIC PARAMETER OPTIMIZATION")
        print("="*50)
        
        # Load hypocenter data
        try:
            hypo_file = input_params.get('hypo_file')
            hypo_sep = input_params.get('hypo_sep', ',')
            
            if not hypo_file:
                raise ValueError("hypo_file must be specified for parameter optimization")
            
            # Load hypocenter data (using same logic as fault_network.py)
            data_input = pd.read_csv(hypo_file, sep=hypo_sep, dtype={'ID': str}, usecols=range(17))
            
            # Extract date and time information
            data_input['Date'] = pd.to_datetime(pd.DataFrame({'year': data_input['YR'],
                                'month': data_input['MO'],
                                'day': data_input['DY'],
                                'hour': data_input['HR'],
                                'minute': data_input['MI'],
                                'second': data_input['SC']}))
            data_input['X'] = data_input['X']
            data_input['Y'] = data_input['Y']
            # Use Z coordinate directly as provided in input data
            
            print(f"Loaded {len(data_input)} events for parameter optimization")
            
        except Exception as e:
            logger.error(f"Failed to load hypocenter data for optimization: {e}")
            logger.warning("Falling back to default parameters")
            return fault_network.faultnetwork3D(input_params)
        
        # Load focal mechanism data if available
        focal_mechanisms = None
        n_matched_focals = 0  # Count of focal mechanisms that match hypocenters
        if input_params.get('validation_bool', False) and input_params.get('foc_file'):
            try:
                foc_file = input_params.get('foc_file')
                foc_sep = input_params.get('foc_sep', ';')
                focal_mechanisms = pd.read_csv(foc_file, sep=foc_sep)
                print(f"Loaded {len(focal_mechanisms)} focal mechanisms for validation")
                
                # Count how many focal mechanisms match hypocenters in this dataset
                if 'ID' in focal_mechanisms.columns and 'ID' in data_input.columns:
                    # Ensure both ID columns have the same data type for matching
                    focal_ids = focal_mechanisms['ID'].astype(str)
                    data_ids = data_input['ID'].astype(str)
                    matched_focals = focal_mechanisms[focal_ids.isin(data_ids)]
                    n_matched_focals = len(matched_focals)
                    print(f"    {n_matched_focals} focal mechanisms found in the current dataset")
                else:
                    print("    Warning: Could not match focal mechanisms to hypocenters (missing ID column)")

            except Exception as e:
                print(f"Failed to load focal mechanism data: {e}")
                focal_mechanisms = None
                n_matched_focals = 0
        
        # Set up parameter optimizer
        optimization_method = input_params.get('optimization_method', 'optuna')
        
        # Extract custom optimization ranges if provided
        custom_r_nn_range = input_params.get('optimization_r_nn_range', None)
        custom_dt_nn_range = input_params.get('optimization_dt_nn_range', None)
        
        # Convert ranges from list to tuple if provided
        if custom_r_nn_range and isinstance(custom_r_nn_range, list) and len(custom_r_nn_range) == 2:
            custom_r_nn_range = tuple(custom_r_nn_range)
        elif custom_r_nn_range and not isinstance(custom_r_nn_range, tuple):
            logger.warning(f"Invalid r_nn range format: {custom_r_nn_range}. Expected [min, max] or (min, max)")
            custom_r_nn_range = None
            
        if custom_dt_nn_range and isinstance(custom_dt_nn_range, list) and len(custom_dt_nn_range) == 2:
            custom_dt_nn_range = tuple(custom_dt_nn_range)
        elif custom_dt_nn_range and not isinstance(custom_dt_nn_range, tuple):
            logger.warning(f"Invalid dt_nn range format: {custom_dt_nn_range}. Expected [min, max] or (min, max)")
            custom_dt_nn_range = None
        
        # Get adaptive weights setting (default: True for improved variable dataset handling)
        use_adaptive_weights = input_params.get('optimization_use_adaptive_weights', True)
        
        optimizer = ParameterOptimizer(data_input, focal_mechanisms, optimization_method,
                                       custom_r_nn_range, custom_dt_nn_range, verbose=False,
                                       use_adaptive_weights=use_adaptive_weights,
                                       n_matched_focals=n_matched_focals,
                                       original_input_params=input_params)
        
        # Perform optimization
        try:
            optimization_kwargs = {}
            if optimization_method == 'grid_search':
                optimization_kwargs['n_points'] = input_params.get('optimization_grid_points', 25)
                optimization_kwargs['plot_results'] = input_params.get('optimization_plot_results', False)
                
                # Set up plot save path if output directory is available
                if optimization_kwargs['plot_results']:
                    if 'out_dir' in input_params:
                        plot_path = os.path.join(input_params['out_dir'], 'parameter_optimization_grid.png')
                        optimization_kwargs['save_plot_path'] = plot_path
                    else:
                        # Default plot path if no output directory specified
                        optimization_kwargs['save_plot_path'] = 'parameter_optimization_grid.png'
                        logger.info("No output directory specified, saving plot to current directory")
                        
            elif optimization_method == 'optuna':
                optimization_kwargs['n_trials'] = input_params.get('optimization_n_trials', 50)
                optimization_kwargs['sampler'] = input_params.get('optimization_sampler', 'tpe')
                optimization_kwargs['n_startup_trials'] = input_params.get('optimization_n_startup_trials', 10)
                optimization_kwargs['early_stopping_rounds'] = input_params.get('optimization_early_stopping_rounds', None)
                optimization_kwargs['early_stopping_threshold'] = input_params.get('optimization_early_stopping_threshold', 1e-4)
                optimization_kwargs['plot_results'] = input_params.get('optimization_plot_results', False)
                
                # Set up plot save path if output directory is available
                if optimization_kwargs['plot_results']:
                    if 'out_dir' in input_params:
                        plot_path = os.path.join(input_params['out_dir'], 'parameter_optimization.png')
                        optimization_kwargs['save_plot_path'] = plot_path
                    else:
                        # Default plot path if no output directory specified
                        optimization_kwargs['save_plot_path'] = 'parameter_optimization.png'
                        logger.info("No output directory specified, saving plot to current directory")
            
            elif optimization_method == 'pareto':
                optimization_kwargs['n_trials'] = input_params.get('optimization_n_trials', 100)  # More trials for Pareto
                optimization_kwargs['sampler'] = input_params.get('optimization_pareto_sampler', 'nsga2')
                optimization_kwargs['n_startup_trials'] = input_params.get('optimization_n_startup_trials', 20)
                optimization_kwargs['population_size'] = input_params.get('optimization_pareto_population', 50)
                optimization_kwargs['plot_results'] = input_params.get('optimization_plot_results', False)
                
                # Set up plot save path if output directory is available
                if optimization_kwargs['plot_results']:
                    if 'out_dir' in input_params:
                        plot_path = os.path.join(input_params['out_dir'], 'parameter_optimization_pareto.png')
                        optimization_kwargs['save_plot_path'] = plot_path
                    else:
                        # Default plot path if no output directory specified
                        optimization_kwargs['save_plot_path'] = 'parameter_optimization_pareto.png'
                        logger.info("No output directory specified, saving plot to current directory")
            
            results = optimizer.optimize(method=optimization_method, **optimization_kwargs)
            recommended_params = optimizer.get_recommended_parameters()
            
            # Update input_params with optimized values
            if r_nn_auto or auto_optimize:
                input_params['r_nn'] = recommended_params['search_radius_meters']
                print(f"Optimized search radius: {recommended_params['search_radius_meters']:.1f} m")
            
            if dt_nn_auto or auto_optimize:
                input_params['dt_nn'] = recommended_params['search_time_window_hours']
                print(f"Optimized time window: {recommended_params['search_time_window_hours']:.1f} h")
            
            # Log optimization results
            print(f"Calculated fault planes: {recommended_params['calculated fault planes']}")
            print(f"Plane recovery rate: {recommended_params['plane_recovery_rate']:.3f}")
            
            if 'expected_angular_difference' in recommended_params:
                print(f"Mean angular difference: {recommended_params['expected_angular_difference']:.1f}°")

            # Save optimization results
            if 'out_dir' in input_params:
                optimization_report_path = os.path.join(input_params['out_dir'], 'parameter_optimization_report.json')
                try:
                    os.makedirs(input_params['out_dir'], exist_ok=True)
                    with open(optimization_report_path, 'w') as f:
                        # Convert numpy types to native Python for JSON serialization
                        json_results = {}
                        for key, value in recommended_params.items():
                            if isinstance(value, (np.integer, np.floating)):
                                json_results[key] = value.item()
                            else:
                                json_results[key] = value
                        
                        # Include the full optimization results for heuristic method
                        optimization_data = {
                            'optimization_results': json_results,
                            'optimization_method': optimization_method,
                            'catalog_statistics': optimizer.catalog_stats
                        }
                        
                        # Add detailed results if available (mainly for grid_search)
                        if hasattr(optimizer, 'optimization_results') and optimizer.optimization_results:
                            # Convert optimization_results for JSON serialization
                            opt_results = optimizer.optimization_results.copy()
                            # Remove non-serializable objects like DataFrames
                            if 'all_results' in opt_results:
                                clean_results = []
                                for result in opt_results['all_results']:
                                    clean_result = result.copy()
                                    if 'data_output' in clean_result:
                                        del clean_result['data_output']  # Remove DataFrame
                                    clean_results.append(clean_result)
                                opt_results['all_results'] = clean_results
                            
                            if 'best_details' in opt_results and 'data_output' in opt_results['best_details']:
                                del opt_results['best_details']['data_output']  # Remove DataFrame
                            
                            optimization_data['detailed_results'] = opt_results
                        
                        json.dump(optimization_data, f, indent=2, default=str)
                except Exception as e:
                    logger.warning(f"Failed to save optimization report: {e}")
                    import traceback
                    logger.debug(f"Full error: {traceback.format_exc()}")
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            logger.warning("Falling back to default parameters")
            # Use default values if optimization fails
            if r_nn_auto:
                input_params['r_nn'] = 1000.0
            if dt_nn_auto:
                input_params['dt_nn'] = 876000.0
    
    # Execute fault network reconstruction with (possibly optimized) parameters
    return fault_network.faultnetwork3D(input_params)
