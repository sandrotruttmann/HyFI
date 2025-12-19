#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS: Auto Classification Module

Please cite: Truttmann et al. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps.

@author: Sandro Truttmann
@contact: sandro.truttmann@gmail.com
@license: GPL-3.0
@date: April 2023
@version: 0.1.1
"""

# Import modules
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from ..utils import utilities
from ..utils import spherecluster


def _orientation_clustering_optimized(df_valid, input_params):
    """
    Perform optimized orientation clustering on valid fault plane data.
    
    Parameters
    ----------
    df_valid : DataFrame
        DataFrame with valid fault plane data (mean_azi, mean_dip, etc.)
    input_params : dict
        Parameters including clustering configuration
        
    Returns
    -------
    df_valid : DataFrame
        Input dataframe with added 'orient_cluster' column containing orientation cluster labels
    """
    # Extract parameters
    auto_determine_clusters = input_params.get('auto_determine_clusters', True)
    n_clusters = input_params.get('n_clusters', 3)
    max_clusters = input_params.get('max_clusters', 8)
    algorithm = input_params.get('algorithm', 'vmf_soft')
    rotation = input_params.get('rotation', True)
    
    # Convert fault plane parameters to normal vectors
    X, Y, Z = df_valid['nor_x_mean'], df_valid['nor_y_mean'], df_valid['nor_z_mean']
    data = np.array([X, Y, Z]).T
    
    # Option to cluster subvertical structures  
    if rotation:
        # Rotate the data to similar directions if necessary
        # Ensure that all vectors point to similar direction as first entry
        v1 = [data[0][0], data[0][1], data[0][2]]
        v1 = v1 / np.linalg.norm(v1)
        # Check every point in the dataset and swap direction if it
        # lies on the other side of the stereoplot
        # (angular difference larger than 90 degrees)
        for j in range(len(data)):
            vj = [data[j, 0], data[j, 1], data[j, 2]]
            vj = vj / np.linalg.norm(vj)
            if np.linalg.norm(v1 - vj) == 0:
                angle_deg = np.nan
            else:
                angle_deg = np.degrees(np.arccos(np.dot(v1, vj)))
            if angle_deg > 90:
                data[j, 0] = data[j, 0] * -1
                data[j, 1] = data[j, 1] * -1
                data[j, 2] = data[j, 2] * -1

    # Automatically determine optimal number of clusters if requested
    if auto_determine_clusters:
        print("Automatically determining optimal number of clusters...")
        convergence_tolerance = input_params.get('convergence_tolerance', None)
        maximum_iterations = input_params.get('maximum_iterations', 300)
        n_clusters, cluster_results = _determine_optimal_clusters(data, max_clusters, algorithm, 
                                                                   convergence_tolerance, maximum_iterations)
        print(f"Using {n_clusters} clusters based on automatic determination.")
    else:
        print(f"Using specified number of clusters: {n_clusters}")

    # Extract convergence parameters (with defaults matching spherecluster)
    # SphericalKMeans uses tol=1e-4, VonMisesFisherMixture uses tol=1e-6
    convergence_tolerance = input_params.get('convergence_tolerance', None)
    maximum_iterations = input_params.get('maximum_iterations', 300)
    
    # Apply clustering with the specified estimator
    if algorithm == 'skm':
        # Spherical k-Means clustering (default tol=1e-4)
        tol = convergence_tolerance if convergence_tolerance is not None else 1e-4
        skm = spherecluster.SphericalKMeans(n_clusters=n_clusters, max_iter=maximum_iterations, tol=tol)
        skm.fit(data)
        labels = skm.labels_
        cluster_centers = skm.cluster_centers_                
    elif algorithm == 'vmf_soft':
        # Von Mises Fisher Mixture soft clustering (default tol=1e-6)
        tol = convergence_tolerance if convergence_tolerance is not None else 1e-6
        vmf_soft = spherecluster.VonMisesFisherMixture(n_clusters=n_clusters, posterior_type='soft', 
                                                        max_iter=maximum_iterations, tol=tol)
        vmf_soft.fit(data)
        labels = vmf_soft.labels_
        cluster_centers = vmf_soft.cluster_centers_
    elif algorithm == 'vmf_hard':
        # Von Mises Fisher Mixture hard clustering (default tol=1e-6)
        tol = convergence_tolerance if convergence_tolerance is not None else 1e-6
        vmf_hard = spherecluster.VonMisesFisherMixture(n_clusters=n_clusters, posterior_type='hard',
                                                        max_iter=maximum_iterations, tol=tol)
        vmf_hard.fit(data)
        labels = vmf_hard.labels_
        cluster_centers = vmf_hard.cluster_centers_
    else:
        # Raise error for unsupported algorithm name
        raise ValueError('Unsupported clustering algorithm: {}'.format(algorithm))
    
    # Add the orientation cluster labels
    df_valid['orient_cluster'] = labels
    
    return df_valid


def _spatial_clustering_with_enhanced_points(df_clustered, df_enhanced, input_params, starting_fault_counter=1, 
                                              sequence_label=None, segmentation_level=None):
    """
    Perform spatial clustering using pre-generated enhanced point cloud.
    
    Parameters
    ----------
    df_clustered : DataFrame
        DataFrame with orientation clustering results (original faults)
    df_enhanced : DataFrame  
        Enhanced point cloud from generate_enhanced_fault_dataset
    input_params : dict
        Parameters including spatial clustering configuration
    starting_fault_counter : int
        Starting value for global fault system counter
    sequence_label : str, optional
        Sequence label (e.g., 'A1', 'B2')
    segmentation_level : str, optional
        Segmentation level (e.g., 'A', 'B')
        
    Returns
    -------
    df_clustered : DataFrame
        Input dataframe with added spatial clustering columns and global fault IDs
    fault_metadata : list
        Metadata for each fault system
    next_counter : int
        Next available counter value
    """
    from .enhanced_point_cloud import get_enhanced_coordinates, get_fault_mapping, aggregate_cluster_labels_to_faults
    from sklearn.cluster import DBSCAN
    
    # Get spatial clustering parameters
    spatial_method = input_params.get('spatial_clustering_method', 'dbscan')
    min_points_spatial = input_params.get('min_points_per_subcluster', 10)
    eps_factor = input_params.get('fault_plane_clustering_eps_factor', 0.3)
    min_samples_factor = input_params.get('fault_plane_clustering_min_samples_factor', 0.3)
    
    print(f"  Using enhanced point cloud with {len(df_enhanced):,} points")
    print(f"  Parameters: spatial_method={spatial_method}, eps_factor={eps_factor}, min_samples_factor={min_samples_factor}")
    
    # Initialize spatial cluster columns
    df_clustered['spatial_cluster'] = 0
    df_clustered['final_cluster_id'] = df_clustered['orient_cluster'].astype(str)
    
    # Initialize fault counter with global counter
    fault_counter = starting_fault_counter
    fault_metadata = []
    
    # Process each orientation cluster separately
    unique_classes = df_clustered['orient_cluster'].unique()
    
    for class_id in unique_classes:
        # Get faults in this orientation cluster
        cluster_fault_indices = df_clustered[df_clustered['orient_cluster'] == class_id].index
        
        print(f"    Cluster {class_id}: applying spatial sub-clustering to {len(cluster_fault_indices)} rupture planes")
        
        # Get enhanced points for faults in this orientation cluster
        enhanced_mask = df_enhanced['source_fault_idx'].isin(cluster_fault_indices)
        df_cluster_enhanced = df_enhanced[enhanced_mask].copy()
        
        if len(df_cluster_enhanced) == 0:
            print(f"    No enhanced points found for cluster {class_id}")
            continue
        
        print(f"        Using {len(df_cluster_enhanced)} enhanced points for clustering")
        
        # Extract coordinates and fault mapping
        points = get_enhanced_coordinates(df_cluster_enhanced)
        fault_mapping = get_fault_mapping(df_cluster_enhanced)
        
        # Apply DBSCAN clustering
        if spatial_method == 'dbscan':
            # Calculate eps based on original hypocenter spread
            original_coords = df_clustered.loc[cluster_fault_indices, ['X', 'Y', 'Z']].values
            hypocenter_range = np.ptp(original_coords, axis=0).mean()
            eps = hypocenter_range * eps_factor
            print(f"        Calculated eps from range: {hypocenter_range:.1f}m × {eps_factor} = {eps:.1f}m")
            
            # Calculate min_samples based on average points per fault
            avg_points_per_fault = len(points) / len(cluster_fault_indices)
            min_samples = max(3, int(avg_points_per_fault * min_samples_factor))
            
            print(f"        DBSCAN: eps={eps:.1f}m, min_samples={min_samples} ({avg_points_per_fault:.1f} avg points/fault)")
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            point_labels = dbscan.fit_predict(points)
            
            n_found_clusters = len(set(point_labels)) - (1 if -1 in point_labels else 0)
            print(f"        Found {n_found_clusters} spatial clusters")
            
            # Aggregate point labels back to fault labels
            fault_clusters = aggregate_cluster_labels_to_faults(point_labels, fault_mapping)
            
            # Create mapping from spatial labels to TEMPORARY IDs
            # These will be converted to permanent FS IDs after successful interpolation
            unique_spatial_labels = set(fault_clusters.values())
            spatial_to_temp_map = {}
            temp_counter = 0
            for spatial_id in sorted(unique_spatial_labels):
                if spatial_id != -1:  # Skip noise points
                    temp_id = f"TEMP{temp_counter:04d}"
                    spatial_to_temp_map[spatial_id] = temp_id
                    temp_counter += 1
                else:
                    spatial_to_temp_map[spatial_id] = np.nan  # Use NaN for noise points
            
            # Update the main dataframe with temporary IDs
            for fault_idx, spatial_label in fault_clusters.items():
                if fault_idx in df_clustered.index:
                    df_clustered.loc[fault_idx, 'spatial_cluster'] = spatial_label
                    temp_id = spatial_to_temp_map[spatial_label]
                    df_clustered.loc[fault_idx, 'final_cluster_id'] = temp_id
            
            # Print results
            spatial_counts = {}
            for fault_idx in cluster_fault_indices:
                if fault_idx in df_clustered.index:
                    spatial_id = df_clustered.loc[fault_idx, 'spatial_cluster']
                    spatial_counts[spatial_id] = spatial_counts.get(spatial_id, 0) + 1
            print(f"    Final spatial clusters: {spatial_counts}")
        
        else:
            print(f"    Warning: Spatial method '{spatial_method}' not implemented for enhanced points")
    
    return df_clustered, fault_metadata, fault_counter


def _spatial_clustering_by_orientation(df_clustered, input_params, starting_fault_counter=1, 
                                        sequence_label=None, segmentation_level=None):
    """
    Perform spatial sub-clustering within each orientation cluster.
    
    Parameters
    ----------
    df_clustered : DataFrame
        DataFrame with orientation clustering results
    input_params : dict
        Parameters including spatial clustering configuration
    starting_fault_counter : int
        Starting value for global fault system counter
    sequence_label : str, optional
        Sequence label (e.g., 'A1', 'B2')
    segmentation_level : str, optional
        Segmentation level (e.g., 'A', 'B')
        
    Returns
    -------
    df_clustered : DataFrame
        Input dataframe with added spatial clustering columns and global fault IDs
    fault_metadata : list
        Metadata for each fault system
    next_counter : int
        Next available counter value
    """
    # Get spatial clustering parameters
    spatial_method = input_params.get('spatial_clustering_method', 'dbscan')
    min_points_spatial = input_params.get('min_points_per_subcluster', 10)
    
    # Get fault plane point parameters
    use_fault_plane_points = input_params.get('use_fault_plane_points_for_clustering', False)
    fault_plane_point_density = input_params.get('fault_plane_point_density_meters', 10.0)
    fault_plane_radius_interval = input_params.get('fault_plane_radius_interval_meters', 10.0)
    fault_plane_eps_factor = input_params.get('fault_plane_clustering_eps_factor', 0.3)
    fault_plane_min_samples_factor = input_params.get('fault_plane_clustering_min_samples_factor', 0.3)
    
    if use_fault_plane_points:
        print(f"  Using fault plane points for enhanced spatial clustering")
        print(f"    Point density: {fault_plane_point_density}m, radius interval: {fault_plane_radius_interval}m")
        print(f"    Sensitivity: eps_factor={fault_plane_eps_factor}, min_samples_factor={fault_plane_min_samples_factor}")
    
    # Initialize spatial cluster columns
    df_clustered['spatial_cluster'] = 0
    df_clustered['final_cluster_id'] = df_clustered['orient_cluster'].astype(str)
    
    # Check if we have coordinate data available for spatial clustering
    coord_cols = ['X', 'Y', 'Z']
    if not all(col in df_clustered.columns for col in coord_cols):
        print("  Warning: Coordinate data not available, skipping spatial sub-clustering")
        return df_clustered
    
    print("  Performing spatial sub-clustering within each orientation cluster...")
    
    # Initialize fault counter with global counter
    fault_counter = starting_fault_counter
    fault_metadata = []
    
    # Process each orientation cluster separately
    unique_classes = df_clustered['orient_cluster'].unique()
    
    for class_id in unique_classes:
        cluster_mask = df_clustered['orient_cluster'] == class_id
        df_cluster = df_clustered[cluster_mask].copy()
        
        print(f"    Cluster {class_id}: applying spatial sub-clustering to {len(df_cluster)} points")
        
        # Perform spatial clustering
        try:
            df_cluster_spatial = _spatial_clustering(
                df_cluster, 
                method=spatial_method, 
                min_points=min_points_spatial,
                use_fault_plane_points=use_fault_plane_points,
                point_density_meters=fault_plane_point_density,
                radius_interval_meters=fault_plane_radius_interval,
                eps_factor=fault_plane_eps_factor,
                min_samples_factor=fault_plane_min_samples_factor
            )
            
            # Update the main dataframe with spatial cluster results
            spatial_labels = df_cluster_spatial['spatial_cluster'].values
            df_clustered.loc[cluster_mask, 'spatial_cluster'] = spatial_labels
            
            # Print spatial clustering results for debugging
            unique_spatial = np.unique(spatial_labels)
            spatial_counts = {label: np.sum(spatial_labels == label) for label in unique_spatial}
            print(f"    Spatial clusters found: {spatial_counts}")
            
            # Create final cluster IDs using TEMPORARY IDs
            # These will be converted to permanent FS IDs after successful interpolation
            # Use proper indexing to avoid alignment issues
            cluster_indices = df_clustered[cluster_mask].index
            unique_spatial_labels = np.unique(spatial_labels)
            
            # Create mapping from spatial labels to TEMPORARY IDs
            spatial_to_temp_map = {}
            temp_counter = 0
            for spatial_id in sorted(unique_spatial_labels):
                if spatial_id != -1:  # Skip noise points
                    temp_id = f"TEMP{temp_counter:04d}"
                    spatial_to_temp_map[spatial_id] = temp_id
                    temp_counter += 1
                else:
                    spatial_to_temp_map[spatial_id] = np.nan  # Use NaN for noise points
            
            # Assign temporary cluster IDs
            for i, (idx, spatial_id) in enumerate(zip(cluster_indices, spatial_labels)):
                temp_id = spatial_to_temp_map[spatial_id]
                df_clustered.loc[idx, 'final_cluster_id'] = temp_id
                
        except Exception as e:
            print(f"    Warning: Spatial clustering failed for cluster {class_id}: {e}")
            continue
    
    return df_clustered, fault_metadata, fault_counter


def _filter_small_clusters(df_clustered, min_events_per_cluster):
    """
    Filter out spatial clusters that have fewer events than min_events_per_cluster.
    Sets final_cluster_id to NaN for events in clusters that are too small.
    
    Parameters
    ----------
    df_clustered : DataFrame
        DataFrame with spatial clustering results
    min_events_per_cluster : int
        Minimum number of events required per final cluster
        
    Returns
    -------
    df_clustered : DataFrame
        DataFrame with small clusters filtered out (set to NaN)
    """
    print(f"  Filtering clusters with < {min_events_per_cluster} events...")
    
    # Count events per final cluster
    cluster_counts = df_clustered['final_cluster_id'].value_counts()
    
    # Identify clusters that are too small
    small_clusters = cluster_counts[cluster_counts < min_events_per_cluster].index
    
    if len(small_clusters) > 0:
        # Count events that will be filtered out
        filtered_events = df_clustered[df_clustered['final_cluster_id'].isin(small_clusters)].shape[0]
        
        print(f"    Filtering {len(small_clusters)} clusters with < {min_events_per_cluster} events")
        print(f"    Events affected: {filtered_events}")
        
        # Set final_cluster_id to NaN for small clusters
        mask = df_clustered['final_cluster_id'].isin(small_clusters)
        df_clustered.loc[mask, 'final_cluster_id'] = np.nan
        
        # Also set spatial_cluster to NaN for consistency
        df_clustered.loc[mask, 'spatial_cluster'] = np.nan
    else:
        print("    No clusters below minimum size threshold")
    
    return df_clustered


def _spatial_clustering(df_cluster, method='dbscan', n_clusters=2, min_points=10, use_fault_plane_points=False, 
                        point_density_meters=12.0, radius_interval_meters=20.0, eps_factor=0.3, min_samples_factor=0.3):
    """
    Perform spatial clustering to split orientation clusters into spatial sub-clusters.
    
    Parameters
    ----------
    df_cluster : DataFrame
        DataFrame containing fault plane data for a single orientation cluster
    method : str
        Clustering method: 'kmeans', 'dbscan', or 'adaptive'
    n_clusters : int
        Number of clusters for K-means (ignored for other methods)
    min_points : int
        Minimum points per cluster
    use_fault_plane_points : bool
        If True, generate multiple points per fault plane for clustering
    point_density_meters : float
        Target distance between points on fault plane circumference
    radius_interval_meters : float
        Fixed interval between concentric circles on fault planes
    eps_factor : float
        Factor to scale eps parameter for DBSCAN when using fault plane points (smaller = more sensitive)
    min_samples_factor : float
        Factor to scale min_samples for DBSCAN when using fault plane points (smaller = more sensitive)
        
    Returns
    -------
    DataFrame
        Input DataFrame with added 'spatial_cluster' column
    """
    
    # Check for coordinate columns with multiple possible names
    possible_coord_names = [
        ['X', 'Y', 'Z'],     # Standard format
        ['LON', 'LAT', 'DEPTH']  # Geographic coordinates
    ]
    
    coord_cols = None
    for coord_set in possible_coord_names:
        if all(col in df_cluster.columns for col in coord_set):
            coord_cols = coord_set
            break
    
    if coord_cols is None:
        # Check if we have fault plane center coordinates that can be derived
        if all(col in df_cluster.columns for col in ['nor_x_mean', 'nor_y_mean', 'nor_z_mean', 'rupt_radius']):
            print(f"    No direct coordinates found, but fault plane data available")
            print(f"    Available columns: {list(df_cluster.columns)}")
            
            # For spatial clustering, we need some spatial coordinates
            # Since we don't have hypocenter coordinates, we cannot perform meaningful spatial clustering
            print(f"    Skipping spatial clustering - no hypocenter coordinates available")
            df_cluster = df_cluster.copy()
            df_cluster['spatial_cluster'] = 0
            return df_cluster
        else:
            print(f"    ERROR: No valid coordinate columns found in DataFrame")
            print(f"    Available columns: {list(df_cluster.columns)}")
            print(f"    Tried: {possible_coord_names}")
            # Skip spatial clustering for this cluster
            df_cluster = df_cluster.copy()
            df_cluster['spatial_cluster'] = 0
            return df_cluster
    
    print(f"    Using coordinate columns: {coord_cols}")
    original_coords = df_cluster[coord_cols].values
    original_indices = df_cluster.index.values
    
    # Determine whether to use fault plane points for enhanced clustering
    if use_fault_plane_points:
        try:
            from ..visualization.visualisation import _generate_fault_plane_points
        except ImportError:
            print("    Warning: Could not import _generate_fault_plane_points. Using single hypocenter points.")
            use_fault_plane_points = False
    
    if use_fault_plane_points:
        print(f"    Generating fault plane points for enhanced spatial clustering")
        
        # Check if we have the required fault plane data for point generation
        required_cols = ['nor_x_mean', 'nor_y_mean', 'nor_z_mean', 'rupt_radius']
        if not all(col in df_cluster.columns for col in required_cols):
            print(f"    Warning: Missing fault plane data for point generation. Required: {required_cols}")
            print(f"    Available columns: {list(df_cluster.columns)}")
            print(f"    Falling back to single hypocenter points.")
            use_fault_plane_points = False
    
    if use_fault_plane_points:
        # Generate multiple points per fault plane using the visualization function
        try:
            # Safety check: estimate point count before generation
            estimated_points = 0
            for _, row in df_cluster.iterrows():
                r = row['rupt_radius']
                # Rough estimate of points per fault
                n_rings = int(r / radius_interval_meters) + 2  # +2 for safety
                avg_points_per_ring = int(2 * np.pi * (r/2) / point_density_meters)
                estimated_points += 1 + n_rings * avg_points_per_ring
            
            if estimated_points > 50000:  # Safety limit
                print(f"    Warning: Estimated {estimated_points} fault plane points would be generated.")
                print(f"    This is too many. Adjusting parameters to reduce count.")
                # Increase point spacing to reduce count
                point_density_meters = max(point_density_meters, estimated_points / 10000)
                radius_interval_meters = max(radius_interval_meters, estimated_points / 5000)
                print(f"    Adjusted: point_density={point_density_meters:.1f}m, radius_interval={radius_interval_meters:.1f}m")
            
            fault_points, _ = _generate_fault_plane_points(
                df_cluster, 
                radius_interval=radius_interval_meters,
                point_density_meters=point_density_meters
            )
            
            print(f"    Generated {len(fault_points)} points from {len(df_cluster)} fault planes")
            
            # Create mapping from each fault plane point back to its original fault
            # This requires reconstructing which points belong to which fault
            point_to_fault_map = []
            point_start_idx = 0
            
            for fault_idx, (_, row) in enumerate(df_cluster.iterrows()):
                # Calculate how many points this fault should have generated
                r = row['rupt_radius']
                
                # Replicate the logic from _generate_fault_plane_points
                # 1. Center point
                n_points_this_fault = 1
                
                # 2. Points on rings at fixed intervals
                current_radius = radius_interval_meters
                while current_radius <= r:
                    circumference = 2 * np.pi * current_radius
                    n_points_circle = max(8, int(circumference / point_density_meters))
                    n_points_this_fault += n_points_circle
                    current_radius += radius_interval_meters
                
                # 3. Edge ring if not already included
                radii = []
                current_radius = radius_interval_meters
                while current_radius <= r:
                    radii.append(current_radius)
                    current_radius += radius_interval_meters
                
                if len(radii) == 0 or radii[-1] < r:
                    # Edge ring was added
                    circumference = 2 * np.pi * r
                    n_points_circle = max(8, int(circumference / point_density_meters))
                    n_points_this_fault += n_points_circle
                
                # Map all these points to this fault
                point_to_fault_map.extend([fault_idx] * n_points_this_fault)
                point_start_idx += n_points_this_fault
            
            points = fault_points
            
        except Exception as e:
            print(f"    Warning: Could not generate fault plane points: {e}")
            print(f"    Falling back to single hypocenter points.")
            # Fall back to hypocenter points
            points = original_coords
            point_to_fault_map = list(range(len(original_coords)))
    else:
        # Use original single hypocenter points
        points = original_coords
        point_to_fault_map = list(range(len(original_coords)))
    
    if len(points) < min_points:
        # Too few points, treat as single cluster
        df_cluster = df_cluster.copy()
        df_cluster['spatial_cluster'] = 0
        return df_cluster
    
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(points)
        
        # Map point clusters back to fault clusters if using fault plane points
        if use_fault_plane_points:
            fault_cluster_map = {}
            for point_idx, fault_idx in enumerate(point_to_fault_map):
                if fault_idx not in fault_cluster_map:
                    fault_cluster_map[fault_idx] = []
                fault_cluster_map[fault_idx].append(cluster_labels[point_idx])
            
            # Assign cluster based on majority vote for each fault
            final_labels = []
            for fault_idx in range(len(df_cluster)):
                from collections import Counter
                cluster_votes = Counter(fault_cluster_map[fault_idx])
                final_labels.append(cluster_votes.most_common(1)[0][0])
            
            cluster_labels = np.array(final_labels)
    
    elif method == 'dbscan':
        # Use adaptive eps based on data spread, with different strategies for fault plane points
        if use_fault_plane_points:
            # For fault plane points, use hypocenter spread as basis, not full point cloud spread
            # This keeps eps sensitive to the original fault separation
            hypocenter_range = np.ptp(original_coords, axis=0).mean()
            
            # Make eps smaller for fault plane points to maintain sensitivity
            eps = hypocenter_range * eps_factor  # Configurable sensitivity
            
            # Also consider typical fault plane size for context
            avg_fault_radius = df_cluster['rupt_radius'].mean() if 'rupt_radius' in df_cluster.columns else 50.0
            
            # Use the smaller of hypocenter-based or fault-size-based eps
            fault_size_eps = avg_fault_radius * 0.8  # Allow some overlap between nearby faults
            eps = min(eps, fault_size_eps)
            
            print(f"    DBSCAN fault-plane clustering: eps={eps:.1f}m (factor={eps_factor}, hypocenter-based)")
            
        else:
            # Original logic for single hypocenter points
            data_range = np.ptp(points, axis=0).mean()
            eps = data_range * 0.5  # 50% of data range
            print(f"    DBSCAN spatial clustering: eps={eps:.1f}m")
        
        min_samples = max(4, min_points // 3)
        
        # For fault plane points, can use lower min_samples since we have more points
        if use_fault_plane_points:
            # Scale min_samples based on average points per fault
            avg_points_per_fault = len(points) / len(df_cluster) if len(df_cluster) > 0 else 10
            min_samples = max(3, int(avg_points_per_fault * min_samples_factor))  # Configurable factor
            print(f"    min_samples={min_samples} (factor={min_samples_factor}, {avg_points_per_fault:.1f} avg points/fault)")
        else:
            print(f"    min_samples={min_samples}")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(points)
        n_found_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"    Found {n_found_clusters} spatial clusters")

        # Handle noise points (label -1) by excluding them from fitting
        if -1 in cluster_labels:
            noise_mask = cluster_labels == -1
            if np.sum(~noise_mask) > 0:  # If there are non-noise points
                # Map noise points back to original faults if using fault plane points
                if use_fault_plane_points:
                    # Create mapping from fault index to cluster label
                    fault_cluster_map = {}
                    for point_idx, fault_idx in enumerate(point_to_fault_map):
                        if cluster_labels[point_idx] != -1:  # Non-noise point
                            if fault_idx not in fault_cluster_map:
                                fault_cluster_map[fault_idx] = []
                            fault_cluster_map[fault_idx].append(cluster_labels[point_idx])
                    
                    # Assign cluster based on majority vote for each fault
                    final_labels = []
                    excluded_faults = []
                    for fault_idx in range(len(df_cluster)):
                        if fault_idx in fault_cluster_map:
                            # Use most common cluster assignment for this fault
                            from collections import Counter
                            cluster_votes = Counter(fault_cluster_map[fault_idx])
                            final_labels.append(cluster_votes.most_common(1)[0][0])
                        else:
                            # All points for this fault were noise
                            excluded_faults.append(fault_idx)
                            final_labels.append(-1)
                    
                    if excluded_faults:
                        print(f"    Excluded {len(excluded_faults)} faults where all fault plane points were noise")
                        # Remove excluded faults
                        keep_mask = np.array([i not in excluded_faults for i in range(len(df_cluster))])
                        df_cluster = df_cluster[keep_mask].reset_index(drop=True)
                        final_labels = [label for i, label in enumerate(final_labels) if i not in excluded_faults]
                        cluster_labels = np.array(final_labels)
                    else:
                        cluster_labels = np.array(final_labels)
                else:
                    # Original single-point clustering
                    df_cluster = df_cluster[~noise_mask].reset_index(drop=True)
                    cluster_labels = cluster_labels[~noise_mask]
                    print(f"    Excluded {np.sum(noise_mask)} noise points from spatial clustering")
            else:
                # All points are noise - skip this cluster
                print(f"    All points identified as noise - skipping cluster")
                df_cluster = df_cluster.copy()
                df_cluster['spatial_cluster'] = -1  # Mark as excluded
                return df_cluster
    
    elif method == 'adaptive':
        # Try DBSCAN first, fall back to K-means if too few clusters
        data_range = np.ptp(points, axis=0).mean()
        eps = data_range * 0.15
        min_samples = max(4, min_points // 4)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(points)
        
        # Check if we got reasonable clustering
        n_found_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        if n_found_clusters < 1:
            # DBSCAN failed, use K-means
            n_clusters_adaptive = min(n_clusters, len(points) // min_points)
            if n_clusters_adaptive < 1:
                n_clusters_adaptive = 1
            
            kmeans = KMeans(n_clusters=n_clusters_adaptive, random_state=0)
            cluster_labels = kmeans.fit_predict(points)
            print(f"    DBSCAN failed, using K-means with {n_clusters_adaptive} clusters")
            
            # Map point clusters back to fault clusters if using fault plane points
            if use_fault_plane_points:
                fault_cluster_map = {}
                for point_idx, fault_idx in enumerate(point_to_fault_map):
                    if fault_idx not in fault_cluster_map:
                        fault_cluster_map[fault_idx] = []
                    fault_cluster_map[fault_idx].append(cluster_labels[point_idx])
                
                # Assign cluster based on majority vote for each fault
                final_labels = []
                for fault_idx in range(len(df_cluster)):
                    from collections import Counter
                    cluster_votes = Counter(fault_cluster_map[fault_idx])
                    final_labels.append(cluster_votes.most_common(1)[0][0])
                
                cluster_labels = np.array(final_labels)
        else:
            # Clean up DBSCAN results - exclude noise points
            if -1 in cluster_labels:
                noise_mask = cluster_labels == -1
                if np.sum(~noise_mask) > 0:  # If there are non-noise points
                    # Map noise points back to original faults if using fault plane points
                    if use_fault_plane_points:
                        # Create mapping from fault index to cluster label
                        fault_cluster_map = {}
                        for point_idx, fault_idx in enumerate(point_to_fault_map):
                            if cluster_labels[point_idx] != -1:  # Non-noise point
                                if fault_idx not in fault_cluster_map:
                                    fault_cluster_map[fault_idx] = []
                                fault_cluster_map[fault_idx].append(cluster_labels[point_idx])
                        
                        # Assign cluster based on majority vote for each fault
                        final_labels = []
                        excluded_faults = []
                        for fault_idx in range(len(df_cluster)):
                            if fault_idx in fault_cluster_map:
                                # Use most common cluster assignment for this fault
                                from collections import Counter
                                cluster_votes = Counter(fault_cluster_map[fault_idx])
                                final_labels.append(cluster_votes.most_common(1)[0][0])
                            else:
                                # All points for this fault were noise
                                excluded_faults.append(fault_idx)
                                final_labels.append(-1)
                        
                        if excluded_faults:
                            print(f"    Excluded {len(excluded_faults)} faults where all fault plane points were noise")
                            # Remove excluded faults
                            keep_mask = np.array([i not in excluded_faults for i in range(len(df_cluster))])
                            df_cluster = df_cluster[keep_mask].reset_index(drop=True)
                            final_labels = [label for i, label in enumerate(final_labels) if i not in excluded_faults]
                            cluster_labels = np.array(final_labels)
                        else:
                            cluster_labels = np.array(final_labels)
                    else:
                        # Original single-point clustering
                        # Keep only non-noise points
                        df_cluster = df_cluster[~noise_mask].reset_index(drop=True)
                        cluster_labels = cluster_labels[~noise_mask]
                        print(f"    Excluded {np.sum(noise_mask)} noise points from adaptive clustering")
                else:
                    # All points are noise - skip this cluster
                    print(f"    All points identified as noise - skipping cluster")
                    df_cluster = df_cluster.copy()
                    df_cluster['spatial_cluster'] = -1  # Mark as excluded
                    return df_cluster
    
    df_cluster = df_cluster.copy()
    
    # Map cluster labels back to original faults
    if use_fault_plane_points and len(point_to_fault_map) == len(cluster_labels):
        # Multiple points per fault - need to aggregate
        fault_cluster_labels = np.zeros(len(df_cluster), dtype=int)
        
        for fault_idx in range(len(df_cluster)):
            # Find all points belonging to this fault
            point_indices = [i for i, f_idx in enumerate(point_to_fault_map) if f_idx == fault_idx]
            
            if point_indices:
                # Get cluster labels for all points of this fault
                fault_point_labels = [cluster_labels[i] for i in point_indices]
                
                # Use majority vote to assign cluster to this fault
                from collections import Counter
                if len(set(fault_point_labels)) == 1:
                    # All points agree
                    fault_cluster_labels[fault_idx] = fault_point_labels[0]
                else:
                    # Use most common label
                    label_counts = Counter(fault_point_labels)
                    fault_cluster_labels[fault_idx] = label_counts.most_common(1)[0][0]
        
        df_cluster['spatial_cluster'] = fault_cluster_labels
    else:
        # Single point per fault or no point mapping
        df_cluster['spatial_cluster'] = cluster_labels
    
    return df_cluster


def _evaluate_spatial_clustering_benefit(df_cluster, spatial_clusters, min_points):
    """
    Evaluate whether spatial clustering provides meaningful separation.
    
    Parameters
    ----------
    df_cluster : DataFrame
        DataFrame containing fault plane data with spatial_cluster column
    spatial_clusters : array
        Array of unique spatial cluster IDs
    min_points : int
        Minimum points per cluster
        
    Returns
    -------
    bool
        True if spatial clustering is beneficial, False otherwise
    """
    
    # Check for coordinate columns with multiple possible names
    possible_coord_names = [
        ['X', 'Y', 'Z'],     # Standard format
        ['LON', 'LAT', 'DEPTH']  # Geographic coordinates
    ]
    
    coord_cols = None
    for coord_set in possible_coord_names:
        if all(col in df_cluster.columns for col in coord_set):
            coord_cols = coord_set
            break
    
    if coord_cols is None:
        print(f"    Warning: No coordinate columns found for spatial clustering evaluation")
        return False
    
    # Remove excluded clusters (marked as -1)
    valid_spatial_clusters = [sc for sc in spatial_clusters if sc != -1]
    
    # If only one valid spatial cluster, no benefit from spatial clustering
    if len(valid_spatial_clusters) <= 1:
        return False
    
    # Check if clusters are well-separated and have sufficient points
    cluster_sizes = []
    cluster_centers = []
    
    for spatial_cluster_id in valid_spatial_clusters:
        df_subcluster = df_cluster[df_cluster['spatial_cluster'] == spatial_cluster_id]
        
        # Check minimum size requirement
        if len(df_subcluster) < min_points:
            continue
            
        cluster_sizes.append(len(df_subcluster))
        
        # Calculate cluster center using detected coordinate columns
        center = df_subcluster[coord_cols].mean().values
        cluster_centers.append(center)
    
    # Need at least 2 valid clusters with sufficient points
    if len(cluster_centers) < 2:
        return False
    
    # Check spatial separation - clusters should be reasonably far apart
    cluster_centers = np.array(cluster_centers)
    min_distance = np.inf
    
    for i in range(len(cluster_centers)):
        for j in range(i + 1, len(cluster_centers)):
            distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
            min_distance = min(min_distance, distance)
    
    # Calculate characteristic size of the data using detected coordinate columns
    all_points = df_cluster[coord_cols].values
    data_extent = np.ptp(all_points, axis=0).max()  # Maximum range in any dimension
    
    # Clusters should be separated by at least 10% of the data extent
    min_separation_threshold = data_extent * 0.1
    
    # Also check that clusters aren't too imbalanced
    min_size = min(cluster_sizes)
    max_size = max(cluster_sizes)
    size_ratio = min_size / max_size if max_size > 0 else 0
    
    is_well_separated = min_distance > min_separation_threshold
    is_balanced = size_ratio > 0.2  # Largest cluster shouldn't be more than 5x the smallest
    
    return is_well_separated and is_balanced


def _determine_optimal_clusters(data, max_clusters=8, algorithm='vmf_soft', 
                                convergence_tolerance=None, maximum_iterations=300):
    """
    Automatically determine the optimal number of clusters using multiple criteria.
    
    Parameters
    ----------
    data : numpy.ndarray
        Normalized data points on unit sphere (n_samples, 3)
    max_clusters : int
        Maximum number of clusters to evaluate (default: 8)
    algorithm : str
        Clustering algorithm to use ('vmf_soft', 'vmf_hard', 'skm')
    convergence_tolerance : float, optional
        Convergence tolerance for clustering algorithms. If None, uses algorithm defaults.
    maximum_iterations : int
        Maximum number of iterations (default: 300)
        
    Returns
    -------
    int
        Optimal number of clusters
    dict
        Detailed results for each cluster count tested
    """
    
    if len(data) < 4:
        print("Too few data points for automatic cluster determination. Using 1 cluster.")
        return 1, {}
    
    # Limit max_clusters based on data size (need at least 2 points per cluster)
    max_clusters = min(max_clusters, len(data) // 2)
    if max_clusters < 1:
        max_clusters = 1
    
    print(f"Evaluating optimal number of clusters (range: 1-{max_clusters})...")
    
    results = {}
    
    # Test different numbers of clusters
    for n_clusters in range(1, max_clusters + 1):
        try:
            # Fit clustering model
            if algorithm == 'skm':
                tol = convergence_tolerance if convergence_tolerance is not None else 1e-4
                model = spherecluster.SphericalKMeans(n_clusters=n_clusters, random_state=42,
                                                      max_iter=maximum_iterations, tol=tol)
            elif algorithm == 'vmf_soft':
                tol = convergence_tolerance if convergence_tolerance is not None else 1e-6
                model = spherecluster.VonMisesFisherMixture(n_clusters=n_clusters, posterior_type='soft', 
                                                           random_state=42, max_iter=maximum_iterations, tol=tol)
            elif algorithm == 'vmf_hard':
                tol = convergence_tolerance if convergence_tolerance is not None else 1e-6
                model = spherecluster.VonMisesFisherMixture(n_clusters=n_clusters, posterior_type='hard', 
                                                           random_state=42, max_iter=maximum_iterations, tol=tol)
            else:
                raise ValueError(f'Unsupported algorithm: {algorithm}')
            
            model.fit(data)
            labels = model.labels_
            
            # Calculate clustering quality metrics
            metrics = {}
            
            # Only calculate metrics if we have more than 1 cluster
            if n_clusters > 1 and len(np.unique(labels)) > 1:
                from sklearn.metrics import silhouette_score, calinski_harabasz_score
                
                # Silhouette score (higher is better, range [-1, 1])
                try:
                    metrics['silhouette'] = silhouette_score(data, labels, metric='cosine')
                except:
                    metrics['silhouette'] = -1
                
                # Calinski-Harabasz score (higher is better)
                try:
                    metrics['calinski_harabasz'] = calinski_harabasz_score(data, labels)
                except:
                    metrics['calinski_harabasz'] = 0
                
                # Custom spherical clustering metrics
                # Calculate within-cluster angular dispersion
                angular_dispersions = []
                cluster_sizes = []
                
                for cluster_id in np.unique(labels):
                    cluster_points = data[labels == cluster_id]
                    cluster_sizes.append(len(cluster_points))
                    
                    if len(cluster_points) > 1:
                        # Calculate mean direction (cluster center)
                        center = np.mean(cluster_points, axis=0)
                        center = center / np.linalg.norm(center)
                        
                        # Calculate angular distances to center
                        cos_angles = np.clip(np.dot(cluster_points, center), -1, 1)
                        angles = np.arccos(np.abs(cos_angles))  # Use abs for hemisphere invariance
                        angular_dispersions.append(np.std(angles))
                    else:
                        angular_dispersions.append(0)
                
                metrics['mean_angular_dispersion'] = np.mean(angular_dispersions)
                metrics['cluster_balance'] = np.min(cluster_sizes) / np.max(cluster_sizes) if cluster_sizes else 0
                
            else:
                # Single cluster case
                metrics['silhouette'] = 0
                metrics['calinski_harabasz'] = 0
                metrics['mean_angular_dispersion'] = 0
                metrics['cluster_balance'] = 1
            
            # Additional metrics
            metrics['n_clusters_found'] = len(np.unique(labels))
            metrics['inertia'] = model.inertia_ if hasattr(model, 'inertia_') else 0
            
            results[n_clusters] = metrics
            
            print(f"  {n_clusters} clusters: Silhouette={metrics['silhouette']:.3f}, "
                  f"Angular dispersion={metrics['mean_angular_dispersion']:.3f}")
        
        except Exception as e:
            print(f"  Error evaluating {n_clusters} clusters: {e}")
            continue
    
    if not results:
        print("Could not evaluate any cluster configurations. Using 1 cluster as fallback.")
        return 1, {}
    
    # Determine optimal number of clusters using combined criteria
    best_n_clusters = _select_best_cluster_count(results)
    
    print(f"Automatically determined optimal clusters: {best_n_clusters}")
    
    return best_n_clusters, results


def _select_best_cluster_count(results):
    """
    Select the best number of clusters based on multiple criteria.
    
    Uses a weighted scoring approach with STRONG preference for simpler models:
    - Silhouette score
    - Angular dispersion (lower is better)
    - Strong penalty for too many clusters (prefers 1-3 clusters)
    """
    
    if len(results) <= 1:
        return list(results.keys())[0] if results else 2
    
    scores = {}
    
    # Normalize metrics for scoring
    silhouettes = [results[k]['silhouette'] for k in results.keys()]
    dispersions = [results[k]['mean_angular_dispersion'] for k in results.keys()]
    balances = [results[k]['cluster_balance'] for k in results.keys()]
    
    max_silhouette = max(silhouettes) if max(silhouettes) > 0 else 1
    max_dispersion = max(dispersions) if max(dispersions) > 0 else 1
    
    for n_clusters, metrics in results.items():
        score = 0
        
        # Silhouette score (higher is better) - reduced weight: 0.25
        if max_silhouette > 0:
            score += 0.25 * (metrics['silhouette'] / max_silhouette)
        
        # Angular dispersion (lower is better) - reduced weight: 0.15
        if max_dispersion > 0:
            score += 0.15 * (1 - metrics['mean_angular_dispersion'] / max_dispersion)
        
        # Cluster balance (higher is better) - reduced weight: 0.1
        score += 0.1 * metrics['cluster_balance']
        
        # STRONG complexity penalty - increased weight: 0.5
        # Heavily penalize models with many clusters
        if n_clusters == 1:
            complexity_penalty = 1.0  # No penalty for single cluster
        elif n_clusters == 2:
            complexity_penalty = 0.9  # Small penalty for 2 clusters
        elif n_clusters == 3:
            complexity_penalty = 0.7  # Moderate penalty for 3 clusters
        else:
            # Heavy penalty for 4+ clusters
            complexity_penalty = 0.5 / (n_clusters - 2)
        
        score += 0.5 * complexity_penalty
        
        scores[n_clusters] = score
    
    # Find best score
    best_n_clusters = max(scores.keys(), key=lambda k: scores[k])
    
    # Additional simplicity check: if a simpler model has reasonably close score, prefer it
    sorted_by_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_score = sorted_by_score[0][1]
    
    for n_clusters, score in sorted(scores.items()):  # Sort by cluster count (ascending)
        # If this simpler model has score within 15% of best, prefer it
        if score >= 0.85 * best_score:
            best_n_clusters = n_clusters
            break
    
    # Final safety check: strongly discourage more than 3 clusters unless truly exceptional
    if best_n_clusters > 3:
        # Check if 1, 2, or 3 cluster solutions are reasonable
        simple_options = {k: v for k, v in scores.items() if k <= 3}
        if simple_options:
            simple_best = max(simple_options.keys(), key=lambda k: scores[k])
            # Only use complex solution if it's significantly better (>40% improvement)
            if scores[simple_best] >= 0.6 * scores[best_n_clusters]:
                best_n_clusters = simple_best
                print(f"    Preferring simpler {simple_best}-cluster solution over {len(scores)}-cluster complexity")
    
    return best_n_clusters


def auto_classification(input_params, df_hyfi, starting_fault_counter=1, sequence_label=None, segmentation_level=None):
    """
    Automatic classification of point cloud based on fault orientations.

    Parameters
    ----------
    input_params : dict
        Input parameters containing clustering configuration.
    df_hyfi : DataFrame
        Input dataframe containing all hypocenter data and computed fault parameters.
        Clustering results will be added as new columns to this dataframe.
    starting_fault_counter : int, optional
        Starting value for global fault system counter (default: 1)
    sequence_label : str, optional
        Label of the sequence being processed (e.g., 'A1', 'B2')
    segmentation_level : str, optional
        Segmentation level letter (e.g., 'A', 'B', 'C')

    Returns
    -------
    df_hyfi : DataFrame
        Input dataframe with added clustering columns and global fault system IDs.
    fault_system_metadata : list
        List of dictionaries containing metadata for each fault system
    next_fault_counter : int
        Next available fault counter value
    """
    print('\n')
    print('='*50)
    print('AUTOMATIC FAULT CLASSIFICATION')
    print('='*50)
    
    # Extract key parameters
    autoclass_bool = input_params.get('autoclass_bool', False)
    
    if not autoclass_bool:
        print("Auto classification is disabled")
        return df_hyfi, [], starting_fault_counter
    
    # Check if we have the required fault plane data
    required_columns = ['rupt_plane_azi', 'rupt_plane_dip']
    missing_columns = [col for col in required_columns if col not in df_hyfi.columns or df_hyfi[col].isna().all()]
    
    if missing_columns:
        print(f"Warning: Missing or empty fault plane data: {missing_columns}")
        print("Skipping auto classification")
        return df_hyfi, [], starting_fault_counter
    
    # Filter out rows with missing fault plane data for clustering
    valid_mask = df_hyfi['rupt_plane_azi'].notna() & df_hyfi['rupt_plane_dip'].notna()
    df_valid = df_hyfi[valid_mask].copy()
    
    if len(df_valid) == 0:
        print("No valid fault plane data available for clustering")
        return df_hyfi, [], starting_fault_counter

    print(f"Using {len(df_valid)} events with valid fault plane data for clustering")
    
    # NEW: Generate enhanced point cloud if requested
    use_enhanced_points = input_params.get('use_fault_plane_points_for_clustering', False)
    
    if use_enhanced_points:
        print("\n--- ENHANCED POINT CLOUD GENERATION ---")
        from .enhanced_point_cloud import generate_enhanced_fault_dataset
        
        radius_interval = input_params.get('fault_plane_radius_interval_meters', 10.0)
        point_density = input_params.get('fault_plane_point_density_meters', 10.0)
        
        df_original, df_enhanced = generate_enhanced_fault_dataset(
            df_valid, 
            radius_interval=radius_interval,
            point_density_meters=point_density,
            enable_enhancement=True
        )
        
        if df_enhanced is not None:
            print(f"Enhanced dataset: {len(df_enhanced):,} points from {len(df_valid)} faults")
            # Store enhanced dataset for later use in visualization/interpolation
            df_hyfi._enhanced_point_cloud = df_enhanced
            df_hyfi._enhancement_params = {
                'radius_interval': radius_interval,
                'point_density': point_density,
                'source_faults': len(df_valid)
            }
        else:
            print("Enhanced point cloud generation failed, using single points")
            df_enhanced = df_valid.copy()
            use_enhanced_points = False
    else:
        df_enhanced = df_valid.copy()
    
    # Perform orientation clustering on the original fault data (not enhanced points)
    print("\n--- ORIENTATION CLUSTERING ---")
    df_clustered = _orientation_clustering_optimized(df_valid, input_params)    # Check if spatial clustering is enabled
    spatial_bool = input_params.get('enable_spatial_clustering', True)
    next_counter = starting_fault_counter
    
    if spatial_bool:
        print("\n--- SPATIAL SUB-CLUSTERING ---")
        
        if use_enhanced_points and hasattr(df_hyfi, '_enhanced_point_cloud'):
            print("Using enhanced point cloud for spatial clustering")
            df_clustered, _, next_counter = _spatial_clustering_with_enhanced_points(
                df_clustered, df_hyfi._enhanced_point_cloud, input_params,
                starting_fault_counter, sequence_label, segmentation_level
            )
        else:
            print("Using single hypocenter points for spatial clustering")
            df_clustered, _, next_counter = _spatial_clustering_by_orientation(
                df_clustered, input_params,
                starting_fault_counter, sequence_label, segmentation_level
            )
        
        # Apply post-clustering quality control
        min_events_per_cluster = input_params.get('min_events_per_cluster', 10)
        print("\n--- POST-CLUSTERING QUALITY CONTROL ---")
        df_clustered = _filter_small_clusters(df_clustered, min_events_per_cluster)
    else:
        # No spatial clustering - set spatial_cluster to 0 for all
        df_clustered['spatial_cluster'] = 0
        df_clustered['final_cluster_id'] = df_clustered['orient_cluster'].astype(str)
    
    # Merge clustering results back to the full dataframe
    # Initialize clustering columns with NaN for all rows
    df_hyfi['orient_cluster'] = np.nan
    df_hyfi['spatial_cluster'] = np.nan
    df_hyfi['final_cluster_id'] = None
    df_hyfi['final_cluster_id_local'] = None  # Local numeric ID for individual sequence outputs
    df_hyfi['sequence_label'] = sequence_label
    df_hyfi['segmentation_level'] = segmentation_level
    
    # Update rows that had valid fault plane data
    for col in ['orient_cluster', 'spatial_cluster', 'final_cluster_id']:
        df_hyfi.loc[valid_mask, col] = df_clustered[col].values
    
    # Create local numeric IDs (1, 2, 3) for individual sequence visualization
    unique_fs_ids = df_clustered['final_cluster_id'].dropna().unique()
    fs_to_local = {fs_id: str(i+1) for i, fs_id in enumerate(sorted(unique_fs_ids))}
    df_clustered['final_cluster_id_local'] = df_clustered['final_cluster_id'].map(fs_to_local)
    df_hyfi.loc[valid_mask, 'final_cluster_id_local'] = df_clustered['final_cluster_id_local'].values
    
    # Summary statistics
    n_orientation_clusters = df_clustered['orient_cluster'].nunique()
    n_final_clusters = df_clustered['final_cluster_id'].nunique()
    
    print(f"\n--- CLUSTERING SUMMARY ---")
    print(f"Fault systems (orientation clusters): {n_orientation_clusters}")
    if spatial_bool:
        print(f"Final faults (after spatial sub-clustering): {n_final_clusters}")
        # Debug: show actual final cluster IDs
        final_cluster_counts = df_clustered['final_cluster_id'].value_counts()
        print(f"Final cluster breakdown:")
        for cluster_id, count in final_cluster_counts.items():
            print(f"  {cluster_id}: {count} events")
    print(f"Events clustered: {len(df_valid)} / {len(df_hyfi)}")
    
    # Return empty metadata - will be generated after successful mesh interpolation
    return df_hyfi, [], next_counter


# def auto_classification(input_params, data_output, data_input=None):
#     """
#     Automatic classification of point cloud based on fault orientations.

#     Parameters
#     ----------
#     input_params : DataFrame
#         Input parameters.
#     data_output : DataFrame
#         Output data containing fault plane parameters.
#     data_input : DataFrame, optional
#         Input data containing original hypocenter coordinates.
#         Required for spatial clustering functionality.
#     n_clusters : int
#         Number of expected clusters.
#     algorithm : str
#         Clustering algorithm.
#     rotation : bool
#         Data rotation.

#     Returns
#     -------
#     Output DataFrame with class labels.

#     """

#     # Unpack input parameters from dictionary
#     for key, value in input_params.items():
#         globals()[key] = value
    
#     # Extract key parameters for local use
#     algorithm = input_params.get('algorithm', 'vmf_soft')
#     n_clusters = input_params.get('n_clusters', 2)
#     rotation = input_params.get('rotation', True)
#     auto_determine_clusters = input_params.get('auto_determine_clusters', False)
#     max_clusters = input_params.get('max_clusters', 8)

#     if input_params['autoclass_bool']:
        
#         print('\n')
#         print('='*50)
#         print('AUTOMATIC CLASSIFICATION')
#         print('='*50)

#         # Extract XYZ columns and remove NaN values
#         data = data_output[['nor_x_mean', 'nor_y_mean', 'nor_z_mean']].dropna()
        
#         # Check if we have any valid data for classification
#         if len(data) == 0:
#             print('Warning: No valid fault plane normal vectors found for automatic classification. Skipping classification.')
#             # Add empty class column and return original data
#             data_output['orient_cluster'] = np.nan
#             return data_output
        
#         X, Y, Z = data['nor_x_mean'], data['nor_y_mean'], data['nor_z_mean']
#         data = np.array([X, Y, Z]).T
        
#         # Option to cluster subvertical structures
#         if rotation:
#             # Rotate the data to similar directions if necessary
#             # Ensure that all vectors point to similar direction as first entry
#             v1 = [data[0][0], data[0][1], data[0][2]]
#             v1 = v1 / np.linalg.norm(v1)
#             # Check every point in the dataset and swap direction if it
#             # lies on the other side of the stereoplot
#             # (angular difference larger than 90 degrees)
#             for j in range(len(data)):
#                 vj = [data[j, 0], data[j, 1], data[j, 2]]
#                 vj = vj / np.linalg.norm(vj)
#                 if np.linalg.norm(v1 - vj) == 0:
#                     angle_deg = np.nan
#                 else:
#                     angle_deg = np.degrees(np.arccos(np.dot(v1, vj)))
#                 if angle_deg > 90:
#                     data[j, 0] = data[j, 0] * -1
#                     data[j, 1] = data[j, 1] * -1
#                     data[j, 2] = data[j, 2] * -1
#                 else:
#                     pass

#         # Automatically determine optimal number of clusters if requested
#         if auto_determine_clusters:
#             print("Automatically determining optimal number of clusters...")
#             n_clusters, cluster_results = _determine_optimal_clusters(data, max_clusters, algorithm)
#             print(f"Using {n_clusters} clusters based on automatic determination.")
#         else:
#             print(f"Using specified number of clusters: {n_clusters}")

#         # Apply clustering with the specified estimator
#         if algorithm == 'skm':
#             # Spherical k-Means clustering
#             skm = spherecluster.SphericalKMeans(n_clusters=n_clusters)
#             skm.fit(data)
#             labels = skm.labels_
#             cluster_centers = skm.cluster_centers_                
#         elif algorithm == 'vmf_soft':
#             # Von Mises Fisher Mixture soft clustering
#             vmf_soft = spherecluster.VonMisesFisherMixture(n_clusters=n_clusters, posterior_type='soft')
#             vmf_soft.fit(data)
#             labels = vmf_soft.labels_
#             cluster_centers = vmf_soft.cluster_centers_
#         elif algorithm == 'vmf_hard':
#             # Von Mises Fisher Mixture hard clustering
#             vmf_hard = spherecluster.VonMisesFisherMixture(n_clusters=n_clusters, posterior_type='hard')
#             vmf_hard.fit(data)
#             labels = vmf_hard.labels_
#             cluster_centers = vmf_hard.cluster_centers_
#         else:
#             # Raise error for unsupported algorithm name
#             raise ValueError('Unsupported clustering algorithm: {}'.format(algorithm))
        
#         # Add the orientation cluster labels to data_output
#         data_output.loc[data_output['nor_x_mean'].notna(), 'orient_cluster'] = labels
#         # Reset the index
#         data_output = data_output.reset_index(drop=True)
        
#         # =============================================================================
#         # SPATIAL SUB-CLUSTERING WITHIN ORIENTATION CLUSTERS
#         # =============================================================================
        
#         print(f"\nPerforming spatial sub-clustering within {n_clusters} orientation clusters...")
        
#         # Get spatial clustering parameters
#         spatial_method = input_params.get('spatial_clustering_method', 'dbscan')
#         min_points_spatial = input_params.get('min_points_per_subcluster', 10)
#         enable_spatial_clustering = input_params.get('enable_spatial_clustering', True)
        
#         # Initialize spatial cluster columns
#         data_output['spatial_cluster'] = 0
#         data_output['final_cluster_id'] = data_output['orient_cluster'].astype(str)
        
#         if enable_spatial_clustering:
#             # Check if we have coordinate data available for spatial clustering
#             if data_input is not None and any(col in data_input.columns for col in ['X', 'Y', 'Z']):
#                 print("  Spatial clustering enabled - will use coordinate data from data_input")
                
#                 # Note: We don't merge coordinates here - the spatial clustering will be performed
#                 # by accessing data_input during the clustering process
                
#             else:
#                 print("  Warning: No coordinate data available - spatial clustering will be disabled")
#                 enable_spatial_clustering = False
            
#             # Process each orientation cluster for spatial sub-clustering
#             if enable_spatial_clustering:
#                 unique_orientation_clusters = data_output['orient_cluster'].dropna().unique()
                
#                 for ori_cluster in unique_orientation_clusters:
#                     ori_cluster_data = data_output[data_output['orient_cluster'] == ori_cluster].reset_index(drop=True)
                    
#                     if len(ori_cluster_data) < min_points_spatial:
#                         print(f"  Orientation cluster {ori_cluster}: too few points ({len(ori_cluster_data)}) - no spatial sub-clustering")
#                         continue
                    
#                     print(f"  Processing orientation cluster {ori_cluster} ({len(ori_cluster_data)} fault planes)...")
                    
#                     # For spatial clustering, we need to merge coordinate data temporarily
#                     if data_input is not None and 'ID' in data_input.columns and 'ID' in data_output.columns:
#                         # Get coordinate columns from data_input
#                         coord_cols_input = None
#                         for coord_set in [['X', 'Y', 'Z']]:
#                             if all(col in data_input.columns for col in coord_set):
#                                 coord_cols_input = coord_set
#                                 break
                        
#                         if coord_cols_input:
#                             # Merge coordinates for this specific cluster only
#                             coord_data = data_input[['ID'] + coord_cols_input].copy()
#                             ori_cluster_data_with_coords = ori_cluster_data.merge(coord_data, on='ID', how='left')
                            
#                             # Apply spatial clustering on the merged data
#                             try:
#                                 clustered_data = _spatial_clustering(
#                                     ori_cluster_data_with_coords, 
#                                     method=spatial_method, 
#                                     n_clusters=2,  # Not used for DBSCAN
#                                     min_points=min_points_spatial
#                                 )
                                
#                                 spatial_clusters = clustered_data['spatial_cluster'].unique()
                                
#                                 # Evaluate if spatial clustering is beneficial
#                                 use_spatial_clustering = _evaluate_spatial_clustering_benefit(
#                                     clustered_data, spatial_clusters, min_points_spatial
#                                 )
                                
#                                 if use_spatial_clustering:
#                                     valid_spatial_clusters = [sc for sc in spatial_clusters if sc != -1]
#                                     print(f"    ✓ Spatial clustering beneficial: split into {len(valid_spatial_clusters)} sub-clusters")
                                    
#                                     # Update spatial cluster labels in main dataframe
#                                     ori_indices = data_output[data_output['orient_cluster'] == ori_cluster].index
#                                     data_output.loc[ori_indices, 'spatial_cluster'] = clustered_data['spatial_cluster'].values
                                    
#                                     # Create final cluster IDs (e.g., "0_0", "0_1", "1_0", etc.)
#                                     for spatial_id in valid_spatial_clusters:
#                                         spatial_indices = ori_indices[clustered_data['spatial_cluster'] == spatial_id]
#                                         final_id = f"{int(ori_cluster)}_{int(spatial_id)}"
#                                         data_output.loc[spatial_indices, 'final_cluster_id'] = final_id
                                        
#                                 else:
#                                     print(f"    × Spatial clustering not beneficial: keeping as single cluster")
#                                     # Keep as single spatial cluster (already initialized as 0)
                                    
#                             except Exception as e:
#                                 print(f"    Warning: Spatial clustering failed for orientation cluster {ori_cluster}: {e}")
#                                 # Keep as single spatial cluster
#                         else:
#                             print(f"    Warning: Cannot access coordinate data for spatial clustering")
#                     else:
#                         print(f"    Warning: Missing ID or coordinate columns for spatial clustering")
                
#                 print(f"✓ Spatial sub-clustering completed")
                
#                 # Print summary of final clusters
#                 final_clusters = data_output['final_cluster_id'].value_counts()
#                 print(f"Final fault clusters created:")
#                 for cluster_id, count in final_clusters.items():
#                     print(f"  Cluster {cluster_id}: {count} fault planes")
                    
#         else:
#             print("Spatial clustering disabled - using orientation clusters only")
        
#         # =============================================================================
#         # PRINT CLUSTER STATISTICS
#         # =============================================================================
        
#         # Print mean directions of each class
#         # Rotate to lower hemisphere if cluster center lies on upper hemisphere
#         def rotation_lowerhemi(nor_x, nor_y, nor_z):
#             if nor_z > 0:
#                 nor_x = nor_x * -1
#                 nor_y = nor_y * -1
#                 nor_z = nor_z * -1
#             else:
#                 pass
            
#             azi, dip = utilities.plane_normal_to_azidip(nor_x, nor_y, nor_z)
#             return azi, dip

#         for q in range(len(cluster_centers)):
#             nor_x = cluster_centers[q][0]
#             nor_y = cluster_centers[q][1]
#             nor_z = cluster_centers[q][2]
            
#             azi, dip = rotation_lowerhemi(nor_x, nor_y, nor_z)
            
#             print(f'Mean fault orientation class {q}: ', azi, '/', dip)

#             # ## Estimate the confidence angle a95
#             # # Extract all planes from class q
#             # df_q = data_output.loc[data_output['orient_cluster'] == q]
#             # df_q = df_q.reset_index(drop=True)
#             # nor_x_list = np.array(df_q['nor_x_mean'])
#             # nor_y_list = np.array(df_q['nor_y_mean'])
#             # nor_z_list = np.array(df_q['nor_z_mean'])
#             # # Calculate the direction of the pole to the first plane
#             # nor_x_1 = nor_x[np.isfinite(nor_x)][0]
#             # nor_y_1 = nor_y[np.isfinite(nor_y)][0]
#             # nor_z_1 = nor_z[np.isfinite(nor_z)][0]
#             # v1 = [nor_x_1, nor_y_1, nor_z_1]
#             # v1 = v1 / np.linalg.norm(v1)
#             # # Check every point in the dataset and swap direction if it
#             # # lies on the other side of the stereoplot
#             # # (angular difference larger than 90 degrees)
#             # for j in range(len(nor_x_list)):
#             #     vj = [nor_x_list[j], nor_y_list[j], nor_z_list[j]]
#             #     vj = vj / np.linalg.norm(vj)
#             #     if np.linalg.norm(v1 - vj) == 0:
#             #         angle_deg = np.nan
#             #     else:
#             #         angle_deg = np.degrees(np.arccos(np.dot(v1, vj)))
#             #     if angle_deg > 90:
#             #         nor_x_list[j] = nor_x_list[j] * -1
#             #         nor_y_list[j] = nor_y_list[j] * -1
#             #         nor_z_list[j] = nor_z_list[j] * -1
#             #     else:
#             #         pass
#             # # Calculate R, N and confidence angle (Borradaile 2003)
#             # N = len(df_q)
#             # v_sum = np.array([np.nansum(nor_x_list), np.nansum(nor_y_list), np.nansum(nor_z_list)])
#             # R = np.linalg.norm(v_sum)
#             # p = 0.05            # confidence = 1-p
#             # a95 = np.arccos(1-((N-R)/R)*((1/p)**(1/(N-1)-1)))
            
#     else:
#         data_output['orient_cluster'] = np.nan

#     return(data_output)

