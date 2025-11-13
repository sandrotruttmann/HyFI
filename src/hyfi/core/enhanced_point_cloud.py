"""
Enhanced Point Cloud Generation for HyFI Workflow

This module provides functionality to generate and manage enhanced fault plane point clouds
that can be used consistently across clustering, visualization, and interpolation processes.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def generate_enhanced_fault_dataset(df_hyfi: pd.DataFrame, 
                                   radius_interval: float = 20.0,
                                   point_density_meters: float = 12.0,
                                   enable_enhancement: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Generate an enhanced fault dataset with fault plane points for improved analysis.
    
    This function creates a comprehensive dataset that includes both the original hypocenter
    data and an enhanced point cloud with multiple points per fault plane. The enhanced
    dataset can be used consistently across clustering, visualization, and interpolation.
    
    Parameters
    ----------
    df_hyfi : pd.DataFrame
        Original fault dataset with hypocenter and fault plane data
    radius_interval : float, default=20.0
        Distance between concentric circles on fault planes (meters)
    point_density_meters : float, default=12.0
        Target distance between points on fault plane circumference (meters)
    enable_enhancement : bool, default=True
        Whether to generate enhanced point cloud or return original data only
        
    Returns
    -------
    df_original : pd.DataFrame
        Original dataset with additional metadata about enhancement
    df_enhanced : pd.DataFrame or None
        Enhanced dataset with fault plane points, or None if enhancement disabled
        Contains additional columns:
        - 'source_fault_idx': Index of the original fault this point belongs to
        - 'point_type': 'center' for hypocenter, 'fault_plane' for generated points
        - 'ring_index': Ring number on fault plane (0=center, 1=first ring, etc.)
        - 'point_index_on_ring': Point index within each ring
    """
    
    if not enable_enhancement:
        print("Enhanced point cloud generation disabled")
        df_enhanced = df_hyfi.copy()
        df_enhanced['source_fault_idx'] = df_enhanced.index
        df_enhanced['point_type'] = 'center'
        df_enhanced['ring_index'] = 0
        df_enhanced['point_index_on_ring'] = 0
        return df_hyfi, df_enhanced
    
    # Check required columns
    required_cols = ['nor_x_mean', 'nor_y_mean', 'nor_z_mean', 'rupt_r', 'X', 'Y', 'Z']
    missing_cols = [col for col in required_cols if col not in df_hyfi.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns for enhanced point generation: {missing_cols}")
        print("Falling back to original dataset only")
        return df_hyfi, None
    
    # Filter out faults with missing data
    valid_mask = (df_hyfi[required_cols].notna().all(axis=1))
    df_valid = df_hyfi[valid_mask].copy()
    
    if len(df_valid) == 0:
        print("No valid fault data for enhanced point generation")
        return df_hyfi, None
    
    print(f"Generating enhanced point cloud for {len(df_valid)} faults...")
    print(f"  Parameters: radius_interval={radius_interval}m, point_density={point_density_meters}m")
    
    # Estimate point count
    total_estimated_points = 0
    for _, row in df_valid.iterrows():
        r = row['rupt_r']
        n_rings = max(1, int(r / radius_interval))
        avg_points_per_ring = max(1, int(2 * np.pi * (r/2) / point_density_meters))
        total_estimated_points += 1 + n_rings * avg_points_per_ring
    
    print(f"  Estimated enhanced points: {total_estimated_points:,} ({total_estimated_points/len(df_valid):.0f} per fault)")
    
    # Safety check for very large point clouds
    if total_estimated_points > 200000:
        print(f"  Warning: {total_estimated_points:,} points is very large. Consider increasing point_density_meters.")
        # Automatically adjust parameters to reduce count
        scale_factor = np.sqrt(200000 / total_estimated_points)
        point_density_meters = point_density_meters / scale_factor
        radius_interval = radius_interval / scale_factor
        print(f"  Auto-adjusted: radius_interval={radius_interval:.1f}m, point_density={point_density_meters:.1f}m")
    
    # Generate enhanced points
    enhanced_points = []
    
    for fault_idx, (orig_idx, row) in enumerate(df_valid.iterrows()):
        # Fault parameters
        center = np.array([row['X'], row['Y'], row['Z']])
        normal = np.array([row['nor_x_mean'], row['nor_y_mean'], row['nor_z_mean']])
        normal = normal / np.linalg.norm(normal)  # Normalize
        radius = row['rupt_r']
        
        # Add the hypocenter (center point)
        center_point = row.copy()
        center_point['source_fault_idx'] = orig_idx
        center_point['point_type'] = 'center'
        center_point['ring_index'] = 0
        center_point['point_index_on_ring'] = 0
        enhanced_points.append(center_point)
        
        # Generate fault plane points in concentric rings
        if radius > 0:
            n_rings = max(1, int(radius / radius_interval))
            
            for ring_idx in range(1, n_rings + 1):
                ring_radius = ring_idx * radius_interval
                
                # Don't exceed fault radius
                if ring_radius > radius:
                    ring_radius = radius
                
                # Calculate number of points on this ring
                ring_circumference = 2 * np.pi * ring_radius
                n_points_on_ring = max(3, int(ring_circumference / point_density_meters))
                
                # Generate points around the ring
                for point_idx in range(n_points_on_ring):
                    angle = 2 * np.pi * point_idx / n_points_on_ring
                    
                    # Create two orthogonal vectors in the fault plane
                    # Find a vector perpendicular to normal
                    if abs(normal[2]) < 0.9:
                        v1 = np.cross(normal, [0, 0, 1])
                    else:
                        v1 = np.cross(normal, [1, 0, 0])
                    v1 = v1 / np.linalg.norm(v1)
                    
                    # Second orthogonal vector
                    v2 = np.cross(normal, v1)
                    v2 = v2 / np.linalg.norm(v2)
                    
                    # Point on fault plane
                    offset = ring_radius * (np.cos(angle) * v1 + np.sin(angle) * v2)
                    point_coords = center + offset
                    
                    # Create point entry
                    point_data = row.copy()
                    point_data['X'] = point_coords[0]
                    point_data['Y'] = point_coords[1] 
                    point_data['Z'] = point_coords[2]
                    point_data['source_fault_idx'] = orig_idx
                    point_data['point_type'] = 'fault_plane'
                    point_data['ring_index'] = ring_idx
                    point_data['point_index_on_ring'] = point_idx
                    
                    enhanced_points.append(point_data)
    
    # Create enhanced DataFrame
    df_enhanced = pd.DataFrame(enhanced_points)
    df_enhanced = df_enhanced.reset_index(drop=True)
    
    print(f"  Generated {len(df_enhanced):,} enhanced points ({len(df_enhanced)/len(df_valid):.0f} per fault)")
    
    # Add enhancement metadata to original dataset
    df_original = df_hyfi.copy()
    df_original['has_enhanced_points'] = df_original.index.isin(df_valid.index)
    df_original['enhancement_params'] = f"radius_interval={radius_interval:.1f}m_density={point_density_meters:.1f}m"
    
    return df_original, df_enhanced


def get_enhanced_coordinates(df_enhanced: pd.DataFrame, 
                           coordinate_columns: list = ['X', 'Y', 'Z']) -> np.ndarray:
    """
    Extract coordinate array from enhanced dataset.
    
    Parameters
    ----------
    df_enhanced : pd.DataFrame
        Enhanced dataset from generate_enhanced_fault_dataset
    coordinate_columns : list, default=['X', 'Y', 'Z']
        Column names for coordinates
        
    Returns
    -------
    np.ndarray
        Coordinate array of shape (n_points, 3)
    """
    return df_enhanced[coordinate_columns].values


def get_fault_mapping(df_enhanced: pd.DataFrame) -> np.ndarray:
    """
    Get mapping from enhanced points back to original faults.
    
    Parameters
    ----------
    df_enhanced : pd.DataFrame
        Enhanced dataset from generate_enhanced_fault_dataset
        
    Returns
    -------
    np.ndarray
        Array where each element is the original fault index for the corresponding enhanced point
    """
    return df_enhanced['source_fault_idx'].values


def aggregate_cluster_labels_to_faults(enhanced_labels: np.ndarray, 
                                     fault_mapping: np.ndarray) -> dict:
    """
    Aggregate cluster labels from enhanced points back to original faults.
    
    Parameters
    ----------
    enhanced_labels : np.ndarray
        Cluster labels for each enhanced point
    fault_mapping : np.ndarray
        Mapping from enhanced points to original faults (from get_fault_mapping)
        
    Returns
    -------
    dict
        Dictionary mapping original fault indices to cluster labels
    """
    from collections import Counter
    
    fault_clusters = {}
    
    for fault_idx in np.unique(fault_mapping):
        # Get all labels for points belonging to this fault
        point_mask = fault_mapping == fault_idx
        point_labels = enhanced_labels[point_mask]
        
        # Remove noise points (-1) for voting
        valid_labels = point_labels[point_labels != -1]
        
        if len(valid_labels) > 0:
            # Use majority vote
            label_counts = Counter(valid_labels)
            fault_clusters[fault_idx] = label_counts.most_common(1)[0][0]
        else:
            # All points were noise
            fault_clusters[fault_idx] = -1
    
    return fault_clusters
