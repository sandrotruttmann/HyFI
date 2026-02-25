#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS: Visualisation Module

Please cite: Truttmann et al. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps.

@author: Sandro Truttmann
@contact: sandro.truttmann@gmail.com
@license: GPL-3.0
@date: April 2023
@version: 0.1.1
"""

# Import the needed modules
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import math
import mplstereonet
from ..utils import utilities
from ..utils import utilities_plot
import matplotlib.pyplot as plt
import os
import pyvista as pv
import open3d as o3d
from numba import jit, prange


# =============================================================================
# NUMBA-OPTIMIZED GEOMETRIC COMPUTATIONS
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def _normalize_vector(v):
    """Normalize a 3D vector (numba-optimized)."""
    norm = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if norm > 0:
        return v / norm
    return v

@jit(nopython=True, cache=True, fastmath=True)
def _cross_product(a, b):
    """Compute cross product a × b (numba-optimized)."""
    result = np.empty(3, dtype=np.float64)
    result[0] = a[1]*b[2] - a[2]*b[1]
    result[1] = a[2]*b[0] - a[0]*b[2]
    result[2] = a[0]*b[1] - a[1]*b[0]
    return result

@jit(nopython=True, cache=True, fastmath=True)
def _generate_disc_vertices_and_faces(center, normal, radius, n_segments):
    """
    Generate vertices and faces for a circular disc with 2 rings (inner + outer).
    Numba-optimized for speed.
    
    Parameters
    ----------
    center : ndarray(3,)
        Center point (hypocenter)
    normal : ndarray(3,)
        Unit normal vector to the plane
    radius : float
        Rupture radius
    n_segments : int
        Number of angular segments
        
    Returns
    -------
    vertices : ndarray(n_vertices, 3)
        Vertex coordinates
    faces : ndarray(n_faces, 3)
        Face vertex indices
    """
    # Create orthonormal basis vectors in the plane
    v1 = np.empty(3, dtype=np.float64)
    v2 = np.empty(3, dtype=np.float64)
    
    # Choose reference vector based on normal orientation
    if abs(normal[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    
    # v1 = normal × ref
    v1 = _cross_product(normal, ref)
    v1 = _normalize_vector(v1)
    
    # v2 = normal × v1
    v2 = _cross_product(normal, v1)
    v2 = _normalize_vector(v2)
    
    # Pre-allocate vertex array: center + inner ring + outer ring
    n_vertices = 1 + 2 * n_segments
    vertices = np.empty((n_vertices, 3), dtype=np.float64)
    
    # Center point
    vertices[0] = center
    
    # Inner ring (half radius)
    inner_radius = radius / 2.0
    for i in range(n_segments):
        angle = 2.0 * np.pi * i / n_segments
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Point in local plane coordinates
        local_x = inner_radius * cos_a
        local_y = inner_radius * sin_a
        
        # Transform to global coordinates
        vertices[1 + i, 0] = center[0] + local_x * v1[0] + local_y * v2[0]
        vertices[1 + i, 1] = center[1] + local_x * v1[1] + local_y * v2[1]
        vertices[1 + i, 2] = center[2] + local_x * v1[2] + local_y * v2[2]
    
    # Outer ring (full radius)
    for i in range(n_segments):
        angle = 2.0 * np.pi * i / n_segments
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Point in local plane coordinates
        local_x = radius * cos_a
        local_y = radius * sin_a
        
        # Transform to global coordinates
        idx = 1 + n_segments + i
        vertices[idx, 0] = center[0] + local_x * v1[0] + local_y * v2[0]
        vertices[idx, 1] = center[1] + local_x * v1[1] + local_y * v2[1]
        vertices[idx, 2] = center[2] + local_x * v1[2] + local_y * v2[2]
    
    # Pre-allocate face array: center→inner (n_segments) + inner→outer (2*n_segments)
    n_faces = 3 * n_segments
    faces = np.empty((n_faces, 3), dtype=np.int32)
    
    # Faces from center to inner ring
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces[i, 0] = 0
        faces[i, 1] = 1 + i
        faces[i, 2] = 1 + next_i
    
    # Faces between inner and outer rings (2 triangles per segment)
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        
        # Inner ring vertex indices
        p1 = 1 + i
        p2 = 1 + next_i
        
        # Outer ring vertex indices
        p3 = 1 + n_segments + i
        p4 = 1 + n_segments + next_i
        
        # Triangle 1
        face_idx = n_segments + 2 * i
        faces[face_idx, 0] = p1
        faces[face_idx, 1] = p3
        faces[face_idx, 2] = p2
        
        # Triangle 2
        face_idx = n_segments + 2 * i + 1
        faces[face_idx, 0] = p2
        faces[face_idx, 1] = p3
        faces[face_idx, 2] = p4
    
    return vertices, faces


# =============================================================================
# POISSON SURFACE RECONSTRUCTION UTILITIES
# =============================================================================


def _compute_normal_vectors(df_subcluster):
    """
    Compute normal vectors from azimuth and dip values.
    
    Parameters
    ----------
    df_subcluster : DataFrame
        DataFrame containing 'rupt_plane_azi' and 'rupt_plane_dip' columns
        
    Returns
    -------
    numpy.ndarray
        Array of normal vectors (n_points, 3)
    """
    azi = np.radians(df_subcluster['rupt_plane_azi'].values)
    dip = np.radians(df_subcluster['rupt_plane_dip'].values)
    
    normals_x = np.sin(dip) * np.sin(azi)
    normals_y = np.sin(dip) * np.cos(azi)
    normals_z = np.cos(dip)
    
    return np.column_stack((normals_x, normals_y, normals_z))


def _evaluate_orientation_consistency(df_cluster, max_angular_deviation=45):
    """
    Evaluate whether fault plane orientations within a cluster are consistent enough
    for interpolation.
    
    Parameters
    ----------
    df_cluster : DataFrame
        DataFrame containing fault plane data with 'rupt_plane_azi' and 'rupt_plane_dip' columns
    max_angular_deviation : float
        Maximum allowed angular deviation from cluster mean (degrees)
        
    Returns
    -------
    tuple
        (is_consistent, angular_spread, mean_azi, mean_dip)
        - is_consistent: bool indicating if orientations are consistent
        - angular_spread: float indicating the max angular deviation (degrees)
        - mean_azi, mean_dip: mean orientation of the cluster
    """
    
    if len(df_cluster) < 2:
        return True, 0.0, df_cluster['rupt_plane_azi'].iloc[0], df_cluster['rupt_plane_dip'].iloc[0]
    
    # Convert orientations to unit normal vectors (using existing normal vectors if available)
    if all(col in df_cluster.columns for col in ['nor_x_mean', 'nor_y_mean', 'nor_z_mean']):
        # Use existing normal vectors
        normals = df_cluster[['nor_x_mean', 'nor_y_mean', 'nor_z_mean']].values
    else:
        # Calculate normal vectors from azimuth/dip
        azi_rad = np.radians(df_cluster['rupt_plane_azi'].values)
        dip_rad = np.radians(df_cluster['rupt_plane_dip'].values)
        
        normals_x = np.sin(dip_rad) * np.sin(azi_rad)
        normals_y = np.sin(dip_rad) * np.cos(azi_rad)
        normals_z = np.cos(dip_rad)
        normals = np.column_stack((normals_x, normals_y, normals_z))
    
    # Normalize all normal vectors
    for i in range(len(normals)):
        normals[i] = normals[i] / np.linalg.norm(normals[i])
    
    # Rotate poles to consistent hemisphere (critical for subvertical planes)
    # Ensure that all vectors point to similar direction as first entry
    v1 = normals[0]
    for j in range(1, len(normals)):
        vj = normals[j]
        # Calculate angle between vectors
        cos_angle = np.clip(np.dot(v1, vj), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(abs(cos_angle)))
        
        # If angle > 90°, the vectors point to opposite hemispheres
        if angle_deg > 90:
            normals[j] = -normals[j]
    
    # Calculate mean normal vector
    mean_normal = np.mean(normals, axis=0)
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    
    # Calculate angular deviations from mean
    angular_deviations = []
    for normal in normals:
        cos_angle = np.clip(np.dot(normal, mean_normal), -1.0, 1.0)
        angle_rad = np.arccos(abs(cos_angle))
        angle_deg = np.degrees(angle_rad)
        angular_deviations.append(angle_deg)
    
    angular_spread = np.max(angular_deviations)
    
    # Convert mean normal back to azimuth/dip
    if abs(mean_normal[2]) > 0.999:  # Nearly vertical
        mean_azi = 0.0
        mean_dip = 90.0 if mean_normal[2] > 0 else -90.0
    else:
        mean_azi = np.degrees(np.arctan2(mean_normal[0], mean_normal[1])) % 360
        mean_dip = np.degrees(np.arccos(abs(mean_normal[2])))
    
    is_consistent = angular_spread <= max_angular_deviation
    
    return is_consistent, angular_spread, mean_azi, mean_dip


def _get_vtp_column_name_mapping():
    """Get column name mapping to match HyFI_results.csv output names."""
    return {
        'rupt_plane_azi': 'rupture_plane_azimuth',
        'rupt_plane_dip': 'rupture_plane_dip',
        'Sn_eff': 'effective_normal_stress',
        'Tau': 'shear_stress',
        'rake': 'rake_angle',
        'instab': 'instability_index',
        'sliptend': 'slip_tendency',
        'dilatend': 'dilation_tendency'
    }

def _add_attributes_to_vtp(vtp_object, data_source, column_list=None, exclude_cols=None):
    """Add attributes to VTP object with proper column name mapping.
    
    Parameters
    ----------
    vtp_object : pyvista.PolyData
        VTP object to add attributes to
    data_source : DataFrame or Series
        Source data (single row Series or full DataFrame)
    column_list : list, optional
        Specific columns to add. If None, adds all columns.
    exclude_cols : list, optional
        Columns to exclude
    """
    rename_map = _get_vtp_column_name_mapping()
    exclude_cols = exclude_cols or []
    
    if column_list is None:
        column_list = data_source.columns if hasattr(data_source, 'columns') else data_source.index
    
    for col in column_list:
        if col in exclude_cols:
            continue
        if col in data_source:
            try:
                # Use renamed column name for VTP attribute
                vtp_col_name = rename_map.get(col, col)
                values = data_source[col] if hasattr(data_source, 'columns') else data_source[col]
                vtp_object[vtp_col_name] = values
            except Exception as e:
                pass

def _generate_fault_plane_points(df_subcluster, radius_interval=10.0, point_density_meters=10.0, max_points=100000):
    """
    Generate point clouds from circular fault plane geometries with systematic spatial coverage.
    
    This function uses the calculated fault plane information (center, radius, normal)
    to generate multiple points on each circular fault plane with homogeneous spatial
    distribution, providing richer geometric information for Poisson surface reconstruction.
    
    The point distribution strategy:
    1. Center point (hypocenter)
    2. Complete circles at fixed radius intervals (configurable spacing)
       - Each circle has points distributed based on circumference for consistent density
    
    Parameters
    ----------
    df_subcluster : DataFrame
        DataFrame containing fault plane data with columns:
        'X', 'Y', 'Z' (hypocenter coordinates), 'rupt_radius' (rupture radius),
        'nor_x_mean', 'nor_y_mean', 'nor_z_mean' (normal vectors)
    radius_interval : float
        Fixed interval in meters between concentric circles (default: 20.0)
    point_density_meters : float
        Target distance in meters between points on each circle circumference (default: 12.0)
        Smaller values create denser point clouds, larger values create sparser ones
    max_points : int
        Maximum total points to generate. If the estimated count exceeds this limit,
        radius_interval and point_density_meters are scaled up proportionally to stay
        within the limit. Default: 100000.
        
    Returns
    -------
    tuple
        (points, normals, rupture_radii, is_hypocenter, event_indices) - numpy arrays
        - points: (n_total_points, 3) coordinates
        - normals: (n_total_points, 3) normal vectors
        - rupture_radii: (n_total_points,) rupture radius for each point
        - is_hypocenter: (n_total_points,) 1 for hypocenter, 0 for enhanced points
        - event_indices: (n_total_points,) index into df_subcluster for each point
        where n_total_points = len(df_subcluster) * actual_points_per_plane
    """
    import sys as _sys
    import time as _time
    _n_total = len(df_subcluster)
    _start   = _time.time()
    _sys.stderr.write(f"    Generating fault plane point cloud for {_n_total} events...\n")

    # Extract numpy arrays upfront – much faster than iterrows
    X_arr  = df_subcluster['X'].values.astype(np.float64)
    Y_arr  = df_subcluster['Y'].values.astype(np.float64)
    Z_arr  = df_subcluster['Z'].values.astype(np.float64)
    R_arr  = df_subcluster['rupt_radius'].values.astype(np.float64)
    NX_arr = df_subcluster['nor_x_mean'].values.astype(np.float64)
    NY_arr = df_subcluster['nor_y_mean'].values.astype(np.float64)
    NZ_arr = df_subcluster['nor_z_mean'].values.astype(np.float64)
    _sys.stderr.write(f"    Arrays extracted. Computing valid mask...\n")

    # Valid mask (vectorised, avoids per-row isnan checks in the loop)
    valid_mask = (
        np.isfinite(NX_arr) & np.isfinite(NY_arr) & np.isfinite(NZ_arr) &
        np.isfinite(R_arr) & (R_arr > 0)
    )
    _sys.stderr.write(f"    Valid events: {valid_mask.sum()} / {_n_total}\n")

    # --- Vectorised pre-estimation and auto-scale ---
    valid_radii_arr = R_arr[valid_mask]
    _r_cap = np.inf  # fallback: no cap
    if len(valid_radii_arr) > 0:
        # Use 99th-percentile radius for estimation – np.max picks up corrupted/outlier values
        r_max_est = np.percentile(valid_radii_arr, 99)
        # Hard cap: treat any radius more than 100x the 99th-percentile as a corrupt value
        _r_cap = r_max_est * 100.0
        _n_outliers = int(np.sum(valid_radii_arr > _r_cap))
        _sys.stderr.write(f"    Rupture radius: median={np.median(valid_radii_arr):.1f} m, "
                          f"99pct={r_max_est:.1f} m, max={np.max(valid_radii_arr):.3g} m, "
                          f"radius_interval={radius_interval:.1f} m "
                          f"(outliers >100x 99pct: {_n_outliers})\n")
        n_rings_max = min(50, max(1, int(r_max_est / radius_interval)) + 1)  # cap rings in estimation too
        pts_per_event_max = 1 + sum(
            max(8, int(2 * np.pi * min(k * radius_interval, r_max_est) / point_density_meters))
            for k in range(1, n_rings_max + 1)
        )
        total_estimated = int(pts_per_event_max * len(valid_radii_arr))
        _sys.stderr.write(f"    Estimated total points: {total_estimated:,} (max {pts_per_event_max} per event)\n")
        if total_estimated > max_points:
            scale_factor = np.sqrt(total_estimated / max_points)
            radius_interval = radius_interval * scale_factor
            point_density_meters = point_density_meters * scale_factor
            _sys.stderr.write(f"    ⚠ Auto-scaling: radius_interval={radius_interval:.1f}m, point_density={point_density_meters:.1f}m\n")

    all_parts_pts  = []
    all_parts_nor  = []
    all_parts_rad  = []
    all_parts_hypo = []
    all_parts_idx  = []
    _pts_so_far = 0  # running counter (avoids slow sum(len(p)...) on every progress print)
    _sys.stderr.write(f"    Starting event loop...\n")

    for event_idx in range(_n_total):
        if event_idx > 0 and event_idx % 2000 == 0:
            _elapsed = _time.time() - _start
            _rate    = event_idx / _elapsed
            _eta     = (_n_total - event_idx) / _rate if _rate > 0 else 0
            _sys.stderr.write(f"    Point cloud progress: {event_idx}/{_n_total} events "
                  f"({_rate:.0f} ev/s, ETA {_eta:.0f}s, {_pts_so_far:,} pts so far)...\n")

        if not valid_mask[event_idx]:
            continue

        p   = np.array([X_arr[event_idx], Y_arr[event_idx], Z_arr[event_idx]])
        r   = R_arr[event_idx]

        # Skip outlier radii that would cause OOM (corrupted values)
        if r > _r_cap:
            _sys.stderr.write(f"      Skipping event {event_idx}: outlier radius {r:.3g} m > cap {_r_cap:.3g} m\n")
            continue

        nor = np.array([NX_arr[event_idx], NY_arr[event_idx], NZ_arr[event_idx]])
        nor = nor / np.linalg.norm(nor)

        # Orthonormal basis in the fault plane
        if abs(nor[2]) < 0.9:
            v1 = np.cross(nor, np.array([0.0, 0.0, 1.0]))
        else:
            v1 = np.cross(nor, np.array([1.0, 0.0, 0.0]))
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(nor, v1)

        # Ring radii (vectorised arange + edge)
        # Cap at 50 rings to prevent runaway on large-radius events
        _MAX_RINGS = 50
        _ri = radius_interval if r / radius_interval <= _MAX_RINGS else r / _MAX_RINGS
        radii = np.arange(_ri, r + 1e-9, _ri)
        if len(radii) == 0 or radii[-1] < r - 1e-9:
            radii = np.append(radii, r)

        # --- Vectorised ring generation ---
        ring_pts_list  = [p.reshape(1, 3)]           # center point
        ring_hypo_list = [np.ones(1, dtype=np.int8)] # center is a hypocenter

        for ring_radius in radii:
            n_pts = max(8, int(2 * np.pi * ring_radius / point_density_meters))
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            # Vectorised: (n_pts, 3) via broadcasting
            pts = p + ring_radius * (np.cos(angles)[:, None] * v1 + np.sin(angles)[:, None] * v2)
            ring_pts_list.append(pts)
            ring_hypo_list.append(np.zeros(n_pts, dtype=np.int8))

        plane_pts  = np.concatenate(ring_pts_list,  axis=0)  # (n_plane_pts, 3)
        plane_hypo = np.concatenate(ring_hypo_list)          # (n_plane_pts,)
        n_plane    = len(plane_pts)

        all_parts_pts.append(plane_pts)
        all_parts_nor.append(np.tile(nor, (n_plane, 1)))
        all_parts_rad.append(np.full(n_plane, r))
        all_parts_hypo.append(plane_hypo)
        all_parts_idx.append(np.full(n_plane, event_idx, dtype=np.int32))
        _pts_so_far += n_plane
        _sys.stderr.write(f"      event {event_idx+1}/{_n_total}: r={r:.1f}m, {len(radii)} rings → {n_plane} pts (total: {_pts_so_far:,})\n")

    if not all_parts_pts:
        return (np.empty((0, 3)), np.empty((0, 3)), np.empty(0),
                np.empty(0, dtype=np.int8), np.empty(0, dtype=np.int32))

    out_pts = np.concatenate(all_parts_pts, axis=0)
    _sys.stderr.write(f"    ✓ Point cloud complete: {len(out_pts):,} points from {_n_total} events "
          f"(took {_time.time() - _start:.1f}s)\n")
    return (
        out_pts,
        np.concatenate(all_parts_nor,  axis=0),
        np.concatenate(all_parts_rad),
        np.concatenate(all_parts_hypo),
        np.concatenate(all_parts_idx),
    )


def _create_circular_fault_disc_meshes(df_subcluster, n_radial_segments=16):
    """
    Create circular disc meshes for fault planes (minimal, efficient for planar surfaces).
    
    Uses numba-optimized geometric computations for speed.
    
    This function creates triangular mesh representations of the circular fault planes
    as discs using minimal geometry: outer ring + 1 inner ring + center. This is optimal
    for planar surfaces where we only need to define the disc boundary and one intermediate ring.
    
    Parameters
    ----------
    df_subcluster : DataFrame
        DataFrame containing fault plane data with columns:
        'X', 'Y', 'Z' (hypocenter coordinates), 'rupt_radius' (rupture radius),
        'nor_x_mean', 'nor_y_mean', 'nor_z_mean' (normal vectors)
    n_radial_segments : int
        Number of angular segments around each ring (default: 16)
        
    Returns
    -------
    list
        List of PyVista PolyData meshes, one for each fault plane
    """
    
    import sys as _dsys
    import time as _dtime

    total_events = len(df_subcluster)

    # Auto-reduce angular resolution for large clusters
    if total_events > 2000 and n_radial_segments > 8:
        n_radial_segments = 8
        _dsys.stderr.write(f"  Auto-reducing disc segments to {n_radial_segments} for large cluster ({total_events} events)\n")
    elif total_events > 1000 and n_radial_segments > 12:
        n_radial_segments = 12
        _dsys.stderr.write(f"  Auto-reducing disc segments to {n_radial_segments} for medium-large cluster ({total_events} events)\n")

    _dsys.stderr.write(f"  Generating circular disc meshes for {total_events} events "
                       f"({n_radial_segments} segments) — pre-allocated single-mesh mode...\n")
    _t0 = _dtime.time()

    # --- Pre-extract ALL data as numpy arrays before the loop ---
    X_arr  = df_subcluster['X'].values.astype(np.float64)
    Y_arr  = df_subcluster['Y'].values.astype(np.float64)
    Z_arr  = df_subcluster['Z'].values.astype(np.float64)
    R_arr  = df_subcluster['rupt_radius'].values.astype(np.float64)
    NX_arr = df_subcluster['nor_x_mean'].values.astype(np.float64)
    NY_arr = df_subcluster['nor_y_mean'].values.astype(np.float64)
    NZ_arr = df_subcluster['nor_z_mean'].values.astype(np.float64)

    # Valid mask (including outlier radius guard)
    valid_mask_d = (
        np.isfinite(NX_arr) & np.isfinite(NY_arr) & np.isfinite(NZ_arr) &
        np.isfinite(R_arr)  & (R_arr > 0)
    )
    # Compute 99th-percentile radius cap to exclude corrupted values
    _valid_r = R_arr[valid_mask_d]
    if len(_valid_r) > 0:
        _r99 = np.percentile(_valid_r, 99)
        _disc_r_cap = _r99 * 100.0
        _n_disc_outliers = int(np.sum(_valid_r > _disc_r_cap))
        if _n_disc_outliers > 0:
            _dsys.stderr.write(f"    Disc radius cap: 99pct={_r99:.1f} m, cap={_disc_r_cap:.3g} m, "
                               f"outliers excluded: {_n_disc_outliers}\n")
        valid_mask_d = valid_mask_d & (R_arr <= _disc_r_cap)
    valid_indices = np.where(valid_mask_d)[0]
    n_valid = len(valid_indices)
    _dsys.stderr.write(f"    Valid discs: {n_valid}/{total_events}\n")

    if n_valid == 0:
        _dsys.stderr.write("    No valid disc events — returning empty.\n")
        return []

    # Per-disc geometry constants
    n_seg      = n_radial_segments
    n_verts_d  = 1 + 2 * n_seg   # center + inner ring + outer ring
    n_faces_d  = 3 * n_seg        # center fan + inner→outer quads (×2 tri each)
    total_verts = n_valid * n_verts_d
    total_faces = n_valid * n_faces_d

    # Build template faces once (local vertex indices 0..n_verts_d-1)
    tmpl_faces = np.empty((n_faces_d, 3), dtype=np.int32)
    for _i in range(n_seg):
        _ni = (_i + 1) % n_seg
        tmpl_faces[_i] = [0, 1 + _i, 1 + _ni]
    for _i in range(n_seg):
        _ni = (_i + 1) % n_seg
        _p1, _p2 = 1 + _i, 1 + _ni
        _p3, _p4 = 1 + n_seg + _i, 1 + n_seg + _ni
        tmpl_faces[n_seg + 2*_i]     = [_p1, _p3, _p2]
        tmpl_faces[n_seg + 2*_i + 1] = [_p2, _p3, _p4]

    # Pre-allocate vertex and face buffers
    all_verts = np.empty((total_verts, 3), dtype=np.float64)
    all_faces = np.empty((total_faces, 3), dtype=np.int32)

    # Also pre-allocate per-point attribute arrays (event-level values repeated n_verts_d times)
    fault_id_arr = np.empty(total_verts, dtype=np.int32)
    radius_arr   = np.empty(total_verts, dtype=np.float64)
    area_arr     = np.empty(total_verts, dtype=np.float64)
    nx_arr_out   = np.empty(total_verts, dtype=np.float64)
    ny_arr_out   = np.empty(total_verts, dtype=np.float64)
    nz_arr_out   = np.empty(total_verts, dtype=np.float64)
    cx_arr_out   = np.empty(total_verts, dtype=np.float64)
    cy_arr_out   = np.empty(total_verts, dtype=np.float64)
    cz_arr_out   = np.empty(total_verts, dtype=np.float64)

    # --- Per-event geometry loop (vertex/face fill only — no pv.PolyData here) ---
    for _li, row_idx in enumerate(valid_indices):
        if _li % 5000 == 0 and _li > 0:
            _el = _dtime.time() - _t0
            _dsys.stderr.write(f"    Disc geometry: {_li}/{n_valid} ({_li/_el:.0f} ev/s)...\n")

        center = np.array([X_arr[row_idx], Y_arr[row_idx], Z_arr[row_idx]], dtype=np.float64)
        radius = float(R_arr[row_idx])
        normal = np.array([NX_arr[row_idx], NY_arr[row_idx], NZ_arr[row_idx]], dtype=np.float64)
        normal = _normalize_vector(normal)

        vertices, _ = _generate_disc_vertices_and_faces(center, normal, radius, n_seg)

        _vs = _li * n_verts_d
        _fs = _li * n_faces_d
        all_verts[_vs:_vs + n_verts_d] = vertices
        all_faces[_fs:_fs + n_faces_d] = tmpl_faces + _li * n_verts_d

        # Scalar values (same for all vertices of this disc)
        fault_id_arr[_vs:_vs + n_verts_d] = row_idx
        radius_arr  [_vs:_vs + n_verts_d] = radius
        area_arr    [_vs:_vs + n_verts_d] = np.pi * radius * radius
        nx_arr_out  [_vs:_vs + n_verts_d] = normal[0]
        ny_arr_out  [_vs:_vs + n_verts_d] = normal[1]
        nz_arr_out  [_vs:_vs + n_verts_d] = normal[2]
        cx_arr_out  [_vs:_vs + n_verts_d] = center[0]
        cy_arr_out  [_vs:_vs + n_verts_d] = center[1]
        cz_arr_out  [_vs:_vs + n_verts_d] = center[2]

    _dsys.stderr.write(f"    Geometry loop done ({_dtime.time()-_t0:.1f}s). Building PyVista mesh...\n")

    # Build ONE PolyData from pre-allocated arrays
    faces_pv = np.hstack([np.full((total_faces, 1), 3, dtype=np.int32), all_faces])
    combined = pv.PolyData(all_verts, faces_pv.flatten())

    combined['fault_id']     = fault_id_arr
    combined['radius']       = radius_arr
    combined['area_m2']      = area_arr
    combined['normal_x']     = nx_arr_out
    combined['normal_y']     = ny_arr_out
    combined['normal_z']     = nz_arr_out
    combined['X']            = cx_arr_out
    combined['Y']            = cy_arr_out
    combined['Z']            = cz_arr_out
    combined['hypocenter_x'] = cx_arr_out.copy()
    combined['hypocenter_y'] = cy_arr_out.copy()
    combined['hypocenter_z'] = cz_arr_out.copy()

    # Scalar attributes from DataFrame columns (event-level → repeat to point-level)
    rename_map = _get_vtp_column_name_mapping()
    attribute_columns = [
        'ID', 'LAT', 'LON', 'DEPTH', 'X', 'Y', 'Z',
        'sequence_label', 'segmentation_level', 'fault_id',
        'Mw', 'rupt_area', 'rupt_radius',
        'rupt_plane_azi', 'rupt_plane_dip', 'rake',
        'Sn_eff', 'Tau', 'instab', 'sliptend', 'dilatend'
    ]
    for col in attribute_columns:
        if col in df_subcluster.columns:
            try:
                vals = df_subcluster[col].values[valid_indices]
                combined[rename_map.get(col, col)] = np.repeat(vals, n_verts_d)
            except Exception:
                pass

    # Temporal attributes
    if 'Date' in df_subcluster.columns and df_subcluster['Date'].notna().any():
        try:
            all_dates = pd.to_datetime(df_subcluster['Date'])
            min_date  = all_dates.min()
            unix_ts   = all_dates.apply(lambda d: d.timestamp() if pd.notna(d) else np.nan).values.astype(np.float64)
            days_arr  = (all_dates - min_date).dt.days.values.astype(np.float32)
            dy_arr    = (all_dates.dt.year + all_dates.dt.dayofyear / 365.25).values.astype(np.float32)
            mo_arr    = all_dates.dt.month.values.astype(np.int32)

            combined['unix_timestamp']   = np.repeat(unix_ts [valid_indices], n_verts_d)
            combined['days_since_first'] = np.repeat(days_arr[valid_indices], n_verts_d)
            combined['decimal_year']     = np.repeat(dy_arr  [valid_indices], n_verts_d)
            combined['month']            = np.repeat(mo_arr  [valid_indices], n_verts_d)
            combined['Date']             = np.repeat(unix_ts [valid_indices], n_verts_d)
        except Exception:
            pass

    _dsys.stderr.write(f"  ✓ Disc mesh done: {total_verts:,} vertices, {total_faces:,} faces "
                       f"(took {_dtime.time()-_t0:.1f}s)\n")
    return [combined]


def _calculate_max_magnitude_from_area(area_m2, scaling_law='leonard2014'):
    """
    Calculate maximum magnitude from fault surface area using established scaling relationships.
    
    Parameters
    ----------
    area_m2 : float
        Fault surface area in square meters
    scaling_law : str
        Scaling relationship to use ('leonard2014', 'wells_coppersmith1994', 'thingbaijam2017')
        
    Returns
    -------
    float
        Maximum moment magnitude (Mw)
    """
    # Convert area from m² to km²
    area_km2 = area_m2 / 1e6
    
    if scaling_law == 'leonard2014':
        # Leonard (2014) - SCR SS earthquakes: Mw = 4.18 + 1.0 * log10(A)
        a, b = 4.18, 1.0
    elif scaling_law == 'wells_coppersmith1994':
        # Wells & Coppersmith (1994) - All fault types: Mw = 3.98 + 1.02 * log10(A)
        a, b = 3.98, 1.02
    elif scaling_law == 'thingbaijam2017':
        # Thingbaijam et al. (2017) - Strike-slip: Mw = 4.441 + 0.846 * log10(A)
        a, b = 4.441, 0.846
    else:
        raise ValueError(f"Unknown scaling law: {scaling_law}")
    
    # Calculate Mw from area: Mw = a + b * log10(A)
    if area_km2 <= 0:
        return np.nan
    
    max_magnitude = a + b * np.log10(area_km2)
    return max_magnitude


def _poisson_reconstruction(points, normals, depth=3, density_threshold=0.4, max_distance_factor=1.5, rupture_radii=None):
    """
    Perform Poisson surface reconstruction on a point cloud with normals.
    
    Parameters
    ----------
    points : numpy.ndarray
        Point coordinates (n_points, 3)
    normals : numpy.ndarray
        Normal vectors (n_points, 3)
    depth : int
        Reconstruction depth (higher = more detail)
    density_threshold : float
        Density quantile threshold for vertex removal (0.0-1.0)
        Lower values keep more vertices, higher values are more aggressive
    max_distance_factor : float
        Maximum allowed distance from input points as a factor of the maximum
        rupture radius in the dataset. Controls spatial extent of the reconstructed surface.
        E.g., 1.5 means mesh vertices can be up to 1.5x the largest rupture radius
        away from the nearest input point. Use values like:
        - 1.2: tight fit (close to rupture discs)
        - 1.5: moderate extension (default, good balance)
        - 2.0: more generous extension
    rupture_radii : numpy.ndarray, optional
        Array of rupture radii corresponding to each fault plane point.
        If provided, the maximum value is used to calculate clipping distance.
        This makes the parameter dataset-independent and based on actual fault sizes.
        If None, falls back to point spacing estimation.
        
    Returns
    -------
    pyvista.PolyData or None
        Reconstructed mesh as PyVista PolyData, or None if reconstruction failed
    """
    
    try:
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
        
        # Ensure normal consistency.
        # orient_normals_consistent_tangent_plane() is known to segfault for
        # nearly-coplanar point clouds (e.g. a single fault cluster where all
        # normals point in roughly the same direction).  Since HyFI normals
        # are already derived from rupture-plane fitting, they are already
        # well-oriented; we only need to fix sign flips.
        # Strategy: majority-vote — flip any normal whose dot-product with
        # the mean normal is negative.  This is safe and cannot segfault.
        n_arr = np.asarray(pcd.normals)  # (N, 3) copy
        if len(n_arr) > 0:
            mean_n = n_arr.mean(axis=0)
            mean_n_norm = np.linalg.norm(mean_n)
            # Only apply when the mean is well-defined (not zero)
            if mean_n_norm > 1e-9:
                mean_n /= mean_n_norm
                dots = n_arr @ mean_n           # (N,)
                flip_mask = dots < 0.0
                n_arr[flip_mask] *= -1.0
                pcd.normals = o3d.utility.Vector3dVector(n_arr)
        
        # Perform Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )
        
        # Remove low-density vertices
        densities = np.asarray(densities)
        if len(densities) > 0:
            vertices_to_remove = densities < np.quantile(densities, density_threshold)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            densities = densities[~vertices_to_remove]
        
        # Convert to PyVista mesh
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        if len(vertices) == 0 or len(faces) == 0:
            return None
        
        # Calculate distance-based clipping threshold using rupture radii
        from scipy.spatial import cKDTree
        
        if rupture_radii is not None and len(rupture_radii) > 0:
            # Use maximum rupture radius from the dataset - dataset-independent and consistent!
            # This allows the interpolated surface to extend far enough to connect all fault planes
            max_rupture_radius = np.max(rupture_radii)
            max_distance = max_rupture_radius * max_distance_factor
            
            # For each mesh vertex, find distance to nearest input point
            tree_input = cKDTree(points)
            distances_to_input, _ = tree_input.query(vertices)
            
            # Keep only vertices within the clipping distance
            vertices_to_keep = distances_to_input <= max_distance
            
        else:
            # Fallback: estimate from point spacing (original method)
            # This is less robust but works if rupture radii are not available
            points_extent = np.max(np.ptp(points, axis=0))
            k_neighbors = min(10, len(points))
            
            tree = cKDTree(points)
            if k_neighbors > 1:
                distances, _ = tree.query(points, k=k_neighbors)
                char_length = np.mean(distances[:, 1:])
            else:
                char_length = points_extent * 0.1
            
            max_distance = char_length * max_distance_factor
            
            tree_input = cKDTree(points)
            distances_to_input, _ = tree_input.query(vertices)
            vertices_to_keep = distances_to_input <= max_distance
        
        # Remove vertices that are too far
        if not np.all(vertices_to_keep):
            # Need to rebuild mesh with filtered vertices
            vertex_map = np.full(len(vertices), -1, dtype=int)
            vertex_map[vertices_to_keep] = np.arange(np.sum(vertices_to_keep))
            
            # Filter vertices
            vertices_filtered = vertices[vertices_to_keep]
            
            # Filter faces - keep only faces where all vertices are kept
            faces_mask = np.all(vertices_to_keep[faces], axis=1)
            faces_filtered = faces[faces_mask]
            
            # Remap face indices to new vertex array
            faces_filtered = vertex_map[faces_filtered]
            
            # Update vertices and faces
            vertices = vertices_filtered
            faces = faces_filtered
            
            # Update densities array to match filtered vertices
            if len(densities) > 0:
                densities = densities[vertices_to_keep]
        
        if len(vertices) == 0 or len(faces) == 0:
            return None
        
        # Format faces for PyVista (add face size)
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])
        pv_mesh = pv.PolyData(vertices, faces_pv.flatten())
        
        # Add density as scalar attribute
        if len(densities) == len(vertices):
            # Normalize densities
            densities_norm = (densities - np.min(densities)) / (np.max(densities) - np.min(densities))
            pv_mesh['densities'] = densities_norm
        
        # Mesh area will be calculated after reconstruction using pv_mesh.area
        # This ensures the area reflects the actual filtered surface
        
        return pv_mesh
    
    except Exception as e:
        print(f"Warning: Poisson reconstruction failed: {e}")
        return None


def create_interpolated_fault_planes(df_hyfi, interpolation_params, include_multiple_scaling_laws=False, starting_fault_counter=1):
    """
    Create interpolated fault plane surfaces using Poisson reconstruction with single dataframe.
    
    Parameters
    ----------
    df_hyfi : DataFrame
        Single dataframe containing all hypocenter and fault plane data
    interpolation_params : dict
        Parameters for interpolation including min_points, radius settings, etc.
    include_multiple_scaling_laws : bool, optional
        If True, calculate and include multiple magnitude scaling laws
    starting_fault_counter : int, optional
        Starting value for global fault counter (default: 1)
        Used to convert temporary cluster IDs to permanent FS IDs
        
    Returns
    -------
    combined_mesh : pyvista.PolyData
        Combined mesh of all fault planes
    individual_meshes : list
        List of individual mesh info dictionaries
    combined_pcd : pyvista.PolyData  
        Combined point cloud of fault plane points
    fault_disc_meshes : list
        List of circular disc meshes
    fault_metadata : list
        Metadata for each successfully interpolated fault
    next_fault_counter : int
        Next available counter value (for continuous numbering across sequences)
    """
    
    print('\n')
    print('='*50)
    print('FAULT PLANE INTERPOLATION')
    print('='*50)
    
    # Extract parameters
    depth = interpolation_params.get('poisson_depth', 3)
    density_threshold = interpolation_params.get('density_threshold', 0.4)
    max_distance_factor = interpolation_params.get('max_distance_factor', 1.5)
    radius_interval = interpolation_params.get('fault_plane_radius_interval_meters', 
                                               interpolation_params.get('radius_interval_meters', 10.0))
    point_density_meters = interpolation_params.get('fault_plane_point_density_meters',
                                                    interpolation_params.get('circle_point_density_meters', 10.0))
    min_points = interpolation_params.get('min_fault_planes_for_interpolation', 10)
    # Safety limit: maximum input points for Poisson reconstruction (prevents OOM on large clusters)
    max_poisson_input_points = interpolation_params.get('max_poisson_input_points', 100000)
    # Safety limit: maximum events for disc mesh generation (set to 0 to disable limiting)
    max_events_for_discs = interpolation_params.get('max_events_for_disc_meshes', 0)
    
    print(f"Interpolation parameters:")
    print(f"  Poisson depth: {depth}")
    print(f"  Density threshold: {density_threshold}")
    print(f"  Max distance factor: {max_distance_factor}")
    print(f"  Radius interval for circles: {radius_interval} m")
    print(f"  Point density on circles: {point_density_meters} m")
    print(f"  Minimum points per cluster: {min_points}")
    print(f"  Max Poisson input points: {max_poisson_input_points:,} (OOM guard)")
    
    df = df_hyfi
    
    # Filter out events with no valid fault plane data (NaN normal vectors)
    required_columns = ['nor_x_mean', 'nor_y_mean', 'nor_z_mean', 'rupt_radius']
    
    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns for interpolation: {missing_columns}")
        return None, [], None, [], [], starting_fault_counter
    
    # Count total events before filtering
    total_events = len(df)
    
    # Filter out events with NaN values in normal vectors or rupture radius
    valid_mask = (
        df['nor_x_mean'].notna() & 
        df['nor_y_mean'].notna() & 
        df['nor_z_mean'].notna() & 
        df['rupt_radius'].notna() & 
        (df['rupt_radius'] > 0)  # Also check for positive rupture radius
    )
    
    df = df[valid_mask].reset_index(drop=True)
    
    if len(df) == 0:
        print(f"No valid fault plane data available after filtering ({total_events} events had NaN or invalid fault plane parameters)")
        print("This typically occurs when no fault planes could be calculated during the fault plane determination step.")
        print("Skipping interpolation step.")
        return None, [], None, [], [], starting_fault_counter
    
    events_filtered = total_events - len(df)
    if events_filtered > 0:
        print(f"Filtered out {events_filtered} events with invalid fault plane data (NaN normal vectors or invalid rupture radius)")
    
    print(f"Using {len(df)} events with valid fault plane data (from {total_events} total events)")
    
    # Determine clustering approach based on available columns
    # For individual sequence outputs, use local IDs (1, 2, 3) instead of global FS IDs
    use_local_ids = 'final_cluster_id_local' in df.columns and df['final_cluster_id_local'].notna().any()
    
    if use_local_ids:
        cluster_column = 'final_cluster_id_local'
        print("Using local cluster IDs for individual sequence visualization")
    elif 'final_cluster_id' in df.columns and df['final_cluster_id'].notna().any():
        cluster_column = 'final_cluster_id'
        print("Using combined orientation+spatial clustering results")
    elif 'orient_cluster' in df.columns:
        cluster_column = 'orient_cluster'
        print("Using orientation-only clustering results (spatial clustering may be disabled)")
    else:
        print("Warning: No clustering results found - treating all data as single cluster")
        df['orient_cluster'] = 0
        cluster_column = 'orient_cluster'
    
    # Get unique final clusters
    unique_clusters = df[cluster_column].dropna().unique()
    print(f"Processing {len(unique_clusters)} fault clusters...", flush=True)
    
    # Initialize results containers
    combined_mesh = pv.PolyData()
    individual_meshes = []
    combined_pcd = pv.PolyData()
    fault_disc_meshes = []
    
    import sys as _sys_vis

    # Process each final cluster (already optimally clustered by auto_class)
    for i, cluster_id in enumerate(unique_clusters):
        df_cluster = df[df[cluster_column] == cluster_id].reset_index(drop=True)
        
        if len(df_cluster) < min_points:
            _sys_vis.stderr.write(f"  Skipping cluster {cluster_id}: too few points ({len(df_cluster)} < {min_points})\n")
            continue
        
        _sys_vis.stderr.write(f"  Processing cluster {cluster_id} ({len(df_cluster)} fault planes)...\n")
        
        # For disc mesh generation, use full cluster (or spatially subsampled if max set)
        _sys_vis.stderr.write(f"    [step 1/4] setting up disc cluster...\n")
        df_cluster_discs = df_cluster
        if max_events_for_discs > 0 and len(df_cluster) > max_events_for_discs:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(df_cluster), size=max_events_for_discs, replace=False)
            idx.sort()
            df_cluster_discs = df_cluster.iloc[idx].reset_index(drop=True)
            _sys_vis.stderr.write(f"    Disc meshes: spatially subsampled to {max_events_for_discs} events\n")
        
        # For Poisson reconstruction, subsample to keep point cloud manageable.
        _sys_vis.stderr.write(f"    [step 2/4] setting up Poisson cluster...\n")
        df_cluster_poisson = df_cluster
        poisson_event_limit = interpolation_params.get('max_events_for_poisson', 0)
        if poisson_event_limit > 0 and len(df_cluster) > poisson_event_limit:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(df_cluster), size=poisson_event_limit, replace=False)
            idx.sort()
            df_cluster_poisson = df_cluster.iloc[idx].reset_index(drop=True)
            _sys_vis.stderr.write(f"    Poisson input: subsampled to {poisson_event_limit} events (from {len(df_cluster)})\n")
        
        # Generate points from circular fault plane geometries
        _sys_vis.stderr.write(f"    [step 3/4] calling _generate_fault_plane_points ({len(df_cluster_poisson)} events)...\n")
        try:
            points, normals, rupture_radii, is_hypocenter, event_indices = _generate_fault_plane_points(
                df_cluster_poisson, radius_interval, point_density_meters,
                max_points=max_poisson_input_points
            )
            
            # Check if we actually got valid points
            if len(points) == 0:
                _sys_vis.stderr.write(f"    Warning: No valid points generated for cluster {cluster_id}\n")
                continue
                
            _sys_vis.stderr.write(f"    Generated {len(points)} points from {len(df_cluster_poisson)} fault planes\n")
            # Create point cloud for this cluster
            pcd = pv.PolyData(points)
            pcd['normals'] = normals
            pcd['is_hypocenter'] = is_hypocenter.astype(np.int32)
            
            # Propagate event attributes to all points
            # Create attribute arrays for all points based on their source event
            rename_map = _get_vtp_column_name_mapping()
            
            # Use same columns as rupture planes
            specified_columns = [
                'ID', 'LAT', 'LON', 'DEPTH', 'X', 'Y', 'Z', 'Date',
                'sequence_label', 'segmentation_level', 'fault_id',
                'Mw', 'rupt_area', 'rupt_radius',
                'rupt_plane_azi', 'rupt_plane_dip', 'rake',
                'Sn_eff', 'Tau', 'instab', 'sliptend', 'dilatend'
            ]
            temporal_columns = ['unix_timestamp', 'days_since_first', 'decimal_year']
            
            for col in specified_columns + temporal_columns:
                if col in df_cluster_poisson.columns:
                    try:
                        # Get values for each point's source event
                        values = df_cluster_poisson.iloc[event_indices][col].values
                        # Use renamed column name for VTP attribute
                        vtp_col_name = rename_map.get(col, col)
                        pcd[vtp_col_name] = values
                    except Exception as e:
                        pass  # Skip columns that can't be added
            
            pcd = pcd.clean()
            
        except Exception as e:
            print(f"    Warning: Failed to generate fault plane points: {e}")
            # Fallback to hypocenter locations only if we have valid normal vectors
            valid_rows = df_cluster.dropna(subset=['nor_x_mean', 'nor_y_mean', 'nor_z_mean', 'rupt_radius'])
            if len(valid_rows) == 0:
                print(f"    Skipping cluster {cluster_id}: no valid fault plane data available")
                continue
            points = valid_rows[['X', 'Y', 'Z']].values.astype(np.float64)
            # Use existing normal vectors for fallback
            normals = valid_rows[['nor_x_mean', 'nor_y_mean', 'nor_z_mean']].values.astype(np.float64)
            # Use rupture radii for fallback
            rupture_radii = valid_rows['rupt_radius'].values.astype(np.float64)
            
            # Create point cloud for this cluster
            pcd = pv.PolyData(points)
            pcd['normals'] = normals
            pcd['is_hypocenter'] = np.ones(pcd.n_points, dtype=np.int32)  # Fallback: all are hypocenters
            
            # Add attributes from valid_rows
            rename_map = _get_vtp_column_name_mapping()
            specified_columns = [
                'ID', 'LAT', 'LON', 'DEPTH', 'X', 'Y', 'Z', 'Date',
                'sequence_label', 'segmentation_level', 'fault_id',
                'Mw', 'rupt_area', 'rupt_radius',
                'rupt_plane_azi', 'rupt_plane_dip', 'rake',
                'Sn_eff', 'Tau', 'instab', 'sliptend', 'dilatend'
            ]
            temporal_columns = ['unix_timestamp', 'days_since_first', 'decimal_year']
            
            for col in specified_columns + temporal_columns:
                if col in valid_rows.columns:
                    try:
                        vtp_col_name = rename_map.get(col, col)
                        pcd[vtp_col_name] = valid_rows[col].values
                    except:
                        pass
            
            pcd = pcd.clean()
        
        # Check if we have enough points for Poisson reconstruction
        if pcd.n_points < 4:  # Minimum points needed for surface reconstruction
            print(f"    Warning: Cluster {cluster_id} has too few points ({pcd.n_points}) for surface reconstruction")
            continue
        
        # Add to combined point cloud
        if combined_pcd.n_points == 0:
            # First point cloud - initialize combined_pcd
            combined_pcd = pcd.copy()
        else:
            # Merge with existing point cloud
            combined_pcd = combined_pcd.merge(pcd)
        
        # Create circular disc meshes for visualization
        disc_meshes = _create_circular_fault_disc_meshes(df_cluster_discs)
        if len(disc_meshes) > 0:
            # Batch combine disc meshes more efficiently - combine every N meshes to avoid memory spike
            if len(fault_disc_meshes) > 0:
                # Combine new meshes with existing ones in batches
                temp_combined = fault_disc_meshes[-1] if len(fault_disc_meshes) == 1 else pv.merge([fault_disc_meshes[-1]] + disc_meshes)
                if len(fault_disc_meshes) > 1:
                    temp_combined = pv.merge(fault_disc_meshes[:-1] + [temp_combined])
                fault_disc_meshes = [temp_combined]
            else:
                # First batch
                if len(disc_meshes) > 1:
                    fault_disc_meshes = [pv.merge(disc_meshes)]
                else:
                    fault_disc_meshes = disc_meshes
        
        # Attempt Poisson surface reconstruction
        try:
            mesh = _poisson_reconstruction(points, normals, depth, density_threshold, max_distance_factor, rupture_radii)
        except Exception as e:
            print(f"    Warning: Poisson reconstruction failed: {e}")
            mesh = None
        
        if mesh is not None:
            # Calculate mesh area AFTER filtering (area is already computed in _poisson_reconstruction)
            # The mesh.area property gives us the final filtered surface area
            mesh_area = mesh.area
            
            # Calculate maximum magnitude from FILTERED area using Leonard (2014) scaling law as default
            max_Mw = _calculate_max_magnitude_from_area(mesh_area, 'leonard2014')
            
            # Ensure mesh is fully triangulated immediately after creation (before any other operations)
            if not mesh.is_all_triangles:
                print(f"    Note: Mesh for cluster {cluster_id} contains non-triangular faces, triangulating...")
                mesh = mesh.triangulate()
            
            # Add cluster metadata attributes to mesh
            mesh['fault_idx'] = np.full(mesh.n_points, i)
            mesh['cluster_id'] = np.full(mesh.n_points, str(cluster_id))
            mesh['area_m2'] = np.full(mesh.n_points, mesh_area)  # Add filtered area as point data
            mesh['max_Mw'] = np.full(mesh.n_points, max_Mw)  # Add maximum magnitude based on filtered area

            # Make fault_idx default
            mesh.active_scalars_name = 'fault_idx'
            
            # Optionally add other scaling laws (not shown in DAG but available in VTP)
            if include_multiple_scaling_laws:
                max_Mw_WC94 = _calculate_max_magnitude_from_area(mesh_area, 'wells_coppersmith1994')
                max_Mw_T17 = _calculate_max_magnitude_from_area(mesh_area, 'thingbaijam2017')
                mesh['max_Mw_wells_coppersmith1994'] = np.full(mesh.n_points, max_Mw_WC94)
                mesh['max_Mw_thingbaijam2017'] = np.full(mesh.n_points, max_Mw_T17)
            
            # Determine orientation and spatial cluster info for export naming
            if cluster_column == 'final_cluster_id_local':
                # Local numeric format (1, 2, 3) for individual sequences
                ori_cluster = str(cluster_id)
                spatial_cluster = '0'
            elif cluster_column == 'final_cluster_id':
                # Parse cluster format: FS0001, FS0002 (global) or legacy formats
                cluster_str = str(cluster_id)
                if pd.isna(cluster_id) or cluster_str == 'nan':
                    # Handle NaN noise points (should not reach here due to dropna() but safety check)
                    ori_cluster = 'noise'
                    spatial_cluster = '0'
                elif cluster_str.startswith('FS'):
                    # Global FS format - use full ID
                    ori_cluster = cluster_str  # FS0001, FS0002, etc.
                    spatial_cluster = '0'
                elif cluster_str.isdigit():
                    # Local numeric format (for individual sequences) - use directly
                    ori_cluster = cluster_str
                    spatial_cluster = '0'
                elif cluster_str.startswith('F'):
                    # Legacy F1, F2, F3 format - use directly
                    ori_cluster = cluster_str
                    spatial_cluster = '0'
                elif '_spatial_' in cluster_str:
                    # Legacy format: "ori_X_spatial_Y" 
                    parts = cluster_str.split('_spatial_')
                    ori_cluster = parts[0].replace('ori_', '')
                    spatial_cluster = parts[1]
                else:
                    # Handle cases where cluster_id might just be the orientation cluster
                    ori_cluster = cluster_str
                    spatial_cluster = '0'
            elif cluster_column == 'orient_cluster':
                # Only orientation clustering
                ori_cluster = str(cluster_id)
                spatial_cluster = '0'
            else:
                # Fallback case
                ori_cluster = str(cluster_id)
                spatial_cluster = '0'
            
            # Add to combined mesh and individual list
            if combined_mesh.n_points == 0:
                # First mesh - just assign it
                combined_mesh = mesh.copy()
            else:
                # Add to existing combined mesh
                combined_mesh = combined_mesh + mesh
            
            individual_meshes.append({
                'mesh': mesh,
                'cluster_id': str(cluster_id),
                'original_cluster_id': str(cluster_id),  # Store original ID for dataframe lookup
                'orientation_cluster': ori_cluster,
                'spatial_cluster': spatial_cluster,
                'fault_idx': i,
                'n_fault_planes': len(df_cluster),
                'n_input_points': len(points),
                'area_m2': mesh_area,  # Store mesh area
                'max_Mw': max_Mw,  # Store maximum magnitude (Leonard 2014)
                # Store cluster statistics for metadata (computed from df_cluster for this specific cluster)
                'cluster_centroid_x': float(df_cluster['X'].mean()) if 'X' in df_cluster.columns else None,
                'cluster_centroid_y': float(df_cluster['Y'].mean()) if 'Y' in df_cluster.columns else None,
                'cluster_centroid_z': float(df_cluster['Z'].mean()) if 'Z' in df_cluster.columns else None,
                'cluster_data': df_cluster.copy()  # Store actual cluster data for metadata generation
            })
            
            print(f"    ✓ Successfully created mesh with {mesh.n_points} vertices, {mesh.n_cells} faces, area {mesh_area:.1f} m², and max Mw {max_Mw:.2f}")
        else:
            print(f"    × Failed to create mesh for cluster {cluster_id}")
    
    # Apply mesh subdivision if requested (for higher density while maintaining smoothness)
    n_subdivisions = interpolation_params.get('mesh_subdivisions', 0)
    
    if n_subdivisions > 0 and len(individual_meshes) > 0:
        print(f"\nApplying mesh subdivision ({n_subdivisions} iterations)...")
        combined_mesh_original_faces = combined_mesh.n_cells
        
        # Check if combined mesh is valid before subdivision
        if combined_mesh.n_cells == 0:
            print(f"  Warning: Combined mesh is empty ({combined_mesh.n_cells} faces), skipping subdivision")
        else:
            # Clean and prepare combined mesh for subdivision
            print(f"  Preparing combined mesh for subdivision...")
            
            # 1. Clean the mesh (remove duplicate points, degenerate cells)
            combined_mesh = combined_mesh.clean(tolerance=1e-6)
            
            # 2. Triangulate to ensure all faces are triangles
            combined_mesh = combined_mesh.triangulate()
            
            # 3. Extract the surface to ensure manifold topology
            combined_mesh = combined_mesh.extract_surface()
            
            # 4. Final cleaning after surface extraction
            combined_mesh = combined_mesh.clean(tolerance=1e-6)
            
            print(f"  Cleaned mesh: {combined_mesh.n_cells} faces, {combined_mesh.n_points} vertices, all_triangles={combined_mesh.is_all_triangles}")
            
            # Subdivide combined mesh using Loop subdivision (maintains smoothness)
            try:
                for i in range(n_subdivisions):
                    combined_mesh = combined_mesh.subdivide(nsub=1, subfilter='loop')
                print(f"  ✓ Combined mesh subdivision: {combined_mesh_original_faces} → {combined_mesh.n_cells} faces ({combined_mesh.n_cells/combined_mesh_original_faces:.1f}x)")
            except Exception as e:
                print(f"  Warning: Combined mesh subdivision failed: {e}")
                print(f"  Continuing with original combined mesh ({combined_mesh_original_faces} faces)")
        
        # Subdivide individual meshes
        for mesh_info in individual_meshes:
            original_faces = mesh_info['mesh'].n_cells
            if mesh_info['mesh'].n_cells == 0:
                print(f"  Warning: Cluster {mesh_info['cluster_id']} mesh is empty, skipping subdivision")
                continue
            
            # Clean and prepare mesh for subdivision
            mesh = mesh_info['mesh']
            
            # 1. Clean the mesh
            mesh = mesh.clean(tolerance=1e-6)
            
            # 2. Triangulate
            mesh = mesh.triangulate()
            
            # 3. Extract surface for manifold topology
            mesh = mesh.extract_surface()
            
            # 4. Final cleaning
            mesh = mesh.clean(tolerance=1e-6)
            
            mesh_info['mesh'] = mesh
            
            try:
                for i in range(n_subdivisions):
                    mesh_info['mesh'] = mesh_info['mesh'].subdivide(nsub=1, subfilter='loop')
                print(f"  ✓ Cluster {mesh_info['cluster_id']}: {original_faces} → {mesh_info['mesh'].n_cells} faces")
            except Exception as e:
                print(f"  Warning: Cluster {mesh_info['cluster_id']} subdivision failed: {e}")
                print(f"  Continuing with original mesh ({original_faces} faces)")
    
    print(f"✓ Interpolation complete. Created {len(individual_meshes)} fault plane meshes and {len(fault_disc_meshes)} circular disc meshes.")
    
    # ============================================================================
    # ASSIGN PERMANENT FS IDs TO SUCCESSFULLY INTERPOLATED MESHES
    # ============================================================================
    # Only successfully interpolated clusters get permanent FS IDs with global counter
    # This ensures continuous numbering across all sequences
    
    print(f"\nAssigning permanent fault IDs (starting from F{starting_fault_counter:04d})...")
    print(f"  Number of successfully interpolated meshes: {len(individual_meshes)}")
    
    # Create mapping from temporary cluster IDs to permanent fault IDs
    temp_to_fault_mapping = {}
    fault_counter = starting_fault_counter
    
    for i, mesh_info in enumerate(individual_meshes):
        # Get the temporary cluster ID from clustering
        temp_cluster_id = mesh_info.get('cluster_id', None)
        
        if temp_cluster_id is None:
            print(f"  Warning: Mesh {i} has no cluster_id, skipping")
            continue
        
        # Assign permanent fault ID using global counter
        permanent_fault_id = f"F{fault_counter:04d}"
        
        # Store the mapping
        temp_to_fault_mapping[temp_cluster_id] = permanent_fault_id
        
        # Update mesh_info with permanent ID
        mesh_info['cluster_id'] = permanent_fault_id
        mesh_info['permanent_fault_id'] = permanent_fault_id
        mesh_info['original_temp_id'] = temp_cluster_id
        
        # Update the mesh point data with permanent ID
        if 'mesh' in mesh_info and mesh_info['mesh'] is not None:
            mesh_info['mesh']['cluster_id'] = np.full(mesh_info['mesh'].n_points, permanent_fault_id)
        
        # Increment the global counter
        fault_counter += 1
    
    next_fault_counter = fault_counter
    
    print(f"✓ Assigned {len(temp_to_fault_mapping)} permanent fault IDs (F{starting_fault_counter:04d} to F{next_fault_counter-1:04d})")
    print(f"  Mapping created: {dict(list(temp_to_fault_mapping.items())[:3])}{'...' if len(temp_to_fault_mapping) > 3 else ''}")
    
    # ============================================================================
    # GENERATE METADATA FOR SUCCESSFULLY INTERPOLATED FAULT SYSTEMS
    # ============================================================================
    
    # Generate fault metadata for successfully interpolated meshes
    fault_metadata = []
    for mesh_info in individual_meshes:
        # Use the PERMANENT FS ID that was just assigned
        fs_id = mesh_info['cluster_id']
        
        # Use the cluster data that was stored during mesh creation
        # This ensures we have the correct data for THIS specific cluster
        cluster_data = mesh_info['cluster_data']
        
        # Get sequence info from cluster data
        if len(cluster_data) > 0:
            sequence_label = cluster_data.get('sequence_label', pd.Series([None])).iloc[0]
            segmentation_level = cluster_data.get('segmentation_level', pd.Series([None])).iloc[0]
        else:
            sequence_label = None
            segmentation_level = None
        
        if len(cluster_data) > 0:
            # Only create metadata for successfully interpolated faults
            
            # Extract stress analysis results from original rupture planes (mean of rupture plane values)
            rupture_mean_instability = None
            rupture_mean_sliptend = None
            rupture_mean_dilatend = None
            if 'instab' in cluster_data.columns:
                rupture_mean_instability = float(cluster_data['instab'].mean()) if not cluster_data['instab'].isna().all() else None
            if 'sliptend' in cluster_data.columns:
                rupture_mean_sliptend = float(cluster_data['sliptend'].mean()) if not cluster_data['sliptend'].isna().all() else None
            if 'dilatend' in cluster_data.columns:
                rupture_mean_dilatend = float(cluster_data['dilatend'].mean()) if not cluster_data['dilatend'].isna().all() else None
            
            # Extract stress analysis results from interpolated mesh faces (mean of cell data)
            mesh_mean_instability = None
            mesh_mean_sliptend = None
            mesh_mean_dilatend = None
            if 'instab' in mesh_info['mesh'].cell_data:
                mesh_mean_instability = float(np.nanmean(mesh_info['mesh'].cell_data['instab']))
            if 'sliptend' in mesh_info['mesh'].cell_data:
                mesh_mean_sliptend = float(np.nanmean(mesh_info['mesh'].cell_data['sliptend']))
            if 'dilatend' in mesh_info['mesh'].cell_data:
                mesh_mean_dilatend = float(np.nanmean(mesh_info['mesh'].cell_data['dilatend']))
            
            # Extract orientation from interpolated mesh faces (mean of normals converted to dip/azimuth)
            mesh_mean_dip = None
            mesh_mean_azimuth = None
            if mesh_info['mesh'].n_cells > 0:
                # Compute normals if not present
                temp_mesh = mesh_info['mesh'].copy()
                if 'Normals' not in temp_mesh.cell_data:
                    temp_mesh = temp_mesh.compute_normals(cell_normals=True, point_normals=False)
                
                face_normals = temp_mesh.cell_data['Normals']
                dips = []
                azimuths = []
                
                for normal in face_normals:
                    nx, ny, nz = normal[0], normal[1], normal[2]
                    
                    # Ensure normal points upward
                    if nz < 0:
                        nx, ny, nz = -nx, -ny, -nz
                    
                    # Calculate dip (angle from horizontal)
                    dip = np.degrees(np.arccos(np.clip(nz, -1, 1)))
                    dips.append(dip)
                    
                    # Calculate dip direction (azimuth)
                    azimuth = np.degrees(np.arctan2(nx, ny)) % 360
                    azimuths.append(azimuth)
                
                # Use proper circular statistics for orientation
                from hyfi.utils.utilities import circular_mean_orientation_from_azimuth_dip
                mesh_mean_azimuth, mesh_mean_dip = circular_mean_orientation_from_azimuth_dip(azimuths, dips)
            
            # Determine VTP filename using permanent FS ID
            vtp_filename = f"fault_{fs_id}.vtp"
            
            # Calculate rupture plane mean orientation using circular statistics
            from hyfi.utils.utilities import circular_mean_orientation_from_azimuth_dip
            rupture_mean_azimuth, rupture_mean_dip = None, None
            if 'rupt_plane_azi' in cluster_data.columns and 'rupt_plane_dip' in cluster_data.columns:
                rupture_mean_azimuth, rupture_mean_dip = circular_mean_orientation_from_azimuth_dip(
                    cluster_data['rupt_plane_azi'].values,
                    cluster_data['rupt_plane_dip'].values
                )
            
            metadata = {
                'fault_id': str(fs_id) if not pd.isna(fs_id) else None,
                'segmentation_level': segmentation_level,
                'sequence_label': sequence_label,
                'vtp_file': f"{sequence_label}/vtp_export/{vtp_filename}" if sequence_label else f"vtp_export/{vtp_filename}",  # Relative path from output directory
                'orientation_cluster': int(cluster_data['orient_cluster'].iloc[0]) if 'orient_cluster' in cluster_data.columns and not pd.isna(cluster_data['orient_cluster'].iloc[0]) else None,
                'spatial_cluster': int(cluster_data['spatial_cluster'].iloc[0]) if 'spatial_cluster' in cluster_data.columns and not pd.isna(cluster_data['spatial_cluster'].iloc[0]) else None,
                'n_rupture_planes': len(cluster_data),
                'n_events': int(cluster_data['N'].sum()) if 'N' in cluster_data.columns else len(cluster_data),
                # Geometric properties from original rupture planes: mean values
                'centroid_x': float(cluster_data['X'].mean()) if 'X' in cluster_data.columns else None,
                'centroid_y': float(cluster_data['Y'].mean()) if 'Y' in cluster_data.columns else None,
                'centroid_z': float(cluster_data['Z'].mean()) if 'Z' in cluster_data.columns else None,
                # Geometric properties from rupture planes: mean orientation using circular statistics
                'rupture_mean_azimuth': rupture_mean_azimuth,
                'rupture_mean_dip': rupture_mean_dip,
                # Geometric properties from interpolated mesh: mean values from face normals
                'mesh_mean_dip': mesh_mean_dip,
                'mesh_mean_azimuth': mesh_mean_azimuth,
                # Mesh properties: from interpolated surface
                'interpolated_mesh_area_m2': mesh_info.get('area_m2'),
                'max_magnitude_leonard2014': mesh_info.get('max_Mw'),
                'mesh_vertices': mesh_info['mesh'].n_points,
                'mesh_faces': mesh_info['mesh'].n_cells,
                # Stress properties from rupture planes: mean values from original rupture planes
                'rupture_mean_instability': rupture_mean_instability,
                'rupture_mean_sliptend': rupture_mean_sliptend,
                'rupture_mean_dilatend': rupture_mean_dilatend,
                # Stress properties from mesh: mean values from interpolated mesh faces
                'mesh_mean_instability': mesh_mean_instability,
                'mesh_mean_sliptend': mesh_mean_sliptend,
                'mesh_mean_dilatend': mesh_mean_dilatend,
            }
            fault_metadata.append(metadata)
            
            # Also update mesh_info with all metadata for VTP export
            mesh_info.update({
                'segmentation_level': segmentation_level,
                'sequence_label': sequence_label,
                'n_events': int(cluster_data['N'].sum()) if 'N' in cluster_data.columns else len(cluster_data),
                'centroid_x': float(cluster_data['X'].mean()) if 'X' in cluster_data.columns else None,
                'centroid_y': float(cluster_data['Y'].mean()) if 'Y' in cluster_data.columns else None,
                'centroid_z': float(cluster_data['Z'].mean()) if 'Z' in cluster_data.columns else None,
                'rupture_mean_azimuth': rupture_mean_azimuth,
                'rupture_mean_dip': rupture_mean_dip,
                'mesh_mean_azimuth': mesh_mean_azimuth,
                'mesh_mean_dip': mesh_mean_dip,
                'rupture_mean_instability': rupture_mean_instability,
                'rupture_mean_sliptend': rupture_mean_sliptend,
                'rupture_mean_dilatend': rupture_mean_dilatend,
                'mesh_mean_instability': mesh_mean_instability,
                'mesh_mean_sliptend': mesh_mean_sliptend,
                'mesh_mean_dilatend': mesh_mean_dilatend,
            })
    
    print(f"  Generated metadata for {len(fault_metadata)} interpolated faults")
    
    # Print mapping summary
    if temp_to_fault_mapping:
        print(f"  Created temp→fault ID mapping for {len(temp_to_fault_mapping)} faults")
    
    # Print total area and magnitude range if meshes were created
    if len(individual_meshes) > 0:
        total_area = sum(mesh_info.get('area_m2', 0) for mesh_info in individual_meshes if isinstance(mesh_info.get('area_m2'), (int, float)))
        magnitudes = [mesh_info.get('max_Mw', 0) for mesh_info in individual_meshes if isinstance(mesh_info.get('max_Mw'), (int, float)) and not np.isnan(mesh_info.get('max_Mw', np.nan))]
        
        if magnitudes:
            min_mag = min(magnitudes)
            max_mag = max(magnitudes)
        
        # Print combined mesh info
        print(f"  Combined mesh: {combined_mesh.n_points} vertices, {combined_mesh.n_cells} faces")
    
    # Check if we have any valid results
    if len(individual_meshes) == 0:
        print("Warning: No fault plane meshes were successfully created.")
        print("This may be due to insufficient valid fault plane data or reconstruction failures.")
        # Still return the point cloud and disc meshes if they exist
        # Return starting counter since no permanent IDs were assigned
        return None, [], combined_pcd if combined_pcd.n_points > 0 else None, fault_disc_meshes, [], starting_fault_counter, {}
    
    return combined_mesh, individual_meshes, combined_pcd, fault_disc_meshes, fault_metadata, next_fault_counter, temp_to_fault_mapping


def export_meshes_as_obj(combined_mesh, individual_meshes, fault_disc_meshes, output_dir, df_hyfi=None):
    """
    Export mesh data (faults, focals, rupture planes, slip vectors) as .obj files.
    
    Exports all mesh geometries in Wavefront OBJ format for use in external 3D software
    like Blender, MeshLab, CloudCompare, etc.
    
    Parameters
    ----------
    combined_mesh : pyvista.PolyData
        Combined mesh of all fault planes
    individual_meshes : list
        List of individual mesh dictionaries
    fault_disc_meshes : list
        List of circular disc mesh dictionaries
    output_dir : str
        Output directory path
    df_hyfi : DataFrame, optional
        Hypocenter dataframe for exporting slip vectors and focal planes
    """
    print("Exporting mesh data as OBJ files...")
    
    # Create OBJ export directory
    obj_dir = os.path.join(output_dir, 'obj_export')
    os.makedirs(obj_dir, exist_ok=True)
    
    # Clean up old OBJ files to avoid confusion from previous runs
    import glob
    old_obj_files = glob.glob(os.path.join(obj_dir, '*.obj'))
    if old_obj_files:
        print(f"  Cleaning up {len(old_obj_files)} old OBJ files...")
        for old_file in old_obj_files:
            try:
                os.remove(old_file)
            except Exception as e:
                print(f"  Warning: Could not remove {old_file}: {e}")
    
    # Export combined fault mesh only (not individual faults)
    if combined_mesh is not None and combined_mesh.n_points > 0:
        try:
            # PyVista can save directly as .obj using the file extension
            combined_file = os.path.join(obj_dir, 'faults_compiled.obj')
            combined_mesh.save(combined_file)
            print(f"  ✓ Saved combined fault mesh: faults_compiled.obj ({combined_mesh.n_points} vertices, {combined_mesh.n_cells} faces)")
        except Exception as e:
            print(f"  Warning: Could not export combined fault mesh as OBJ: {e}")
        
    # Export rupture plane meshes (circular discs)
    if fault_disc_meshes is not None and len(fault_disc_meshes) > 0:
        print(f"  Exporting rupture plane meshes...")
        try:
            # Create combined rupture plane mesh
            combined_disc_mesh = None
            
            for i, disc_info in enumerate(fault_disc_meshes):
                try:
                    # Handle both dictionary format and direct mesh format
                    if hasattr(disc_info, 'get'):
                        disc_mesh = disc_info.get('mesh')
                    else:
                        disc_mesh = disc_info
                    
                    if disc_mesh is not None and hasattr(disc_mesh, 'n_points') and disc_mesh.n_points > 0:
                        if combined_disc_mesh is None:
                            combined_disc_mesh = disc_mesh.copy()
                        else:
                            combined_disc_mesh = combined_disc_mesh.merge(disc_mesh)
                
                except Exception as e:
                    print(f"    Warning: Could not add rupture plane {i} to combined mesh: {e}")
                    continue
            
            if combined_disc_mesh is not None and combined_disc_mesh.n_points > 0:
                combined_disc_file = os.path.join(obj_dir, 'rupture_planes.obj')
                combined_disc_mesh.save(combined_disc_file)
                print(f"  ✓ Saved rupture planes: rupture_planes.obj ({combined_disc_mesh.n_points} vertices, {combined_disc_mesh.n_cells} faces)")
            
        except Exception as e:
            print(f"  Warning: Could not export rupture planes as OBJ: {e}")
    
    # Export focal mechanism meshes if available
    if df_hyfi is not None:
        try:
            # Check if focal mechanism data exists
            focal_cols = ['Strike1', 'Dip1', 'Strike2', 'Dip2', 'pref_foc']
            if all(col in df_hyfi.columns for col in focal_cols):
                valid_focal_mask = (
                    df_hyfi['pref_foc'].notna() &
                    df_hyfi['pref_foc'].isin([1, 2]) &
                    (((df_hyfi['pref_foc'] == 1) & df_hyfi['Strike1'].notna() & df_hyfi['Dip1'].notna()) |
                     ((df_hyfi['pref_foc'] == 2) & df_hyfi['Strike2'].notna() & df_hyfi['Dip2'].notna()))
                )
                
                focal_events = df_hyfi[valid_focal_mask].copy()
                
                if len(focal_events) > 0:
                    print(f"  Exporting focal mechanism meshes...")
                    # Create focal mechanism disc meshes
                    focal_meshes = _create_focal_mechanism_disc_meshes(focal_events)
                    
                    if focal_meshes:
                        # Create combined focal mesh
                        combined_focal_mesh = None
                        
                        for mesh_info in focal_meshes:
                            try:
                                mesh = mesh_info['mesh']
                                
                                if mesh is not None and mesh.n_points > 0:
                                    if combined_focal_mesh is None:
                                        combined_focal_mesh = mesh.copy()
                                    else:
                                        combined_focal_mesh = combined_focal_mesh.merge(mesh)
                            except Exception as e:
                                continue
                        
                        if combined_focal_mesh is not None and combined_focal_mesh.n_points > 0:
                            combined_focal_file = os.path.join(obj_dir, 'focals_compiled.obj')
                            combined_focal_mesh.save(combined_focal_file)
                            print(f"  ✓ Saved focal mechanisms: focals_compiled.obj ({combined_focal_mesh.n_points} vertices, {combined_focal_mesh.n_cells} faces)")
        
        except Exception as e:
            print(f"  Warning: Could not export focal mechanisms as OBJ: {e}")
    
    # Export slip vectors if available
    if df_hyfi is not None:
        try:
            required_cols = ['X', 'Y', 'Z', 'nor_x_mean', 'nor_y_mean', 'nor_z_mean', 'rupt_radius', 'rake']
            df_valid = df_hyfi.dropna(subset=required_cols).copy()
            
            if len(df_valid) > 0:
                print(f"  Exporting slip vectors...")
                
                # Create slip vector line segments
                line_points = []
                line_connections = []
                
                for i, (pt, row) in enumerate(zip(df_valid[['X', 'Y', 'Z']].values, df_valid.iterrows())):
                    _, row_data = row
                    x, y, z = pt
                    p = [x, y, z]
                    r = row_data['rupt_radius']
                    nor = np.array([row_data['nor_x_mean'], row_data['nor_y_mean'], row_data['nor_z_mean']])
                    rake = row_data['rake']
                    
                    # Calculate slip vector endpoint
                    u, v, w = utilities_plot.slipvector_3D(p, r, nor, rake)
                    
                    # Bidirectional vector
                    slip_vec = np.array([2 * (u - x), 2 * (v - y), 2 * (w - z)])
                    
                    start = pt - slip_vec / 2
                    end = pt + slip_vec / 2
                    
                    line_points.append(start)
                    line_points.append(end)
                    line_connections.append([2, len(line_points)-2, len(line_points)-1])
                
                line_points = np.array(line_points, dtype=np.float64)
                line_connections = np.hstack(line_connections)
                
                # Create line mesh
                line_mesh = pv.PolyData(line_points, lines=line_connections)
                
                # Apply tube filter
                tube_radius = 2.0
                tubes = line_mesh.tube(radius=tube_radius, n_sides=8)
                
                # Save as OBJ
                slip_vectors_file = os.path.join(obj_dir, 'slip_vectors.obj')
                tubes.save(slip_vectors_file)
                print(f"  ✓ Saved slip vectors: slip_vectors.obj ({tubes.n_points} vertices, {tubes.n_cells} faces)")
        
        except Exception as e:
            print(f"  Warning: Could not export slip vectors as OBJ: {e}")
    
    print(f"OBJ export complete. Files saved to: {obj_dir}")


def export_meshes_as_ply(combined_mesh, individual_meshes, fault_disc_meshes, point_cloud, output_dir, df_hyfi=None):
    """
    Export mesh and point cloud data as .ply files.
    
    Exports all mesh geometries and point clouds in PLY format, which preserves vertex attributes,
    colors, and is widely supported by point cloud processing software.
    
    Parameters
    ----------
    combined_mesh : pyvista.PolyData
        Combined mesh of all fault planes
    individual_meshes : list
        List of individual mesh dictionaries
    fault_disc_meshes : list
        List of circular disc mesh dictionaries
    point_cloud : pyvista.PolyData
        Combined point cloud of fault plane points
    output_dir : str
        Output directory path
    df_hyfi : DataFrame, optional
        Hypocenter dataframe for exporting additional point clouds
    """
    print("Exporting mesh and point cloud data as PLY files...")
    
    # Create PLY export directory
    ply_dir = os.path.join(output_dir, 'ply_export')
    os.makedirs(ply_dir, exist_ok=True)
    
    # Clean up old PLY files to avoid confusion from previous runs
    import glob
    old_ply_files = glob.glob(os.path.join(ply_dir, '*.ply'))
    if old_ply_files:
        print(f"  Cleaning up {len(old_ply_files)} old PLY files...")
        for old_file in old_ply_files:
            try:
                os.remove(old_file)
            except Exception as e:
                print(f"  Warning: Could not remove {old_file}: {e}")
    
    # Export combined fault mesh
    if combined_mesh is not None and combined_mesh.n_points > 0:
        try:
            combined_file = os.path.join(ply_dir, 'faults_compiled.ply')
            combined_mesh.save(combined_file)
            print(f"  ✓ Saved combined fault mesh: faults_compiled.ply ({combined_mesh.n_points} vertices)")
        except Exception as e:
            print(f"  Warning: Could not export combined fault mesh as PLY: {e}")
    
    # Export individual fault meshes
    if individual_meshes:
        print(f"  Exporting {len(individual_meshes)} individual fault meshes...")
        for mesh_info in individual_meshes:
            try:
                mesh = mesh_info['mesh']
                cluster_id = mesh_info['cluster_id']
                
                # Use F-based cluster naming for consistency
                if cluster_id.startswith('F') and cluster_id != 'F_noise':
                    filename = f"fault_{cluster_id}.ply"
                else:
                    fault_idx = mesh_info['fault_idx']
                    filename = f"fault_{fault_idx}.ply"
                
                filepath = os.path.join(ply_dir, filename)
                mesh.save(filepath)
                
            except Exception as e:
                print(f"    Warning: Could not export individual fault mesh {cluster_id}: {e}")
    
    # Export rupture plane meshes
    if fault_disc_meshes is not None and len(fault_disc_meshes) > 0:
        print(f"  Exporting rupture plane meshes...")
        try:
            # Create combined rupture plane mesh
            combined_disc_mesh = None
            
            for i, disc_info in enumerate(fault_disc_meshes):
                try:
                    if hasattr(disc_info, 'get'):
                        disc_mesh = disc_info.get('mesh')
                    else:
                        disc_mesh = disc_info
                    
                    if disc_mesh is not None and hasattr(disc_mesh, 'n_points') and disc_mesh.n_points > 0:
                        if combined_disc_mesh is None:
                            combined_disc_mesh = disc_mesh.copy()
                        else:
                            combined_disc_mesh = combined_disc_mesh.merge(disc_mesh)
                
                except Exception as e:
                    print(f"    Warning: Could not add rupture plane {i}: {e}")
                    continue
            
            if combined_disc_mesh is not None and combined_disc_mesh.n_points > 0:
                combined_disc_file = os.path.join(ply_dir, 'rupture_planes.ply')
                combined_disc_mesh.save(combined_disc_file)
                print(f"  ✓ Saved rupture planes: rupture_planes.ply ({combined_disc_mesh.n_points} vertices)")
            
        except Exception as e:
            print(f"  Warning: Could not export rupture planes as PLY: {e}")
    
    # Export fault plane point cloud
    if point_cloud is not None and point_cloud.n_points > 0:
        try:
            pcd_file = os.path.join(ply_dir, 'fault_plane_pointcloud.ply')
            point_cloud.save(pcd_file)
            print(f"  ✓ Saved fault plane point cloud: fault_plane_pointcloud.ply ({point_cloud.n_points} points)")
        except Exception as e:
            print(f"  Warning: Could not export fault plane point cloud as PLY: {e}")
    
    # Export hypocenter point cloud
    if df_hyfi is not None and len(df_hyfi) > 0:
        try:
            hypocenter_points = df_hyfi[['X', 'Y', 'Z']].values
            hypocenter_pcd = pv.PolyData(hypocenter_points)
            
            # Add attributes
            for col in ['ID', 'MAG', 'EX', 'EY', 'EZ']:
                if col in df_hyfi.columns:
                    hypocenter_pcd[col] = df_hyfi[col].values
            
            hypocenter_file = os.path.join(ply_dir, 'hypocenters.ply')
            hypocenter_pcd.save(hypocenter_file)
            print(f"  ✓ Saved hypocenter point cloud: hypocenters.ply ({len(df_hyfi)} points)")
        except Exception as e:
            print(f"  Warning: Could not export hypocenter point cloud as PLY: {e}")
    
    # Export focal mechanism meshes if available
    if df_hyfi is not None:
        try:
            focal_cols = ['Strike1', 'Dip1', 'Strike2', 'Dip2', 'pref_foc']
            if all(col in df_hyfi.columns for col in focal_cols):
                valid_focal_mask = (
                    df_hyfi['pref_foc'].notna() &
                    df_hyfi['pref_foc'].isin([1, 2]) &
                    (((df_hyfi['pref_foc'] == 1) & df_hyfi['Strike1'].notna() & df_hyfi['Dip1'].notna()) |
                     ((df_hyfi['pref_foc'] == 2) & df_hyfi['Strike2'].notna() & df_hyfi['Dip2'].notna()))
                )
                
                focal_events = df_hyfi[valid_focal_mask].copy()
                
                if len(focal_events) > 0:
                    print(f"  Exporting focal mechanism meshes...")
                    focal_meshes = _create_focal_mechanism_disc_meshes(focal_events)
                    
                    if focal_meshes:
                        combined_focal_mesh = None
                        
                        for mesh_info in focal_meshes:
                            try:
                                mesh = mesh_info['mesh']
                                
                                if mesh is not None and mesh.n_points > 0:
                                    if combined_focal_mesh is None:
                                        combined_focal_mesh = mesh.copy()
                                    else:
                                        combined_focal_mesh = combined_focal_mesh.merge(mesh)
                            except Exception as e:
                                continue
                        
                        if combined_focal_mesh is not None and combined_focal_mesh.n_points > 0:
                            combined_focal_file = os.path.join(ply_dir, 'focals_compiled.ply')
                            combined_focal_mesh.save(combined_focal_file)
                            print(f"  ✓ Saved focal mechanisms: focals_compiled.ply ({combined_focal_mesh.n_points} vertices)")
        
        except Exception as e:
            print(f"  Warning: Could not export focal mechanisms as PLY: {e}")
    
    # Export slip vectors if available
    if df_hyfi is not None:
        try:
            required_cols = ['X', 'Y', 'Z', 'nor_x_mean', 'nor_y_mean', 'nor_z_mean', 'rupt_radius', 'rake']
            df_valid = df_hyfi.dropna(subset=required_cols).copy()
            
            if len(df_valid) > 0:
                print(f"  Exporting slip vectors...")
                
                # Create slip vector line segments
                line_points = []
                line_connections = []
                
                for i, (pt, row) in enumerate(zip(df_valid[['X', 'Y', 'Z']].values, df_valid.iterrows())):
                    _, row_data = row
                    x, y, z = pt
                    p = [x, y, z]
                    r = row_data['rupt_radius']
                    nor = np.array([row_data['nor_x_mean'], row_data['nor_y_mean'], row_data['nor_z_mean']])
                    rake = row_data['rake']
                    
                    # Calculate slip vector endpoint
                    u, v, w = utilities_plot.slipvector_3D(p, r, nor, rake)
                    
                    # Bidirectional vector
                    slip_vec = np.array([2 * (u - x), 2 * (v - y), 2 * (w - z)])
                    
                    start = pt - slip_vec / 2
                    end = pt + slip_vec / 2
                    
                    line_points.append(start)
                    line_points.append(end)
                    line_connections.append([2, len(line_points)-2, len(line_points)-1])
                
                line_points = np.array(line_points, dtype=np.float64)
                line_connections = np.hstack(line_connections)
                
                # Create line mesh and apply tube filter
                line_mesh = pv.PolyData(line_points, lines=line_connections)
                tube_radius = 2.0
                tubes = line_mesh.tube(radius=tube_radius, n_sides=8)
                
                # Save as PLY
                slip_vectors_file = os.path.join(ply_dir, 'slip_vectors.ply')
                tubes.save(slip_vectors_file)
                print(f"  ✓ Saved slip vectors: slip_vectors.ply ({tubes.n_points} vertices)")
        
        except Exception as e:
            print(f"  Warning: Could not export slip vectors as PLY: {e}")
    
    print(f"PLY export complete. Files saved to: {ply_dir}")


def export_basic_vtp(df_hyfi, output_dir, fault_disc_meshes=None, use_focal_constraints=False):
    """
    Export basic VTP files (hypocenters, rupture planes, focals, slip vectors) without interpolated meshes.
    
    This function is called when interpolation is disabled or fails, but we still want to export
    the fundamental data for visualization.
    
    Parameters
    ----------
    df_hyfi : DataFrame
        HyFI results dataframe with hypocenter data
    output_dir : str
        Output directory path
    fault_disc_meshes : list, optional
        List of circular disc mesh dictionaries for rupture planes.
        If not provided, they will be generated from the dataframe.
    use_focal_constraints : bool, default=False
        Whether to also export enhanced focal fault planes
    """
    print("Exporting basic VTP files (without interpolated meshes)...")
    
    # Create VTP output directory
    vtp_dir = os.path.join(output_dir, 'vtp_export')
    os.makedirs(vtp_dir, exist_ok=True)
    
    # If fault_disc_meshes not provided, generate them on the fly
    if fault_disc_meshes is None or len(fault_disc_meshes) == 0:
        if df_hyfi is not None and len(df_hyfi) > 0:
            # Generate circular disc meshes for all events with valid fault plane data
            try:
                required_cols = ['nor_x_mean', 'nor_y_mean', 'nor_z_mean', 'rupt_radius']
                df_valid = df_hyfi.dropna(subset=required_cols)
                if len(df_valid) > 0:
                    fault_disc_meshes = _create_circular_fault_disc_meshes(df_valid)
                    print(f"  Generated {len(fault_disc_meshes)} circular disc meshes from event fault planes")
                else:
                    fault_disc_meshes = []
            except Exception as e:
                print(f"  Warning: Could not generate circular disc meshes: {e}")
                fault_disc_meshes = []
        else:
            fault_disc_meshes = []
    
    # Export hypocenters
    if df_hyfi is not None and len(df_hyfi) > 0:
        # Create point cloud from hypocenter coordinates
        points = df_hyfi[['X', 'Y', 'Z']].values
        hypocenter_pcd = pv.PolyData(points)
        
        # Get column name mapping for VTP export
        rename_map = _get_vtp_column_name_mapping()
        
        # Specified columns for hypocenters
        specified_columns = ['ID', 'LAT', 'LON', 'DEPTH', 'X', 'Y', 'Z', 'EX', 'EY', 'EZ',
                           'YR', 'MO', 'DY', 'HR', 'MI', 'SC', 'MAG',
                           'NCCP', 'NCCS', 'NCTP', 'NCTS', 'RCC', 'RCT', 'CID']
        
        # Always include temporal attributes if available
        temporal_columns = ['unix_timestamp', 'days_since_first', 'decimal_year']
        
        # Add specified columns
        for col in specified_columns + temporal_columns:
            if col in df_hyfi.columns:
                try:
                    # Use renamed column name for VTP attribute
                    vtp_col_name = rename_map.get(col, col)
                    hypocenter_pcd[vtp_col_name] = df_hyfi[col].values
                except Exception as e:
                    pass
        
        # Add temporal data if Date column exists
        if 'Date' in df_hyfi.columns:
            try:
                dates = pd.to_datetime(df_hyfi['Date'])
                min_date = dates.min()
                days_since_first = (dates - min_date).dt.days.values
                hypocenter_pcd['days_since_first'] = days_since_first
                
                unix_timestamps = dates.astype('int64') // 10**9
                hypocenter_pcd['unix_timestamp'] = unix_timestamps.astype(float)
                
                decimal_years = dates.dt.year + dates.dt.dayofyear / 365.25
                hypocenter_pcd['decimal_year'] = decimal_years.values
                
                hypocenter_pcd['month'] = dates.dt.month.values
                
                print(f"    Added temporal data spanning {days_since_first.max()} days")
            except Exception as e:
                print(f"    Warning: Could not add Date column to VTP: {e}")
        
        hypocenter_file = os.path.join(vtp_dir, 'hypocenters.vtp')
        hypocenter_pcd.save(hypocenter_file)
        print(f"  Saved hypocenter point cloud: {hypocenter_file} ({len(df_hyfi)} points)")
    
    # Export rupture plane meshes
    if fault_disc_meshes is not None and len(fault_disc_meshes) > 0:
        print(f"  Exporting {len(fault_disc_meshes)} rupture plane meshes...")
        
        try:
            combined_disc_mesh = None
            for i, disc_info in enumerate(fault_disc_meshes):
                try:
                    disc_mesh = disc_info.get('mesh') if hasattr(disc_info, 'get') else disc_info
                    
                    if disc_mesh is not None and hasattr(disc_mesh, 'n_points') and disc_mesh.n_points > 0:
                        if combined_disc_mesh is None:
                            combined_disc_mesh = disc_mesh.copy()
                        else:
                            combined_disc_mesh = combined_disc_mesh.merge(disc_mesh)
                except Exception as e:
                    print(f"    Warning: Failed to add disc mesh {i}: {e}")
                    continue
            
            if combined_disc_mesh is not None and combined_disc_mesh.n_points > 0:
                combined_disc_file = os.path.join(vtp_dir, 'rupture_planes.vtp')
                combined_disc_mesh.save(combined_disc_file)
                print(f"  Saved rupture planes: {combined_disc_file}")
        except Exception as e:
            print(f"  Warning: Failed to create combined disc mesh: {e}")
    
    # Export enhanced focal fault planes if requested
    if use_focal_constraints and df_hyfi is not None:
        export_enhanced_focal_planes_vtp(df_hyfi, output_dir, use_focal_constraints)
    
    # Export slip vectors
    if df_hyfi is not None:
        export_slip_vectors_vtp(df_hyfi, output_dir)
    
    print(f"Basic VTP export complete. Files saved to: {vtp_dir}")


def export_interpolated_planes_vtp(combined_mesh, individual_meshes, point_cloud, output_dir, fault_disc_meshes=None, df_hyfi=None, use_focal_constraints=False, export_obj=False):
    """
    Export interpolated planes, original hypocenters, and circular disc meshes as VTP files.
    
    Parameters
    ----------
    combined_mesh : pyvista.PolyData
        Combined mesh of all fault planes
    individual_meshes : list
        List of individual mesh dictionaries
    point_cloud : pyvista.PolyData
        Combined point cloud of fault plane points
    output_dir : str
        Output directory path
    fault_disc_meshes : list, optional
        List of circular disc mesh dictionaries
    df_hyfi : DataFrame, optional
        Original hypocenter dataframe for creating hypocenter point cloud
    use_focal_constraints : bool, default=False
        Whether to also export enhanced focal fault planes
    export_obj : bool, default=False
        Whether to export meshes as OBJ files
    """
    if combined_mesh is None:
        return
    
    print("Exporting interpolated planes to VTP files...")
    
    # Create interpolation output directory
    vtp_dir = os.path.join(output_dir, 'vtp_export')
    os.makedirs(vtp_dir, exist_ok=True)
    
    # Clean up old VTP files to avoid confusion from previous runs
    import glob
    old_vtp_files = glob.glob(os.path.join(vtp_dir, 'fault_*.vtp'))
    if old_vtp_files:
        print(f"  Cleaning up {len(old_vtp_files)} old fault VTP files...")
        for old_file in old_vtp_files:
            try:
                os.remove(old_file)
            except Exception as e:
                print(f"  Warning: Could not remove {old_file}: {e}")
    
    # Export individual meshes with enhanced information (process first to add metadata)
    for mesh_info in individual_meshes:
        # Work on a copy to avoid modifying the original mesh
        mesh = mesh_info['mesh'].copy()
        ori_cluster = mesh_info['orientation_cluster']
        spatial_cluster = mesh_info['spatial_cluster']
        fault_idx = mesh_info['fault_idx']
        n_fault_planes = mesh_info.get('n_fault_planes', 'unknown')
        n_input_points = mesh_info.get('n_input_points', 'unknown')
        area_m2 = mesh_info.get('area_m2', 'unknown')
        max_Mw = mesh_info.get('max_Mw', 'unknown')
        
        # Use permanent FS ID for VTP filename
        cluster_id = mesh_info['cluster_id']
        filename = f"fault_{cluster_id}.vtp"
        
        # Add metadata as cell data attributes (for each triangle/face)
        if mesh.n_cells > 0:
            # Get metadata from mesh_info
            fs_id = mesh_info.get('cluster_id', cluster_id)
            segmentation_level = mesh_info.get('segmentation_level')
            sequence_label = mesh_info.get('sequence_label')
            
            # Add scalar attributes (same value for all cells)
            if fs_id is not None:
                mesh.cell_data['fault_id'] = np.full(mesh.n_cells, str(fs_id))
            if segmentation_level is not None:
                mesh.cell_data['segmentation_level'] = np.full(mesh.n_cells, segmentation_level)
            if sequence_label is not None:
                mesh.cell_data['sequence_label'] = np.full(mesh.n_cells, str(sequence_label))
            
            mesh.cell_data['vtp_file'] = np.full(mesh.n_cells, filename)
            mesh.cell_data['n_events'] = np.full(mesh.n_cells, mesh_info.get('n_events', n_fault_planes))
            
            # Centroid coordinates
            if 'centroid_x' in mesh_info and mesh_info['centroid_x'] is not None:
                mesh.cell_data['centroid_x'] = np.full(mesh.n_cells, mesh_info['centroid_x'])
            if 'centroid_y' in mesh_info and mesh_info['centroid_y'] is not None:
                mesh.cell_data['centroid_y'] = np.full(mesh.n_cells, mesh_info['centroid_y'])
            if 'centroid_z' in mesh_info and mesh_info['centroid_z'] is not None:
                mesh.cell_data['centroid_z'] = np.full(mesh.n_cells, mesh_info['centroid_z'])
            
            # Rupture plane mean orientation
            if 'rupture_mean_azimuth' in mesh_info and mesh_info['rupture_mean_azimuth'] is not None:
                mesh.cell_data['rupture_mean_azimuth'] = np.full(mesh.n_cells, mesh_info['rupture_mean_azimuth'])
            if 'rupture_mean_dip' in mesh_info and mesh_info['rupture_mean_dip'] is not None:
                mesh.cell_data['rupture_mean_dip'] = np.full(mesh.n_cells, mesh_info['rupture_mean_dip'])
            
            # Mesh mean orientation
            if 'mesh_mean_azimuth' in mesh_info and mesh_info['mesh_mean_azimuth'] is not None:
                mesh.cell_data['mesh_mean_azimuth'] = np.full(mesh.n_cells, mesh_info['mesh_mean_azimuth'])
            if 'mesh_mean_dip' in mesh_info and mesh_info['mesh_mean_dip'] is not None:
                mesh.cell_data['mesh_mean_dip'] = np.full(mesh.n_cells, mesh_info['mesh_mean_dip'])
            
            # Mesh properties
            if isinstance(area_m2, (int, float)):
                mesh.cell_data['mesh_area_m2'] = np.full(mesh.n_cells, area_m2)
            if isinstance(max_Mw, (int, float)) and not np.isnan(max_Mw):
                mesh.cell_data['max_mag'] = np.full(mesh.n_cells, max_Mw)
            
            mesh.cell_data['mesh_vertices'] = np.full(mesh.n_cells, mesh.n_points)
            mesh.cell_data['mesh_faces'] = np.full(mesh.n_cells, mesh.n_cells)
            
            # Rupture mean stress parameters
            if 'rupture_mean_instability' in mesh_info and mesh_info['rupture_mean_instability'] is not None:
                mesh.cell_data['rupture_mean_instability'] = np.full(mesh.n_cells, mesh_info['rupture_mean_instability'])
            if 'rupture_mean_sliptend' in mesh_info and mesh_info['rupture_mean_sliptend'] is not None:
                mesh.cell_data['rupture_mean_sliptend'] = np.full(mesh.n_cells, mesh_info['rupture_mean_sliptend'])
            if 'rupture_mean_dilatend' in mesh_info and mesh_info['rupture_mean_dilatend'] is not None:
                mesh.cell_data['rupture_mean_dilatend'] = np.full(mesh.n_cells, mesh_info['rupture_mean_dilatend'])
            
            # Mesh mean stress parameters (from cell data means)
            if 'mesh_mean_instability' in mesh_info and mesh_info['mesh_mean_instability'] is not None:
                mesh.cell_data['mesh_mean_instability'] = np.full(mesh.n_cells, mesh_info['mesh_mean_instability'])
            if 'mesh_mean_sliptend' in mesh_info and mesh_info['mesh_mean_sliptend'] is not None:
                mesh.cell_data['mesh_mean_sliptend'] = np.full(mesh.n_cells, mesh_info['mesh_mean_sliptend'])
            if 'mesh_mean_dilatend' in mesh_info and mesh_info['mesh_mean_dilatend'] is not None:
                mesh.cell_data['mesh_mean_dilatend'] = np.full(mesh.n_cells, mesh_info['mesh_mean_dilatend'])
        
        # Note: Cell data (stress params for each face) is already added by stress_analysis module
        # Rename cell data columns to match HyFI_results.csv naming
        rename_map = _get_vtp_column_name_mapping()
        if mesh.n_cells > 0:
            # Rename cell data attributes if they exist
            for old_name, new_name in rename_map.items():
                if old_name in mesh.cell_data:
                    mesh.cell_data[new_name] = mesh.cell_data[old_name]
                    # Remove old name to avoid duplication
                    del mesh.cell_data[old_name]
        
        # Remove all point data attributes (keep only coordinates)
        mesh.clear_point_data()
        
        # Store the modified mesh back in mesh_info for combined mesh rebuild
        mesh_info['mesh_with_metadata'] = mesh
        
        filepath = os.path.join(vtp_dir, filename)
        mesh.save(filepath)
        if isinstance(max_Mw, (int, float)) and not np.isnan(max_Mw):
            print(f"  Saved mesh: {filename} ({n_fault_planes} fault planes, {n_input_points} input points, {area_m2:.1f} m² area, max Mw {max_Mw:.2f})")
        else:
            print(f"  Saved mesh: {filename} ({n_fault_planes} fault planes, {n_input_points} input points, {area_m2:.1f} m² area)")
    
    # Now rebuild combined mesh with all metadata from individual meshes
    if individual_meshes:
        print("  Rebuilding combined mesh with metadata...")
        rebuilt_combined_mesh = None
        
        # First, collect all unique cell data keys from all meshes
        all_cell_data_keys = set()
        for mesh_info in individual_meshes:
            mesh = mesh_info.get('mesh_with_metadata')
            if mesh and mesh.n_cells > 0:
                all_cell_data_keys.update(mesh.cell_data.keys())
        
        # Ensure all meshes have all cell data keys (fill missing with NaN)
        for mesh_info in individual_meshes:
            mesh = mesh_info.get('mesh_with_metadata')
            if mesh and mesh.n_cells > 0:
                for key in all_cell_data_keys:
                    if key not in mesh.cell_data:
                        # Add missing key with NaN values
                        mesh.cell_data[key] = np.full(mesh.n_cells, np.nan)
        
        # Now merge meshes
        for mesh_info in individual_meshes:
            mesh = mesh_info.get('mesh_with_metadata')
            if mesh and mesh.n_points > 0:
                if rebuilt_combined_mesh is None:
                    rebuilt_combined_mesh = mesh.copy()
                else:
                    rebuilt_combined_mesh = rebuilt_combined_mesh + mesh
        
        # Save rebuilt combined mesh
        if rebuilt_combined_mesh is not None and rebuilt_combined_mesh.n_points > 0:
            # Clear point data (keep only coordinates)
            rebuilt_combined_mesh.clear_point_data()
            
            combined_file = os.path.join(vtp_dir, 'faults_compiled.vtp')
            rebuilt_combined_mesh.save(combined_file)
            print(f"  Saved combined mesh: {combined_file} ({rebuilt_combined_mesh.n_cells} faces with {len(rebuilt_combined_mesh.cell_data.keys())} attributes)")
        
    # Print area and magnitude summary
    if individual_meshes:
        areas = [mesh_info.get('area_m2', 0) for mesh_info in individual_meshes if isinstance(mesh_info.get('area_m2'), (int, float))]
        magnitudes = [mesh_info.get('max_Mw', 0) for mesh_info in individual_meshes if isinstance(mesh_info.get('max_Mw'), (int, float)) and not np.isnan(mesh_info.get('max_Mw', np.nan))]      
            
        # Export area and magnitude summary to CSV
        if areas or magnitudes:
            summary_data = []
            for mesh_info in individual_meshes:
                summary_data.append({
                    'cluster_id': mesh_info['cluster_id'],
                    'fault_idx': mesh_info['fault_idx'],
                    'n_fault_planes': mesh_info['n_fault_planes'],
                    'n_input_points': mesh_info['n_input_points'],
                    'mesh_vertices': mesh_info['mesh'].n_points,
                    'mesh_faces': mesh_info['mesh'].n_cells,
                    'area_m2': mesh_info.get('area_m2', 0),
                    'max_Mw': mesh_info.get('max_Mw', np.nan),
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(output_dir, 'interpolated_faults_summary.csv')
            summary_df.to_csv(summary_file, index=False)
            print(f"    Area and magnitude summary saved to: {summary_file}")


    # Export original hypocenter point cloud
    if df_hyfi is not None and len(df_hyfi) > 0:
        
        # Create point cloud from original hypocenters
        hypocenter_points = df_hyfi[['X', 'Y', 'Z']].values
        hypocenter_pcd = pv.PolyData(hypocenter_points)
        
        # Add hypocenter data as point arrays with proper renaming
        rename_map = _get_vtp_column_name_mapping()
        
        # Specified columns for hypocenters
        specified_columns = ['ID', 'LAT', 'LON', 'DEPTH', 'X', 'Y', 'Z', 'EX', 'EY', 'EZ',
                           'YR', 'MO', 'DY', 'HR', 'MI', 'SC', 'MAG',
                           'NCCP', 'NCCS', 'NCTP', 'NCTS', 'RCC', 'RCT', 'CID']
        
        # Always include temporal attributes if available
        temporal_columns = ['unix_timestamp', 'days_since_first', 'decimal_year']
        
        # Add specified columns
        for col in specified_columns + temporal_columns:
            if col in df_hyfi.columns:
                try:
                    # Use renamed column name for VTP attribute
                    vtp_col_name = rename_map.get(col, col)
                    hypocenter_pcd[vtp_col_name] = df_hyfi[col].values
                except:
                    pass
        
        # Handle Date column separately - convert datetime to numeric formats for ParaView compatibility
        if 'Date' in df_hyfi.columns:
            try:
                # Convert to datetime if not already
                dates = pd.to_datetime(df_hyfi['Date'])
                                
                # Days since first event (good for temporal coloring)
                min_date = dates.min()
                days_since_first = (dates - min_date).dt.days.values
                hypocenter_pcd['days_since_first'] = days_since_first
                
                # Unix timestamp (preserves full temporal information)
                unix_timestamps = dates.astype('int64') // 10**9  # Convert to seconds
                hypocenter_pcd['unix_timestamp'] = unix_timestamps.astype(float)
                
                # Year as decimal (good for multi-year datasets)
                decimal_years = dates.dt.year + dates.dt.dayofyear / 365.25
                hypocenter_pcd['decimal_year'] = decimal_years.values
                
                # Month number (good for seasonal analysis)
                hypocenter_pcd['month'] = dates.dt.month.values
                
                print(f"    Added temporal data: {len(days_since_first)} events from {min_date.strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
                print(f"    Date range: {days_since_first.max()} days total")
                
            except Exception as e:
                print(f"    Warning: Could not add Date column to VTP: {e}")
        
        # Add fault plane data if available
        for col in ['final_cluster_id']:
            if col in df_hyfi.columns:
                try:
                    # Convert None and NaN to consistent np.nan values
                    values = df_hyfi[col].replace('None', np.nan).replace([None], np.nan).values
                    hypocenter_pcd[col] = values
                except Exception as e:
                    print(f"    Warning: Could not add {col} column to VTP: {e}")
        
        hypocenter_file = os.path.join(vtp_dir, 'hypocenters.vtp')
        hypocenter_pcd.save(hypocenter_file)
        print(f"  Saved hypocenter point cloud: {hypocenter_file} ({len(df_hyfi)} points)")
    
    # Also export fault plane point cloud for reference
    if point_cloud is not None and point_cloud.n_points > 0:
        pcd_file = os.path.join(vtp_dir, 'enhanced_pointcloud.vtp')
        point_cloud.save(pcd_file)
        print(f"  Saved fault plane point cloud: {pcd_file}")
    
    # Export rupture plane meshes - combined only
    if fault_disc_meshes is not None and len(fault_disc_meshes) > 0:
        print(f"  Exporting {len(fault_disc_meshes)} rupture plane meshes as combined file...")

        # Create combined rupture plane mesh directly (skip individual exports)
        try:
            combined_disc_mesh = None
            valid_combined_meshes = 0
            
            for i, disc_info in enumerate(fault_disc_meshes):
                try:
                    # Handle both dictionary format and direct mesh format
                    if hasattr(disc_info, 'get'):  # Dictionary format
                        disc_mesh = disc_info.get('mesh')
                    else:  # Direct mesh format
                        disc_mesh = disc_info
                    
                    if disc_mesh is not None and hasattr(disc_mesh, 'n_points') and disc_mesh.n_points > 0:
                        # Keep the original mesh with all its attributes instead of creating a clean copy
                        # This preserves all the attribute data we added to each disc
                        
                        if combined_disc_mesh is None:
                            combined_disc_mesh = disc_mesh.copy()
                        else:
                            # Use merge method to combine meshes while preserving attributes
                            try:
                                combined_disc_mesh = combined_disc_mesh.merge(disc_mesh)
                            except Exception as merge_error:
                                print(f"        Merge failed ({merge_error}), trying manual combination...")
                                # Fallback: manually combine points and faces
                                points1 = combined_disc_mesh.points
                                points2 = disc_mesh.points
                                faces1 = combined_disc_mesh.faces
                                faces2 = disc_mesh.faces
                                
                                # Offset face indices for second mesh
                                n_points1 = len(points1)
                                faces2_offset = faces2.copy()
                                # Adjust face indices (skip the first element which is the number of points per face)
                                for j in range(1, len(faces2_offset)):
                                    if j % 4 != 0:  # Skip count elements (assuming triangular faces)
                                        faces2_offset[j] += n_points1
                                
                                combined_points = np.vstack([points1, points2])
                                combined_faces = np.hstack([faces1, faces2_offset])
                                new_combined_mesh = pv.PolyData(combined_points, combined_faces)
                                
                                # Manually combine attribute arrays
                                for key in combined_disc_mesh.point_data.keys():
                                    if key in disc_mesh.point_data.keys():
                                        combined_attr = np.hstack([
                                            combined_disc_mesh.point_data[key],
                                            disc_mesh.point_data[key]
                                        ])
                                        new_combined_mesh[key] = combined_attr
                                
                                combined_disc_mesh = new_combined_mesh
                        
                        valid_combined_meshes += 1
                except Exception as e:
                    print(f"    Warning: Failed to add disc mesh {i} to combined mesh: {e}")
                    continue
            
            if combined_disc_mesh is not None and combined_disc_mesh.n_points > 0:
                combined_disc_file = os.path.join(vtp_dir, 'rupture_planes.vtp')
                combined_disc_mesh.save(combined_disc_file)
                print(f"  Saved combined disc mesh: {combined_disc_file} ({valid_combined_meshes} discs)")
            else:
                print(f"  Warning: No valid disc meshes to combine")
                
        except Exception as e:
            print(f"  Warning: Failed to create combined disc mesh: {e}")
    
    # Export enhanced focal fault planes if requested and available
    if use_focal_constraints and df_hyfi is not None:
        export_enhanced_focal_planes_vtp(df_hyfi, output_dir, use_focal_constraints)
    
    # Export slip vectors if df_hyfi is available
    if df_hyfi is not None:
        export_slip_vectors_vtp(df_hyfi, output_dir)
    
    # Export as OBJ files if requested
    if export_obj:
        export_meshes_as_obj(combined_mesh, individual_meshes, fault_disc_meshes, output_dir, df_hyfi)
    
    print(f"VTP export complete. Files saved to: {vtp_dir}")


def _create_focal_mechanism_disc_meshes(df_focal_events, n_radial_segments=16, n_rings=5, default_radius=100.0):
    """
    Create a single combined disc mesh for all focal mechanism planes.

    Uses pre-allocated numpy arrays to avoid per-event pv.PolyData() calls,
    giving orders-of-magnitude speedup over the old iterrows approach.

    Returns
    -------
    pv.PolyData or None
        A single combined mesh with all valid focal discs, or None if no valid events.
    """
    import sys as _sys_foc
    n = len(df_focal_events)
    _sys_foc.stderr.write(f"[focal_disc] Building combined disc mesh for {n} focal events...\n")
    _sys_foc.stderr.flush()

    # --- geometry constants -------------------------------------------
    n_seg   = n_radial_segments
    n_verts_d = 1 + n_rings * n_seg          # center + ring vertices
    # faces: center→ring1 (n_seg tris) + (n_rings-1) pairs * n_seg * 2 tris
    n_faces_d = n_seg + (n_rings - 1) * n_seg * 2

    # --- build template face array (offsets applied per event later) --
    _template_faces = np.empty((n_faces_d, 3), dtype=np.int64)
    fidx = 0
    # center to first ring
    for s in range(n_seg):
        _template_faces[fidx] = [0, 1 + s, 1 + (s + 1) % n_seg]
        fidx += 1
    # between rings
    for ri in range(n_rings - 1):
        rs  = 1 + ri * n_seg
        nrs = 1 + (ri + 1) * n_seg
        for s in range(n_seg):
            ns = (s + 1) % n_seg
            _template_faces[fidx]     = [rs + s, nrs + s, rs + ns]
            _template_faces[fidx + 1] = [rs + ns, nrs + s, nrs + ns]
            fidx += 2

    # ring-radius fractions for vertex generation
    _ring_fracs = np.array([(ri + 1) / n_rings for ri in range(n_rings)])
    _angles = np.array([2 * np.pi * s / n_seg for s in range(n_seg)])
    _cos_a  = np.cos(_angles)
    _sin_a  = np.sin(_angles)

    # --- extract columns -----------------------------------------------
    df = df_focal_events.reset_index(drop=True)

    X_arr = pd.to_numeric(df['X'],  errors='coerce').values
    Y_arr = pd.to_numeric(df['Y'],  errors='coerce').values
    Z_arr = pd.to_numeric(df['Z'],  errors='coerce').values
    pf_arr = pd.to_numeric(df['pref_foc'], errors='coerce').values

    S1_arr = pd.to_numeric(df['Strike1'], errors='coerce').values
    D1_arr = pd.to_numeric(df['Dip1'],    errors='coerce').values
    S2_arr = pd.to_numeric(df['Strike2'], errors='coerce').values
    D2_arr = pd.to_numeric(df['Dip2'],    errors='coerce').values

    R1_arr = pd.to_numeric(df.get('Rake1', pd.Series(np.nan, index=df.index)), errors='coerce').values if 'Rake1' in df.columns else np.full(n, np.nan)
    R2_arr = pd.to_numeric(df.get('Rake2', pd.Series(np.nan, index=df.index)), errors='coerce').values if 'Rake2' in df.columns else np.full(n, np.nan)

    # Select strike/dip/rake by pref_foc
    strike_arr = np.where(pf_arr == 1, S1_arr, S2_arr)
    dip_arr    = np.where(pf_arr == 1, D1_arr, D2_arr)
    rake_arr   = np.where(pf_arr == 1, R1_arr, R2_arr)

    R_arr = pd.to_numeric(df['rupt_radius'], errors='coerce').values

    # --- valid mask ----------------------------------------------------
    valid_mask = (
        np.isfinite(X_arr) & np.isfinite(Y_arr) & np.isfinite(Z_arr) &
        np.isfinite(strike_arr) & np.isfinite(dip_arr) &
        np.isfinite(R_arr) & (R_arr > 0)
    )

    # Outlier radius cap: skip discs whose radius exceeds 100× the 99th percentile
    _r_valid = R_arr[valid_mask & np.isfinite(R_arr)]
    if len(_r_valid) > 0:
        _r99 = float(np.percentile(_r_valid, 99))
        _r_cap = _r99 * 100.0
        n_capped = int(np.sum(valid_mask & (R_arr > _r_cap)))
        if n_capped:
            _sys_foc.stderr.write(f"[focal_disc] Skipping {n_capped} events with radius > {_r_cap:.1f}m\n")
            _sys_foc.stderr.flush()
        valid_mask = valid_mask & (R_arr <= _r_cap)

    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)

    if n_valid == 0:
        _sys_foc.stderr.write("[focal_disc] No valid focal disc events found.\n")
        _sys_foc.stderr.flush()
        return None

    _sys_foc.stderr.write(f"[focal_disc] Building {n_valid} discs ({n - n_valid} skipped)...\n")
    _sys_foc.stderr.flush()

    # --- pre-allocate combined arrays ---------------------------------
    all_verts = np.empty((n_valid * n_verts_d, 3), dtype=np.float64)
    all_faces = np.empty((n_valid * n_faces_d, 3), dtype=np.int64)

    # per-point scalar arrays (numeric)
    _s_X  = np.empty(n_valid * n_verts_d, dtype=np.float64)
    _s_Y  = np.empty(n_valid * n_verts_d, dtype=np.float64)
    _s_Z  = np.empty(n_valid * n_verts_d, dtype=np.float64)
    _s_R  = np.empty(n_valid * n_verts_d, dtype=np.float64)
    _s_area = np.empty(n_valid * n_verts_d, dtype=np.float64)
    _s_strike = np.empty(n_valid * n_verts_d, dtype=np.float64)
    _s_dip    = np.empty(n_valid * n_verts_d, dtype=np.float64)
    _s_rake   = np.empty(n_valid * n_verts_d, dtype=np.float64)
    _s_pf     = np.empty(n_valid * n_verts_d, dtype=np.float64)

    # Optional numeric columns collected at the end
    _opt_num_cols = {}
    for _col in ['LAT', 'LON', 'DEPTH', 'Mw', 'A', 'epsilon', 'Strike1', 'Dip1', 'Rake1',
                 'Strike2', 'Dip2', 'Rake2', 'unix_timestamp', 'days_since_first', 'decimal_year',
                 'segmentation_level']:
        if _col in df.columns:
            _opt_num_cols[_col] = np.empty(n_valid * n_verts_d, dtype=np.float64)

    # --- fill arrays event by event (geometry only in loop) -----------
    for _li, _ei in enumerate(valid_indices):
        _voff = _li * n_verts_d
        _foff = _li * n_faces_d

        cx, cy, cz = X_arr[_ei], Y_arr[_ei], Z_arr[_ei]
        r  = R_arr[_ei]
        st = strike_arr[_ei]
        dp = dip_arr[_ei]

        # Normal vector
        _dip_azi = np.radians((st + 90.0) % 360.0)
        _dip_rad = np.radians(dp)
        _nx = np.sin(_dip_rad) * np.sin(_dip_azi)
        _ny = np.sin(_dip_rad) * np.cos(_dip_azi)
        _nz = np.cos(_dip_rad)
        _n  = np.array([_nx, _ny, _nz])
        _n /= np.linalg.norm(_n)

        # In-plane orthonormal basis
        _ref = np.array([0.0, 0.0, 1.0]) if abs(_n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        _v1 = np.cross(_n, _ref); _v1 /= np.linalg.norm(_v1)
        _v2 = np.cross(_n, _v1); _v2 /= np.linalg.norm(_v2)

        # Center vertex
        all_verts[_voff] = [cx, cy, cz]

        # Ring vertices (vectorized per ring)
        for _ri, _rf in enumerate(_ring_fracs):
            _rr = r * _rf
            _base = _voff + 1 + _ri * n_seg
            # shape (n_seg, 3)
            _pts = (_rr * (_cos_a[:, None] * _v1 + _sin_a[:, None] * _v2)
                    + np.array([cx, cy, cz]))
            all_verts[_base:_base + n_seg] = _pts

        # Faces (template + per-event vertex offset)
        all_faces[_foff:_foff + n_faces_d] = _template_faces + _li * n_verts_d

        # Scalars (filled via np.repeat below is cleaner — just store per-event values)
        _area = np.pi * r * r
        _sl = slice(_voff, _voff + n_verts_d)
        _s_X[_sl]      = cx
        _s_Y[_sl]      = cy
        _s_Z[_sl]      = cz
        _s_R[_sl]      = r
        _s_area[_sl]   = _area
        _s_strike[_sl] = st
        _s_dip[_sl]    = dp
        _s_rake[_sl]   = rake_arr[_ei]
        _s_pf[_sl]     = pf_arr[_ei]
        for _col, _arr in _opt_num_cols.items():
            _arr[_sl] = pd.to_numeric(df.loc[_ei, _col], errors='coerce') if _col in df.columns else np.nan

    # --- build single PolyData ----------------------------------------
    # PyVista faces format: [3, i0, i1, i2, 3, ...]
    faces_pv = np.hstack([np.full((n_valid * n_faces_d, 1), 3, dtype=np.int64), all_faces])
    combined = pv.PolyData(all_verts, faces_pv.flatten())

    combined['X']      = _s_X
    combined['Y']      = _s_Y
    combined['Z']      = _s_Z
    combined['hypocenter_x'] = _s_X
    combined['hypocenter_y'] = _s_Y
    combined['hypocenter_z'] = _s_Z
    combined['radius'] = _s_R
    combined['area_m2'] = _s_area
    combined['Strike'] = _s_strike
    combined['Dip']    = _s_dip
    combined['Rake']   = _s_rake
    combined['pref_foc'] = _s_pf

    for _col, _arr in _opt_num_cols.items():
        combined[_col] = _arr

    # Date / temporal
    if 'Date' in df.columns:
        _dates = pd.to_datetime(df['Date'].iloc[valid_indices], errors='coerce')
        _ts    = _dates.apply(lambda d: d.timestamp() if not pd.isna(d) else np.nan).values
        combined['Date'] = np.repeat(_ts, n_verts_d)

    # String columns (VTK requires uniform dtype per array — store as float ID_num)
    combined['ID_num'] = np.repeat(valid_indices.astype(np.float64), n_verts_d)

    _sys_foc.stderr.write(f"[focal_disc] Done — combined mesh: {combined.n_points} pts, {combined.n_cells} faces\n")
    _sys_foc.stderr.flush()
    return combined


def export_enhanced_focal_planes_vtp(df_hyfi, output_dir, use_focal_constraints=False):
    """
    Export enhanced focal fault planes as VTP mesh surfaces.
    
    This function generates circular disc mesh surfaces from focal mechanism data and exports them
    as VTP files for visualization and analysis, similar to rupture plane meshes.
    
    Parameters
    ----------
    df_hyfi : DataFrame
        Main hypocenter dataframe containing focal mechanism data
    output_dir : str
        Output directory path
    use_focal_constraints : bool, default=False
        Whether to actually generate and export focal mechanism meshes
    """
    if not use_focal_constraints:
        print("Enhanced focal planes export skipped (use_focal_constraints=False)")
        return
        
    # Check if focal mechanism data is available
    # Use pref_foc as single source of truth for which plane to export
    focal_cols = ['Strike1', 'Dip1', 'Strike2', 'Dip2', 'pref_foc']
    if not all(col in df_hyfi.columns for col in focal_cols):
        print("Enhanced focal planes export skipped (missing focal mechanism data or pref_foc column)")
        return
    
    # Filter events that have a preferred focal plane determined (from A or geometric selection)
    # pref_foc is set in model_validation: A=1/2 uses specified plane, A=0 uses best-fitting plane
    valid_focal_mask = (
        df_hyfi['pref_foc'].notna() &
        df_hyfi['pref_foc'].isin([1, 2]) &
        (((df_hyfi['pref_foc'] == 1) & df_hyfi['Strike1'].notna() & df_hyfi['Dip1'].notna()) |
         ((df_hyfi['pref_foc'] == 2) & df_hyfi['Strike2'].notna() & df_hyfi['Dip2'].notna()))
    )
    
    focal_events = df_hyfi[valid_focal_mask].copy()
    
    if len(focal_events) == 0:
        print("Enhanced focal planes export skipped (no valid focal mechanism events with pref_foc)")
        return
    
    # Report on active plane strategy
    if 'A' in focal_events.columns:
        n_specified = ((focal_events['A'] == 1) | (focal_events['A'] == 2)).sum()
        n_geometric = (focal_events['A'] == 0).sum()
        print(f"Exporting enhanced focal fault plane meshes for {len(focal_events)} events...")
        print(f"  - {n_specified} events with specified active plane (A=1 or A=2)")
        print(f"  - {n_geometric} events with geometrically determined plane (A=0 → best-fitting plane)")
    else:
        print(f"Exporting enhanced focal fault plane meshes for {len(focal_events)} events...")
    print(f"  Using 'pref_foc' column as single source of truth for plane selection")
    
    # Create focal planes output directory
    focal_dir = os.path.join(output_dir, 'vtp_export')
    os.makedirs(focal_dir, exist_ok=True)
    
    # Generate focal mechanism disc meshes (returns single combined PolyData)
    combined_focal_mesh = _create_focal_mechanism_disc_meshes(focal_events)

    if combined_focal_mesh is None or combined_focal_mesh.n_points == 0:
        print("No focal mechanism meshes generated")
        return

    # Export combined focal mechanism mesh
    combined_focal_file = os.path.join(focal_dir, 'focals_compiled.vtp')
    combined_focal_mesh.save(combined_focal_file)
    print(f"  Saved combined focal mechanism mesh: {combined_focal_file} "
          f"({combined_focal_mesh.n_cells} faces for {len(focal_events)} events)")
    print(f"Enhanced focal planes VTP export complete. Files saved to: {focal_dir}")




def export_slip_vectors_vtp(df_hyfi, output_dir):
    """
    Export slip vectors as line segments for ParaView visualization.
    
    Creates bidirectional line segments representing slip vectors at each hypocenter.
    Lines are exported with tube geometry for better 3D visibility.
    
    Parameters
    ----------
    df_hyfi : DataFrame
        Hypocenter dataframe with fault plane data
    output_dir : str
        Output directory path
    """
    print("Exporting slip vectors as line segments...")
    
    # Create VTP export directory
    vtp_dir = os.path.join(output_dir, 'vtp_export')
    os.makedirs(vtp_dir, exist_ok=True)
    
    # Filter events with valid slip vector data
    required_cols = ['X', 'Y', 'Z', 'nor_x_mean', 'nor_y_mean', 'nor_z_mean', 'rupt_radius', 'rake']
    df_valid = df_hyfi.dropna(subset=required_cols).copy()
    
    if len(df_valid) == 0:
        print("  No valid slip vector data available (missing required columns)")
        return
    
    print(f"  Creating slip vectors for {len(df_valid)} events...")
    
    # Get hypocenter locations
    points = df_valid[['X', 'Y', 'Z']].values.astype(np.float64)
    
    # Calculate slip vectors for each event
    from ..utils import utilities_plot
    
    slip_vectors = []
    for idx, row in df_valid.iterrows():
        x, y, z = row['X'], row['Y'], row['Z']
        p = [x, y, z]
        r = row['rupt_radius']
        nor = np.array([row['nor_x_mean'], row['nor_y_mean'], row['nor_z_mean']])
        rake = row['rake']
        
        # Calculate slip vector endpoint using existing slipvector_3D function
        u, v, w = utilities_plot.slipvector_3D(p, r, nor, rake)
        
        # Slip vector from hypocenter to endpoint (3D vector)
        # Bidirectional: multiply by 2 for full bidirectional display
        slip_vec = np.array([2 * (u - x), 2 * (v - y), 2 * (w - z)], dtype=np.float32)
        slip_vectors.append(slip_vec)
    
    slip_vectors = np.array(slip_vectors, dtype=np.float32)
    
    # Create bidirectional line segments
    line_points = []
    line_connections = []
    
    # Attributes for each line point (start and end)
    point_rake = []
    point_rupture_radius = []
    point_magnitude = []
    point_cluster_id = []
    point_event_id = []
    # New attributes from request
    point_lat = []
    point_lon = []
    point_depth = []
    point_x = []
    point_y = []
    point_z = []
    point_date = []
    point_sequence_label = []
    point_segmentation_level = []
    point_fault_id = []
    point_mw = []
    
    for i, (pt, vec) in enumerate(zip(points, slip_vectors)):
        row = df_valid.iloc[i]
        
        # Create bidirectional line: from -vec/2 to +vec/2
        start = pt - vec / 2
        end = pt + vec / 2
        
        line_points.append(start)
        line_points.append(end)
        line_connections.append([2, len(line_points)-2, len(line_points)-1])
        
        # Add attributes for both start and end points
        point_rake.extend([row['rake'], row['rake']])
        point_rupture_radius.extend([row['rupt_radius'], row['rupt_radius']])
        
        # Core location attributes (from request)
        if 'ID' in df_valid.columns:
            event_id = str(row['ID'])
            point_event_id.extend([event_id, event_id])
        
        if 'LAT' in df_valid.columns:
            lat = row['LAT'] if not pd.isna(row['LAT']) else 0.0
            point_lat.extend([lat, lat])
        
        if 'LON' in df_valid.columns:
            lon = row['LON'] if not pd.isna(row['LON']) else 0.0
            point_lon.extend([lon, lon])
        
        if 'DEPTH' in df_valid.columns:
            depth = row['DEPTH'] if not pd.isna(row['DEPTH']) else 0.0
            point_depth.extend([depth, depth])
        
        # X, Y, Z coordinates (from request)
        point_x.extend([row['X'], row['X']])
        point_y.extend([row['Y'], row['Y']])
        point_z.extend([row['Z'], row['Z']])
        
        # Date attribute (from request)
        if 'Date' in df_valid.columns and not pd.isna(row['Date']):
            try:
                event_date = pd.to_datetime(row['Date'])
                unix_timestamp = event_date.timestamp()
                point_date.extend([unix_timestamp, unix_timestamp])
            except:
                point_date.extend([0.0, 0.0])
        
        # Sequence/cluster information (from request)
        if 'sequence_label' in df_valid.columns:
            seq_label = str(row['sequence_label']) if not pd.isna(row['sequence_label']) else ''
            point_sequence_label.extend([seq_label, seq_label])
        
        if 'segmentation_level' in df_valid.columns:
            seg_level = row['segmentation_level'] if not pd.isna(row['segmentation_level']) else 0
            point_segmentation_level.extend([seg_level, seg_level])
        
        if 'fault_id' in df_valid.columns:
            fs_id = str(row['fault_id']) if not pd.isna(row['fault_id']) else ''
            point_fault_id.extend([fs_id, fs_id])
        
        # Magnitude (from request - using Mw if available, otherwise MAG)
        if 'Mw' in df_valid.columns:
            mw = row['Mw'] if not pd.isna(row['Mw']) else 0.0
            point_mw.extend([mw, mw])
        
        if 'MAG' in df_valid.columns:
            mag = row['MAG'] if not pd.isna(row['MAG']) else 0.0
            point_magnitude.extend([mag, mag])
        
        if 'final_cluster_id' in df_valid.columns:
            cluster = str(row['final_cluster_id'])
            point_cluster_id.extend([cluster, cluster])
    
    line_points = np.array(line_points, dtype=np.float64)
    line_connections = np.hstack(line_connections)
    
    # Create line mesh
    line_mesh = pv.PolyData(line_points, lines=line_connections)
    
    # Add point attributes (always present)
    line_mesh['rake_angle'] = np.array(point_rake, dtype=np.float32)  # Use renamed column name
    line_mesh['rupture_radius_m'] = np.array(point_rupture_radius, dtype=np.float32)
    
    # Add core location attributes
    if point_x:
        line_mesh['X'] = np.array(point_x, dtype=np.float32)
        line_mesh['Y'] = np.array(point_y, dtype=np.float32)
        line_mesh['Z'] = np.array(point_z, dtype=np.float32)
    
    if point_lat:
        line_mesh['LAT'] = np.array(point_lat, dtype=np.float32)
    
    if point_lon:
        line_mesh['LON'] = np.array(point_lon, dtype=np.float32)
    
    if point_depth:
        line_mesh['DEPTH'] = np.array(point_depth, dtype=np.float32)
    
    # Add temporal attribute
    if point_date:
        line_mesh['Date'] = np.array(point_date, dtype=np.float64)
    
    # Add sequence/cluster information
    if point_sequence_label:
        line_mesh['sequence_label'] = np.array(point_sequence_label)
    
    if point_segmentation_level:
        line_mesh['segmentation_level'] = np.array(point_segmentation_level, dtype=np.float32)
    
    if point_fault_id:
        line_mesh['fault_id'] = np.array(point_fault_id)
    
    # Add magnitude attributes
    if point_mw:
        line_mesh['Mw'] = np.array(point_mw, dtype=np.float32)
    
    if point_magnitude:
        line_mesh['magnitude'] = np.array(point_magnitude, dtype=np.float32)
        line_mesh['MAG'] = np.array(point_magnitude, dtype=np.float32)  # MAG alias
    
    if point_cluster_id:
        line_mesh['cluster_id'] = np.array(point_cluster_id)
    
    if point_event_id:
        line_mesh['event_id'] = np.array(point_event_id)
        line_mesh['ID'] = np.array(point_event_id)  # ID alias (from request)
    
    # Apply tube filter to make lines thicker (visible in 3D)
    # Tube radius based on typical rupture radius scale
    tube_radius = 2.0
    print(f'  Applying tube filter with radius: {tube_radius:.2f} m')
    
    tubes = line_mesh.tube(radius=tube_radius, n_sides=8)
    
    # Save the tube geometry
    slip_vectors_file = os.path.join(vtp_dir, 'slip_vectors.vtp')
    tubes.save(slip_vectors_file)
    
    print(f"  ✓ Saved: slip_vectors.vtp")
    print(f"    - {len(df_valid)} slip vector lines (bidirectional)")
    print(f"    - Tube radius: {tube_radius:.2f} m")
    print(f"    - Attributes: rake, rupture_radius_m, magnitude, cluster_id, event_id")
    print(f"")
    print(f"  Rake range: {df_valid['rake'].min():.1f}° to {df_valid['rake'].max():.1f}°")
    print(f"")
    print(f"  ParaView Usage:")
    print(f"  1. Open slip_vectors.vtp")
    print(f"  2. Color by 'rake' (Cool to Warm colormap, range: -180 to 180)")
    print(f"  3. Lines display as 3D tubes automatically")
    print(f"  4. Adjust opacity/representation as needed")


def model_3d_multi_sequence(input_params, enriched_catalog, combined_fault_planes):
    """
    Generate an interactive 3D model for multi-sequence results with cluster-colored hypocenters.

    Parameters
    ----------
    input_params : dict
        Input parameters including output directory and project title.
    enriched_catalog : DataFrame
        Full catalog with cluster labels and analysis results.
    combined_fault_planes : DataFrame
        Combined fault plane data from all clusters.

    Returns
    -------
    Interactive 3D model, saved in the output directory.
    """
    print('')
    print('Creating multi-sequence 3D visualization with cluster-colored hypocenters...')
    
    # Check if data is empty
    if len(enriched_catalog) == 0:
        print("Warning: No data to visualize - skipping visualization")
        return
    
    # Unpack input parameters
    out_dir = input_params.get('out_dir', './output')
    project_title = input_params.get('project_title', 'Multi-Sequence Analysis')
    
    fig = go.Figure()
    
    # Define a color palette for clusters
    cluster_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange  
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
        '#ff9999',  # light red
        '#66b3ff',  # light blue
        '#99ff99',  # light green
        '#ffcc99',  # light orange
        '#ff99cc',  # light pink
        '#c2c2f0',  # light purple
        '#ffb3e6',  # light magenta
        '#c4e17f',  # light lime
        '#76d7c4',  # light teal
        '#f7dc6f'   # light yellow
    ]
    
    # Get unique clusters (excluding unclustered events for now)
    clustered_data = enriched_catalog[enriched_catalog['sequence_label'] != 'unclustered'].copy()
    unique_clusters = sorted(clustered_data['sequence_label'].unique()) if len(clustered_data) > 0 else []
    
    # Plot clustered hypocenters by cluster
    for i, cluster_name in enumerate(unique_clusters):
        cluster_data = clustered_data[clustered_data['sequence_label'] == cluster_name]
        
        if len(cluster_data) == 0:
            continue
            
        # Use modulo to cycle through colors if we have more clusters than colors
        color = cluster_colors[i % len(cluster_colors)]
        
        # Convert dates for hover info if available
        if 'Date' in cluster_data.columns:
            cluster_data = cluster_data.copy()
            cluster_data['Date'] = pd.to_datetime(cluster_data['Date'])
        
        # Create safer hovertemplate
        hovertext = []
        for _, row in cluster_data.iterrows():
            text = f"<b>Event ID:</b> {row.get('ID', 'N/A')}<br>"
            text += f"<b>Cluster:</b> {row.get('sequence_label', 'N/A')}<br>"
            text += f"<b>Segmentation Level:</b> {row.get('segmentation_level', 'N/A')}<br>"
            text += f"<b>Magnitude:</b> {row.get('MAG', 'N/A')}<br>"
            text += f"<b>Depth:</b> {row.get('DEPTH', 'N/A')} m<br>"
            text += f"<b>Analysis Status:</b> {row.get('analysis_status', 'N/A')}<br>"
            hovertext.append(text)
        
        trace = go.Scatter3d(
            x=cluster_data['X'],
            y=cluster_data['Y'], 
            z=cluster_data['Z'],
            mode='markers',
            name=f'Cluster {cluster_name}',
            marker=dict(
                color=color,
                opacity=0.7,
                size=4,
                line=dict(width=0.5, color='black')
            ),
            text=hovertext,
            hovertemplate='%{text}<extra></extra>'
        )
        fig.add_trace(trace)
    
    # Plot unclustered events (noise) in gray
    unclustered_data = enriched_catalog[enriched_catalog['sequence_label'] == 'unclustered']
    if len(unclustered_data) > 0:
        # Create safer hovertemplate for unclustered events
        hovertext_unclustered = []
        for _, row in unclustered_data.iterrows():
            text = f"<b>Event ID:</b> {row.get('ID', 'N/A')}<br>"
            text += f"<b>Status:</b> Unclustered<br>"
            text += f"<b>Magnitude:</b> {row.get('MAG', 'N/A')}<br>"
            text += f"<b>Depth:</b> {row.get('DEPTH', 'N/A')} m<br>"
            hovertext_unclustered.append(text)
        
        trace_unclustered = go.Scatter3d(
            x=unclustered_data['X'],
            y=unclustered_data['Y'],
            z=unclustered_data['Z'],
            mode='markers',
            name='Unclustered',
            marker=dict(
                color='lightgray',
                opacity=0.3,
                size=2,
                line=dict(width=0, color='gray')
            ),
            text=hovertext_unclustered,
            hovertemplate='%{text}<extra></extra>'
        )
        fig.add_trace(trace_unclustered)
    
    # Add fault planes if available
    if combined_fault_planes is not None and len(combined_fault_planes) > 0:
        # Group fault planes by source cluster
        for cluster_name in combined_fault_planes['source_cluster'].unique():
            cluster_faults = combined_fault_planes[combined_fault_planes['source_cluster'] == cluster_name]
            
            # Get cluster color (same as used for hypocenters)
            cluster_idx = unique_clusters.index(cluster_name) if cluster_name in unique_clusters else 0
            fault_color = cluster_colors[cluster_idx % len(cluster_colors)]
            
            # Create fault plane traces
            for _, fault in cluster_faults.iterrows():
                # Get fault plane orientation
                azi = fault.get('rupt_plane_azi', np.nan)
                dip = fault.get('rupt_plane_dip', np.nan)
                
                if pd.isna(azi) or pd.isna(dip):
                    continue
                
                # Get event location (this fault belongs to this event)
                event_data = enriched_catalog[enriched_catalog['ID'] == fault['ID']]
                if len(event_data) == 0:
                    continue
                    
                event = event_data.iloc[0]
                center_x, center_y, center_z = event['X'], event['Y'], event['Z']
                
                # Create fault plane mesh
                plane_size = 200  # meters
                strike = (azi - 90) % 360  # Convert dip azimuth back to strike (strike = dip_azi - 90)
                
                # Calculate fault plane corners
                corners = utilities.calculate_plane_corners(
                    center_x, center_y, center_z, 
                    strike, dip, plane_size
                )
                
                if corners is not None:
                    # Create mesh for the fault plane
                    x_plane = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]]
                    y_plane = [corners[0][1], corners[1][1], corners[2][1], corners[3][1]]
                    z_plane = [corners[0][2], corners[1][2], corners[2][2], corners[3][2]]
                    
                    trace_plane = go.Mesh3d(
                        x=x_plane,
                        y=y_plane,
                        z=z_plane,
                        i=[0, 0],
                        j=[1, 2], 
                        k=[2, 3],
                        color=fault_color,
                        opacity=0.3,
                        name=f'Fault {cluster_name}',
                        showlegend=False,
                        hovertemplate=f'<b>Fault Plane</b><br>Cluster: {cluster_name}<br>Strike: {strike:.1f}°<br>Dip: {dip:.1f}°<extra></extra>'
                    )
                    fig.add_trace(trace_plane)
    
    # Calculate axis ranges for better visualization
    all_x = enriched_catalog['X'].dropna()
    all_y = enriched_catalog['Y'].dropna() 
    all_z = enriched_catalog['Z'].dropna()
    
    if len(all_x) > 0:
        x_margin = (all_x.max() - all_x.min()) * 0.1
        y_margin = (all_y.max() - all_y.min()) * 0.1
        z_margin = (all_z.max() - all_z.min()) * 0.1
        
        x_range = [all_x.min() - x_margin, all_x.max() + x_margin]
        y_range = [all_y.min() - y_margin, all_y.max() + y_margin]
        z_range = [all_z.min() - z_margin, all_z.max() + z_margin]
    else:
        x_range = [0, 1000]
        y_range = [0, 1000]
        z_range = [-1000, 0]
    
    # Cameraview standard (Top view) - match original model_3d
    eye = dict(x=0, y=-0.1, z=2)    
    
    # Define the figure layout parameters - match original model_3d exactly
    fig.update_layout(
        template='plotly_white',
        title=f'Hypocenter-Based Imaging of Active Faults (Truttmann et al. 2023): {project_title} - Cluster-Colored Hypocenters',
        hovermode=None,
        showlegend=True,
        legend={'itemclick': 'toggle'},
        scene=dict(
            xaxis_title='Easting [m]',
            yaxis_title='Northing [m]',
            zaxis_title='Depth [m]',
            xaxis=dict(
                range=x_range,
                tickformat='d',
                separatethousands=True,
                showspikes=False,
                showgrid=True,
                zeroline=True
                ),
            yaxis=dict(
                range=y_range,
                tickformat='d',
                separatethousands=True,
                showspikes=False,
                showgrid=True,
                zeroline=True
                ),
            zaxis=dict(
                range=z_range,
                tickformat='d',
                separatethousands=True,
                showspikes=False,
                showgrid=True,
                zeroline=True
                ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=eye)
            ),
        margin=dict(
            l=0,
            r=20,
            b=20,
            t=40)
        )
    
    fig.update_xaxes(title_standoff=20)
    
    # Save output
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, 'clustered_hypocenters.html')
    fig.write_html(output_file)
    
    print(f"3D model with clustered hypocenters saved to: {output_file}")
    print(f"Visualized {len(unique_clusters)} clusters with {len(clustered_data)} clustered events")
    print(f"Plus {len(unclustered_data)} unclustered events")


def model_3d(input_params, data_input, data_output):
    """
    Generate an interactive 3D model with plotly using single dataframe approach.

    Parameters
    ----------
    input_params : dict
        Input parameters.
    data_input : DataFrame
        Input data (legacy parameter, not used with single dataframe approach).
    data_output : DataFrame
        Output data (legacy parameter, not used with single dataframe approach).

    Returns
    -------
    Interactive 3D model, saved in the output directory.

    """

    print('\n')
    print('='*50)
    print('VISUALIZATION')
    print('='*50)

    
    # Check if data is empty
    if len(data_input) == 0 and len(data_output) == 0:
        print("Warning: No data to visualize - skipping visualization")
        return
    
    # Unpack input parameters from dictionary
    for key, value in input_params.items():
        globals()[key] = value

    df = pd.merge(data_input, data_output, on='ID').reset_index(drop=True)

    fig = go.Figure()

    ############################################################################
    # Plot hypocenters
    df['Date'] = pd.to_datetime(df['Date'])
    min_date = df['Date'].min()
    color_date = df['Date'].apply(lambda x: (x - min_date).days)
    tick_interval = 365
    max_days = color_date.max()
    colticks = np.arange(0, (int(max_days / 1) + 1) * 1, tick_interval)
    coldatetimes = [min_date + datetime.timedelta(days=i)
                    for i in colticks.tolist()]
    coltext = [i.strftime("%d-%b-%Y") for i in coldatetimes]
    trace = go.Scatter3d(
        x=df['X'],
        y=df['Y'],
        z=df['Z'],
        mode='markers',
        marker=dict(
            color=color_date,
            colorscale='Rainbow',
            opacity=0.5,
            colorbar=dict(
                title='Date',
                tickvals=colticks,
                ticktext=coltext,
                xanchor='left',
                x=0
                ),
            size=3,
            showscale=True),
        customdata=data_output,
        hovertemplate=
            '<b>Event ID:</b> %{customdata[0]} <br>'
            '<b>Class:</b> %{customdata[20]} <br>'
            '<br>'
            '<b>Fault parameters:</b> <br>'
            '<b>Rupture radius (m):</b> %{customdata[2]:.0f} <br>'
            '<b>Fault Plane Orientation:</b> %{customdata[12]} / %{customdata[13]} <br>'
            # '<b>κ:</b> %{customdata[10]:.0f} <br>'
            '<b>kappa:</b> %{customdata[10]:.0f} <br>'
            # '<b>β:</b> %{customdata[11]:.0f} <br>'
            '<b>beta:</b> %{customdata[11]:.0f} <br>'
            '<br>'
            '<b>Stress parameters:</b> <br>'
            '<b>Rake:</b> %{customdata[17]:.0f} <br>',
        legendgroup='hypocenter',
        name='Relocated hypocenters',
        showlegend=True,
        visible='legendonly',
        # visible=True,
        )
    fig.add_traces(trace)
    
    # Plot outliers (events with clust_labels = -1)
    # Check if there are outliers in data_input
    if 'clust_labels' in data_input.columns:
        outliers = data_input[data_input['clust_labels'] == -1]
        if not outliers.empty:
            trace = go.Scatter3d(
                x=outliers['X'],
                y=outliers['Y'],
                z=outliers['Z'],
                mode='markers',
                marker=dict(
                    # color=color_date,
                    color='black',
                    opacity=0.5,
                    colorscale='Rainbow',
                    size=3,
                    showscale=True),
                legendgroup='hypocenter_outliers',
                name='Relocated hypocenters (outliers)',
                showlegend=True,
                visible='legendonly',
                )
            fig.add_traces(trace)

    # # Plot hypocenter clusters
    # if 'clust_labels' in df.columns:
    #     column = np.array(df['clust_labels'])
    #     cmap = 'turbo'      # only these work: turbo, terrain
    #     minval = np.nanmin(column)
    #     maxval = np.nanmax(column) + 1
    #     colorsteps = len(df['clust_labels'].unique())
    #     colors = utilities_plot.colorscale(column, cmap, minval, maxval, colorsteps, cmap_reverse=False)

    #     trace = go.Scatter3d(
    #         x=df['X'],
    #         y=df['Y'],
    #         z=df['Z'],
    #         mode='markers',
    #         marker=dict(
    #             color=colors,
    #             colorbar=dict(
    #                 title='Cluster',
    #                 tickvals=colticks,
    #                 ticktext=coltext,
    #                 xanchor='left',
    #                 x=0
    #                 ),
    #             size=3,
    #             showscale=True),
    #         customdata=data_output,
    #         hovertemplate=
    #             '<b>Event ID:</b> %{customdata[0]} <br>'
    #             '<b>Cluster Nr.:</b> %{customdata[3]} <br>',
    #         legendgroup='hypocenter_clustered',
    #         name='Relocated hypocenters (clustered)',
    #         showlegend=True,
    #         visible=True,
    #         )
    #     fig.add_traces(trace)


    ############################################################################
    # Plot error ellipsoids
    # Workaround to only show one legend entry for fault planes: create an array
    # with only the first value True with the length of the number of events
    ex_dropna = df['EX'].dropna()
    if len(ex_dropna) > 0:
        idx = ex_dropna.index[0]
        legend_show = [False for i in range(len(df))]
        legend_show[idx] = True
    else:
        legend_show = [False for i in range(len(df))]

    for i in range(len(df)):
        # Create error ellipse at the zero point
        phi = np.linspace(0, 2 * np.pi, 10)
        theta = np.linspace(-np.pi / 2, np.pi / 2, 10)
        phi, theta = np.meshgrid(phi, theta)
        x = np.cos(theta) * np.sin(phi) * df['EX'][i] * 3
        y = np.cos(theta) * np.cos(phi) * df['EY'][i] * 3
        z = np.sin(theta) * df['EZ'][i] * 3
    
        # Shift error ellipse to the right xyz coordinates
        x = x + df['X'][i]
        y = y + df['Y'][i]
        z = z + df['Z'][i]
    
        trace = go.Mesh3d(x=x.flatten(),
                          y=y.flatten(),
                          z=z.flatten(),
                          color='grey',
                          opacity=0.2,
                          alphahull=0,
                          hoverinfo='none',
                          showlegend=legend_show[i],
                          name='Error ellipsoids (3σ)',
                          legendgroup='Error ellipsoids (3σ)',
                          visible='legendonly')
        fig.add_trace(trace)
    
    # Plot error ellipsoids of outliers (events with clust_labels = -1)
    # Check if there are outliers in data_input
    if 'clust_labels' in data_input.columns:
        outliers = data_input[data_input['clust_labels'] == -1]
        if not outliers.empty:
            ex_outliers_dropna = outliers['EX'].dropna()
            if len(ex_outliers_dropna) > 0:
                idx = ex_outliers_dropna.index[0]
                legend_show = [False for i in range(len(outliers))]
                legend_show[idx - outliers.index[0]] = True  # Adjust index for subset
            else:
                legend_show = [False for i in range(len(outliers))]

            for i, (row_idx, row) in enumerate(outliers.iterrows()):
                # Create error ellipse at the zero point
                phi = np.linspace(0, 2 * np.pi, 10)
                theta = np.linspace(-np.pi / 2, np.pi / 2, 10)
                phi, theta = np.meshgrid(phi, theta)
                x = np.cos(theta) * np.sin(phi) * row['EX'] * 3
                y = np.cos(theta) * np.cos(phi) * row['EY'] * 3
                z = np.sin(theta) * row['EZ'] * 3
            
                # Shift error ellipse to the right xyz coordinates
                x = x + row['X']
                y = y + row['Y']
                z = z + row['Z']
            
                trace = go.Mesh3d(x=x.flatten(),
                                y=y.flatten(),
                                z=z.flatten(),
                                color='grey',
                                opacity=0.2,
                                alphahull=0,
                                hoverinfo='none',
                                showlegend=legend_show[i],
                                name='Error ellipsoids (outliers) (3σ)',
                                legendgroup='Error ellipsoids (outliers) (3σ)',
                                visible='legendonly')
                fig.add_trace(trace)

    ############################################################################
    # Plot the calculated active planes
    if 'Strike1' in df.columns:
        # Colors for focal planes
        colormap = 'RdYlGn_r'
        if 'epsilon' in df.columns:
            column = np.array(df['epsilon'])
            if np.all(np.isnan(column)):
                colors = ['gray'] * len(df)
            else:
                minval = 0
                # maxval = np.nanmax(column)
                # maxval = math.ceil(maxval * 10) / 10.0
                maxval = 45
                colorsteps = 40
                colors = utilities_plot.colorscale(column, colormap, minval, maxval, colorsteps)
        else:
            colors = ['gray'] * len(df)
        
        # Workaround to only show one legend entry for fault planes: create an array
        # with only the first value True with the length of the number of events
        strike1_dropna = df['Strike1'].dropna()
        if len(strike1_dropna) > 0:
            idx = strike1_dropna.index[0]
            legend_show = [False for i in range(len(df))]
            legend_show[idx] = True
        else:
            legend_show = [False for i in range(len(df))]
    
        show_pref_unpref_legend = True
        for i in range(len(df)):
            if pd.isnull(df['Strike1'][i]) is True:
                pass
            else:
                # Select the focal plane with the smaller angular difference to the
                # reconstructed fault plane
                if df['pref_foc'][i] == 1:
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df['Strike1'][i]
                                                            + 90 % 360,
                                                            df['Dip1'][i])
                    nor_pref = np.array([nor_x, nor_y, nor_z])
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df['Strike2'][i]
                                                            + 90 % 360,
                                                            df['Dip2'][i])
                    nor_nonpref = np.array([nor_x, nor_y, nor_z])
                    foc_color = 'black'
        
                elif df['pref_foc'][i] == 2:
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df['Strike2'][i]
                                                            + 90 % 360,
                                                            df['Dip2'][i])
                    nor_pref = np.array([nor_x, nor_y, nor_z])
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df['Strike1'][i]
                                                            + 90 % 360,
                                                            df['Dip1'][i])
                    nor_nonpref = np.array([nor_x, nor_y, nor_z])
                    foc_color = 'black'
                else:
                    continue                    
                    
                # Get XYZ coordinates of the points of the circular fault plane
                x = df['X'][i]
                y = df['Y'][i]
                z = df['Z'][i]
                p = [x, y, z]
                r = df['rupt_radius'][i]
                X_pref, Y_pref, Z_pref = utilities_plot.circleplane(p, r, nor_pref)
                X_nonpref, Y_nonpref, Z_nonpref = utilities_plot.circleplane(p, r, nor_nonpref)
        
                # Preferred focal plane
                focals_pref = go.Scatter3d(
                    x=X_pref,
                    y=Y_pref,
                    z=Z_pref,
                    mode='lines',
                    line=dict(
                        color=colors[i],
                        # color=foc_color,
                        width=10),
                    hoverinfo='none',
                    legendgroup='FM: calc. act. plane',
                    name='FM: calc. act. plane',
                    showlegend=show_pref_unpref_legend,
                    visible='legendonly'
                    )
                
                # Rake of preferred focal
                r = df['rupt_radius'][i]
                if pd.isnull(df['pref_foc'][i]):
                    # No preferred focal plane data - use default (plane 1)
                    rake = df['Rake1'][i]
                    rake_color = 'gray'
                elif df['pref_foc'][i] == 1:
                    rake = df['Rake1'][i]
                    rake_color = 'black'
                elif df['pref_foc'][i] == 2:
                    rake = df['Rake2'][i]
                    rake_color = 'black'
                else:
                    rake = df['Rake1'][i]
                    rake_color = 'lightgrey'
                u, v, w = utilities_plot.slipvector_3D(p, r, nor_pref, rake)
                xx = [x + (x - u), u]
                yy = [y + (y - v), v]
                zz = [z + (z - w), w]
                trace = go.Scatter3d(x=xx, y=yy, z=zz,
                                      mode='lines',
                                      line=dict(
                                          color=rake_color,
                                          width=10),
                                      legendgroup='FM: calc. act. plane (Slip vectors)',
                                      hoverinfo='none',
                                      name='FM: calc. act. plane (Slip vectors)',
                                      showlegend=show_pref_unpref_legend,
                                      visible='legendonly'
                                      )
                fig.add_trace(trace)

                # Non-preferred focal plane
                focals_nonpref = go.Scatter3d(
                    x=X_nonpref,
                    y=Y_nonpref,
                    z=Z_nonpref,
                    mode='lines',
                    line=dict(
                        color=foc_color,
                        width=5),
                    hoverinfo='none',
                    legendgroup='FM: calc. non-act. plane',
                    name='FM: calc. non-act. plane',
                    showlegend=show_pref_unpref_legend,
                    visible='legendonly'
                    )
                fig.add_traces([focals_pref, focals_nonpref])

                show_pref_unpref_legend = False



    ############################################################################
    # Plot the known active planes
    if 'Strike1' in df.columns and 'A' in df.columns:
    
        do_known = False
        show_known_legend = True
        show_unk_legend = True
        for i in range(len(df)):
            if pd.isnull(df['A'][i]) is True:
                pass
            else:
                if df['A'][i] == 1:
                    known_strike = 'Strike1'
                    known_dip = 'Dip1'
                    foc_color = 'black'
                    do_known = True
        
                elif df['A'][i] == 2:
                    known_strike = 'Strike2'
                    known_dip = 'Dip2'
                    foc_color = 'black'
                    do_known = True

                if do_known:
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df[known_strike][i]
                                                            + 90 % 360,
                                                            df[known_dip][i])
                    nor_known = np.array([nor_x, nor_y, nor_z])
                    foc_color = 'black'                    
                        
                    # Get XYZ coordinates of the points of the circular fault plane
                    x = df['X'][i]
                    y = df['Y'][i]
                    z = df['Z'][i]
                    p = [x, y, z]
                    r = df['rupt_radius'][i]
                    X_known, Y_known, Z_known = utilities_plot.circleplane(p, r, nor_known)
            
                    # Known focal plane
                    focals_known = go.Scatter3d(
                        x=X_known,
                        y=Y_known,
                        z=Z_known,
                        mode='lines',
                        line=dict(
                            color=foc_color,
                            width=10),
                        hoverinfo='none',
                        legendgroup='FM: known act. plane',
                        name='FM: known act. plane',
                        showlegend=show_known_legend,
                        visible='legendonly'
                        )
                    
                    fig.add_traces([focals_known])

                    # Rake of known focal
                    r = df['rupt_radius'][i]
                    if df['A'][i] == 1:
                        rake = df['Rake1'][i]
                        rake_color = 'black'
                    elif df['A'][i] == 2:
                        rake = df['Rake2'][i]
                        rake_color = 'black'
                    u, v, w = utilities_plot.slipvector_3D(p, r, nor_known, rake)
                    xx = [x + (x - u), u]
                    yy = [y + (y - v), v]
                    zz = [z + (z - w), w]
                    trace = go.Scatter3d(x=xx, y=yy, z=zz,
                                        mode='lines',
                                        line=dict(
                                            color=rake_color,
                                            width=30),
                                        legendgroup='FM: known act. plane (Slip vectors)',
                                        hoverinfo='none',
                                        name='FM: known act. plane (Slip vectors)',
                                        showlegend=show_known_legend,
                                        visible='legendonly'
                                        )
                    fig.add_trace(trace)

                    do_known = False
                    show_known_legend = False

                if df['A'][i] == 0:
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df['Strike1'][i]
                                                            + 90 % 360,
                                                            df['Dip1'][i])
                    nor_unk1 = np.array([nor_x, nor_y, nor_z])
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df['Strike2'][i]
                                                            + 90 % 360,
                                                            df['Dip2'][i])
                    nor_unk2 = np.array([nor_x, nor_y, nor_z])
                        
                    # Get XYZ coordinates of the points of the circular fault plane
                    x = df['X'][i]
                    y = df['Y'][i]
                    z = df['Z'][i]
                    p = [x, y, z]
                    r = df['rupt_radius'][i]
                    X_unk1, Y_unk1, Z_unk1 = utilities_plot.circleplane(p, r, nor_unk1)
                    X_unk2, Y_unk2, Z_unk2 = utilities_plot.circleplane(p, r, nor_unk2)
            
                    foc_color = 'grey'
                    # Unknown focal plane 1
                    focals_unk1 = go.Scatter3d(
                        x=X_unk1,
                        y=Y_unk1,
                        z=Z_unk1,
                        mode='lines',
                        line=dict(
                            color=foc_color,
                            dash='dashdot',
                            width=10),
                        hoverinfo='none',
                        legendgroup='FM: unk. act. plane (1)',
                        name='FM: unk. act. plane (1)',
                        showlegend=show_unk_legend,
                        visible='legendonly'
                        )
                    
                    # Unknown focal plane 2
                    focals_unk2 = go.Scatter3d(
                        x=X_unk2,
                        y=Y_unk2,
                        z=Z_unk2,
                        mode='lines',
                        line=dict(
                            color=foc_color,
                            dash='dot',
                            width=10),
                        hoverinfo='none',
                        legendgroup='FM: unk. act. plane (2)',
                        name='FM: unk. act. plane (2)',
                        showlegend=show_unk_legend,
                        visible='legendonly'
                        )
                    fig.add_traces([focals_unk1, focals_unk2])

                    show_unk_legend = False


    ############################################################################
    # Plot the fault planes
    # Prioritize final_cluster_id (orientation + spatial) over orient_cluster (orientation only)
    if 'final_cluster_id' in df.columns and not data_output['final_cluster_id'].isna().all():
        # Use combined orientation + spatial clustering
        column = data_output['final_cluster_id']
        
        # Convert string IDs to numeric for coloring (handle FS0001, F1_0, etc.)
        numeric_ids = []
        unique_ids = {}
        counter = 0
        for val in column:
            if pd.isna(val):
                numeric_ids.append(np.nan)
            else:
                str_val = str(val)
                if str_val not in unique_ids:
                    unique_ids[str_val] = counter
                    counter += 1
                numeric_ids.append(unique_ids[str_val])
        
        column = pd.Series(numeric_ids)
        cmap = 'gnuplot'
        minval = np.nanmin(column) - 1.1
        maxval = np.nanmax(column) + 0.1
        colorsteps = len(unique_ids)
        colors = utilities_plot.colorscale(column, cmap, minval, maxval, colorsteps, cmap_reverse=False)
    elif 'orient_cluster' in df.columns and not data_output['orient_cluster'].isna().all():
        # Fallback to orientation-only clustering
        column = data_output['orient_cluster']
        cmap = 'gnuplot'
        minval = np.nanmin(data_output['orient_cluster']) - 1.1
        maxval = np.nanmax(data_output['orient_cluster']) + 0.1
        colorsteps = len(data_output['orient_cluster'].unique())
        colors = utilities_plot.colorscale(column, cmap, minval, maxval, colorsteps, cmap_reverse=False)
    else:
        colors = ['black'] * len(df)

    column = df['kappa']
    minval = 0
    # maxval = np.nanmax(column)/2
    maxval = 100000
    opac = utilities_plot.opacity(column, minval, maxval, 1000)

    # Workaround to only show one legend entry for fault planes: create an array
    # with only the first value True with the length of the number of events
    legend_show = [False for i in range(len(df))]
    idx = df['rupt_plane_azi'].dropna()
    try:
        idx = idx.index[0]   
        legend_show[idx] = True
    except IndexError:
        print("indexerror")
        pass
    
    for i in range(len(df)):
        # Get XYZ coordinates of the points of the circular fault plane around the
        # hypocenter (point p)
        p = [df['X'][i],
              df['Y'][i],
              df['Z'][i]
              ]
        r = df['rupt_radius'][i]
        nor = np.array([df['nor_x_mean'][i],
                        df['nor_y_mean'][i],
                        df['nor_z_mean'][i]])
        X, Y, Z = utilities_plot.circleplane(p, r, nor)

        faults = go.Scatter3d(
            x=X,
            y=Y,
            z=Z,
            mode='lines',
            line=dict(
                color=colors[i],
                width=6),
            opacity=0.8,
            hoverinfo='none',
            legendgroup='3D Fault Model',
            name='3D Fault Model',
            showlegend=legend_show[i],
            visible='legendonly',
            )
        fig.add_traces(faults)

    df_k = df[df['rupt_plane_dip'].isna()]
    df_k = df_k[df_k['clust_labels'] != -1]     # Remove outliers from df_k (-1 in clust_labels)
    df_k = df_k.reset_index(drop=True)
    trace = go.Scatter3d(
        x=df_k['X'],
        y=df_k['Y'],
        z=df_k['Z'],
        mode='markers',
        marker=dict(
            color='rgba(0, 0, 0, 0.2)',
            size=2,
            showscale=False),
        legendgroup='3D Fault Model',
        showlegend=False,
        visible='legendonly',
        )
    fig.add_traces(trace)

    ############################################################################
    # Plot stress states
    if 'instab' in df.columns:
        # Plot the fault planes with fault instability
        colormap = 'plasma'
        column = np.array(df['instab'])
        minval = 0
        # maxval = np.nanmax(column)
        maxval = 1
        colorsteps = 50
        colors = utilities_plot.colorscale(column, colormap, minval, maxval, colorsteps)
        
        for i in range(len(df)):
            # Get XYZ coordinates of the points of the circular fault plane around the
            # hypocenter (point p)
            p = [df['X'][i],
                  df['Y'][i],
                  df['Z'][i]
                  ]
            r = df['rupt_radius'][i]
            nor = np.array([df['nor_x_mean'][i],
                            df['nor_y_mean'][i],
                            df['nor_z_mean'][i]])
            X, Y, Z = utilities_plot.circleplane(p, r, nor)
        
            faults = go.Scatter3d(
                x=X,
                y=Y,
                z=Z,
                mode='lines',
                line=dict(
                    color=colors[i],
                    width=6),
                hoverinfo='none',
                legendgroup='Fault instability I',
                name='Fault instability I',
                showlegend=legend_show[i],
                visible='legendonly'
                )
            fig.add_traces(faults)
            
        df_k = df[df['rupt_plane_dip'].isna()]
        df_k = df_k.reset_index(drop=True)
        trace = go.Scatter3d(
            x=df_k['X'],
            y=df_k['Y'],
            z=df_k['Z'],
            mode='markers',
            marker=dict(
                color='rgba(0, 0, 0, 0.2)',
                size=2,
                showscale=False),
            legendgroup='Fault instability I',
            name='Fault instability I',
            hoverinfo='none',
            showlegend=False,
            visible='legendonly',
            )
        fig.add_traces(trace)

        
        # Plot the slip vectors
        colormap = 'twilight_shifted'
        column = np.array(df['rake'])
        minval = -180
        maxval = 180
        colorsteps = 360
        colors = utilities_plot.colorscale(column, colormap, minval, maxval, colorsteps)
        
        for i in range(len(df)):
            if df['rake'][i] == np.nan:
                pass
            else:
                x = df['X'][i]
                y = df['Y'][i]
                z = df['Z'][i]
                p = [x, y, z]
                r = df['rupt_radius'][i]
                nor = np.array([df['nor_x_mean'][i],
                                df['nor_y_mean'][i],
                                df['nor_z_mean'][i]])
                if df['rake'][i] < 0:
                    rake = df['rake'][i]
                else:
                    rake = df['rake'][i]
                u, v, w = utilities_plot.slipvector_3D(p, r, nor, rake)
                xx = [x + (x - u), u]
                yy = [y + (y - v), v]
                zz = [z + (z - w), w]
                trace = go.Scatter3d(x=xx, y=yy, z=zz,
                                      mode='lines',
                                      line=dict(
                                          color=colors[i],
                                          width=6),
                                      legendgroup='Slip vectors',
                                      name='Slip vectors',
                                      hoverinfo='none',
                                      showlegend=legend_show[i],
                                      visible='legendonly'
                                      )
                fig.add_trace(trace)
        
        df_k = df[df['rupt_plane_dip'].isna()]
        df_k = df_k.reset_index(drop=True)
        trace = go.Scatter3d(
            x=df_k['X'],
            y=df_k['Y'],
            z=df_k['Z'],
            mode='markers',
            marker=dict(
                color='rgba(0, 0, 0, 0.2)',
                size=2,
                showscale=False),
            legendgroup='Slip vectors',
            name='Slip vectors',
            hoverinfo='none',
            showlegend=False,
            visible='legendonly',
            )
        fig.add_traces(trace)
        
    else:
        pass

    ############################################################################
    # Plot interpolated fault planes (if available)
    try:
        # Check if interpolated planes exist
        out_path = input_params.get('out_dir', './output')
        vtp_dir = os.path.join(out_path, 'vtp_export')
        
        if os.path.exists(vtp_dir):
            # Look for individual fault plane VTP files
            VTP_files = [f for f in os.listdir(vtp_dir) if f.startswith('fault_') and f.endswith('.vtp')]
            
            if VTP_files:
                print(f"Adding {len(VTP_files)} interpolated fault planes to 3D model...")
                
                # Color palette for different fault planes
                interp_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
                
                # Track legend entries to avoid duplicates
                legend_shown = {'interpolated_planes': False}
                
                for i, VTP_file in enumerate(VTP_files):
                    try:
                        VTP_path = os.path.join(vtp_dir, VTP_file)
                        
                        # Load VTP mesh
                        mesh = pv.read(VTP_path)
                        
                        if mesh.n_points > 0:
                            # Get vertices and faces
                            vertices = mesh.points
                            
                            # Extract triangle faces for plotly
                            if mesh.n_cells > 0:
                                # Get triangular faces
                                faces = []
                                for cell_id in range(mesh.n_cells):
                                    cell = mesh.get_cell(cell_id)
                                    if cell.n_points == 3:  # Triangle
                                        faces.append(cell.point_ids)
                                
                                if faces:
                                    faces = np.array(faces)
                                    
                                    # Create mesh3d trace
                                    color = interp_colors[i % len(interp_colors)]
                                    
                                    # Extract orientation info from filename
                                    file_info = VTP_file.replace('fault_plane_', '').replace('.vtp', '')
                                    
                                    trace = go.Mesh3d(
                                        x=vertices[:, 0],
                                        y=vertices[:, 1], 
                                        z=vertices[:, 2],
                                        i=faces[:, 0],
                                        j=faces[:, 1],
                                        k=faces[:, 2],
                                        color=color,
                                        opacity=0.6,
                                        name='Interpolated Fault Planes',
                                        legendgroup='interpolated_planes',
                                        showlegend=not legend_shown['interpolated_planes'],
                                        hovertemplate=f'<b>Interpolated Fault Plane</b><br>{file_info}<br><extra></extra>',
                                        visible='legendonly'
                                    )
                                    
                                    fig.add_trace(trace)
                                    legend_shown['interpolated_planes'] = True
                                    
                    except Exception as e:
                        print(f"Warning: Could not load interpolated plane {VTP_file}: {e}")
                        continue
                
                print("✓ Interpolated fault planes added to 3D model")
            else:
                print("No individual fault plane VTP files found")
        else:
            print("No interpolated_planes directory found")
            
    except Exception as e:
        print(f"Warning: Could not add interpolated planes to 3D model: {e}")

    ############################################################################
    # Layout settings
    # Calculate the "empty white box" to ensure equal axes of the 3D plot
    # Work-around to ensure equal axes of the 3D plot using all data_input
    x_range, y_range, z_range = utilities_plot.equal_axes(df['X'],
                                            df['Y'],
                                            df['Z'])

    # Cameraview standard (Top view)
    eye = dict(x=0, y=-0.1, z=2)    
    
    # Define the figure layout parameters
    fig.update_layout(
        template='plotly_white',
        title=f'Hypocenter-Based Imaging of Active Faults (Truttmann et al. 2023): {input_params.get("project_title", "Fault Imaging Analysis")}',
        hovermode=None,
        showlegend=True,
        legend={'itemclick': 'toggle'},
        scene=dict(
            xaxis_title='Easting [m]',
            yaxis_title='Northing [m]',
            zaxis_title='Depth [m]',
            xaxis=dict(
                range=x_range,
                tickformat='d',
                separatethousands=True,
                showspikes=False,
                showgrid=True,
                zeroline=True
                ),
            yaxis=dict(
                range=y_range,
                tickformat='d',
                separatethousands=True,
                showspikes=False,
                showgrid=True,
                zeroline=True
                ),
            zaxis=dict(
                range=z_range,
                tickformat='d',
                separatethousands=True,
                showspikes=False,
                showgrid=True,
                zeroline=True
                ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=eye)
            ),
        margin=dict(
            l=0,
            r=20,
            b=20,
            t=40)
        )
    
    fig.update_xaxes(title_standoff=20)
    
    # Save output
    out_path = input_params['out_dir']
    os.makedirs(out_path, exist_ok=True)

    fig.write_html(out_path + '/3D_model.html')

    return


def faults_stereoplot(input_params, data_output):
    # Unpack input parameters from dictionary
    for key, value in input_params.items():
        globals()[key] = value
    
    # remove rows with nan values in mean_azi and mean dip
    df_output = data_output.dropna(subset=['rupt_plane_azi', 'rupt_plane_dip']).reset_index(drop=True)
    
    # Check if df_output is empty after dropping NaN values
    if len(df_output) == 0:
        print("Warning: No valid data for stereoplot after removing NaN values")
        return
    
    if 'orient_cluster' in df_output.columns:
        column = df_output['orient_cluster'].to_numpy()
        cmap = 'gnuplot'
        minval = np.nanmin(column) - 0.1
        maxval = np.nanmax(column) + 0.1
        colorsteps = len(df_output['orient_cluster'].unique())
        colors = utilities_plot.colorscale_mplstereonet(column, cmap, minval, maxval, colorsteps, cmap_reverse=False)
    # if 'clust_labels' in df_output.columns:
    #     column = df_output['clust_labels'].to_numpy()
    #     cmap = 'turbo'
    #     minval = np.nanmin(column)
    #     maxval = np.nanmax(column) + 1
    #     colorsteps = len(df_output['clust_labels'].unique())
    #     colors = utilities_plot.colorscale_mplstereonet(column, cmap, minval, maxval, colorsteps, cmap_reverse=False)
        
    else:
        colors = ['black'] * len(df_output)

    column = df_output['kappa'].to_numpy()
    minval = 0
    if len(column) > 0:
        maxval = np.nanmax(column)
    else:
        maxval = 1
    opacity = utilities_plot.opacity(column, minval, maxval, 20)
        
    mm = 1/25.4
    fig, ax = mplstereonet.subplots(figsize=(100*mm, 100*mm))
    
    # Plot poles
    strikes = df_output['rupt_plane_azi'].to_numpy() - 90 % 360
    dips = df_output['rupt_plane_dip'].to_numpy()
    for i in range(len(df_output)):
        ax.pole(strikes[i], dips[i],
                marker='o',
                c=colors[i],
                markersize=2,
                markeredgecolor='black',
                markeredgewidth=0.1,
                alpha=opacity[i])
        
        annotate = False
        if annotate:
            # annotate i in polar coordinates
            idx = int(df_output['clust_labels'][i])
            lon, lat = mplstereonet.pole(strikes[i], dips[i])
            ax.text(lon, lat, f'{idx}', fontsize=4, color='black',
                )

        
    # # annotate points with right color
    # for i in range(len(df_output)):
    #     ax.annotate(i, xy=(df_output['rupt_plane_azi'][i] - 90 % 360, df_output['rupt_plane_dip'][i]), color=colors[i], fontsize=5)
    
    # # add legend of clusters
    # if 'clust_labels' in df_output.columns:
    #     for i in range(len(df_output['clust_labels'].unique())):
    #         ax.pole(0, 0, marker='o', c=colors[i], markersize=5, alpha=1, label=i)
    #     ax.legend(loc='upper right')

    # Plot density countours
    cax = ax.density_contourf((df_output['rupt_plane_azi'] - 90 % 360),
                        df_output['rupt_plane_dip'],
                        cmap='Greys',
                        alpha=0.9,
                        measurement='poles',
                        method='exponential_kamb',
                        sigma=5,
                        )

    # ax.set_azimuth_ticks(angles=[0, 180], labels=['North', 'South'])
    # fig.colorbar(cax, label='Density')
    ax.set_azimuth_ticks([])
    ax.grid()
    
    # Print nr of fault planes in lower right corner
    ax.text(0.98, 0.02, f'n = {len(df_output)}',
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes)

    
    # Save figure
    out_path = input_params['out_dir']
    os.makedirs(out_path, exist_ok=True)

    fig.savefig(out_path + '/Stereoplot.pdf', bbox_inches='tight')
    plt.close(fig)
    
    return
    

def nmc_histogram(input_params, data_input, per_X, per_Y, per_Z):
    """
    Parameters
    ----------
    per_X : TYPE
        DESCRIPTION.
    per_Y : TYPE
        DESCRIPTION.
    per_Z : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """   
    
    print('Plotting MC dataset histograms')
 
    mm = 1/25.4
    # Create output path (if not existing yet)
    out_path = input_params['out_dir']
    os.makedirs(out_path, exist_ok=True)
    newpath = out_path + '/ErrorDistributions'
    os.makedirs(newpath, exist_ok=True)
    
    # Precompute values outside the loop
    binwidth = 1
    linewidth1 = 1
    linewidth2 = 0.7
    plt.rcParams.update({'font.size': 8})


    for i in range(len(data_input)):
        fig, axs = plt.subplots(nrows=1, ncols=3,
                                figsize=(190*mm, 70*mm,), sharey=True,
                                tight_layout=True)

        axs[0].hist(per_X.loc[i, 1:], density=True,
                    bins=np.arange(min(per_X.loc[i, 1:]),
                               max(per_X.loc[i, 1:]) + binwidth, binwidth),
                    color='grey')
        axs[1].hist(per_Y.loc[i, 1:], density=True,
                    bins=np.arange(min(per_Y.loc[i, 1:]),
                               max(per_Y.loc[i, 1:]) + binwidth, binwidth),
                    color='grey')
        axs[2].hist(per_Z.loc[i, 1:], density=True,
                    bins=np.arange(min(per_Z.loc[i, 1:]),
                               max(per_Z.loc[i, 1:]) + binwidth, binwidth),
                    color='grey')
        
        y_axis_max = axs[0].get_ylim()[1]
        axs[0].vlines(data_input.loc[i, 'X'], 0, y_axis_max, color='black', linewidth=linewidth1)
        axs[1].vlines(data_input.loc[i, 'Y'], 0, y_axis_max, color='black', linewidth=linewidth1)
        axs[2].vlines(data_input.loc[i, 'Z'], 0, y_axis_max, color='black', linewidth=linewidth1)
        
        axs[0].vlines(data_input.loc[i, 'X'] + 0.5 * data_input.loc[i, 'EX'], 0, y_axis_max, color='black', linestyles='dashed', linewidth=linewidth2)
        axs[1].vlines(data_input.loc[i, 'Y'] + 0.5 * data_input.loc[i, 'EY'], 0, y_axis_max, color='black', linestyles='dashed', linewidth=linewidth2)
        axs[2].vlines(data_input.loc[i, 'Z'] + 0.5 * data_input.loc[i, 'EZ'], 0, y_axis_max, color='black', linestyles='dashed', linewidth=linewidth2)
        axs[0].vlines(data_input.loc[i, 'X'] - 0.5 * data_input.loc[i, 'EX'], 0, y_axis_max, color='black', linestyles='dashed', linewidth=linewidth2)
        axs[1].vlines(data_input.loc[i, 'Y'] - 0.5 * data_input.loc[i, 'EY'], 0, y_axis_max, color='black', linestyles='dashed', linewidth=linewidth2)
        axs[2].vlines(data_input.loc[i, 'Z'] - 0.5 * data_input.loc[i, 'EZ'], 0, y_axis_max, color='black', linestyles='dashed', linewidth=linewidth2)
        
        axs[0].set_xlabel('Easting X [m]')
        axs[1].set_xlabel('Northing Y [m]')
        axs[2].set_xlabel('Depth Z [m]')
        
        axs[0].set_ylabel('Probability density')
        
        # Set limits of histogram to ensure same scale of x-axis
        X_diff = abs(max(per_X.loc[i, 1:]) - min(per_X.loc[i, 1:]))
        Y_diff = abs(max(per_Y.loc[i, 1:]) - min(per_Y.loc[i, 1:]))
        Z_diff = abs(max(per_Z.loc[i, 1:]) - min(per_Z.loc[i, 1:]))
        max_range = 0.5 * max(X_diff, Y_diff, Z_diff)
        X_mean = data_input.loc[i, 'X']
        Y_mean = data_input.loc[i, 'Y']
        Z_mean = data_input.loc[i, 'Z']
        axs[0].set_xlim(X_mean - max_range, X_mean + max_range)
        axs[1].set_xlim(Y_mean - max_range, Y_mean + max_range)
        axs[2].set_xlim(Z_mean - max_range, Z_mean + max_range)
                
        ID = data_input.loc[i, 'ID']
                
        fig.savefig(newpath + f'/ErrorDist_{ID}.pdf')
        plt.close(fig)
        
    return


def model_3d_single_df(df_hyfi, input_params):
    """
    Generate an interactive 3D model with plotly using single dataframe approach.

    Parameters
    ----------
    df_hyfi : DataFrame
        Single dataframe containing all hypocenter and fault plane data.
    input_params : dict
        Input parameters.

    Returns
    -------
    Interactive 3D model, saved in the output directory.

    """

    print('\n')
    print('='*50)
    print('VISUALIZATION')
    print('='*50)

    
    # Check if data is empty
    if len(df_hyfi) == 0:
        print("Warning: No data to visualize - skipping visualization")
        return
    
    # Use df_hyfi directly
    df = df_hyfi.copy()

    fig = go.Figure()

    ############################################################################
    # Plot hypocenters
    df['Date'] = pd.to_datetime(df['Date'])
    min_date = df['Date'].min()
    color_date = df['Date'].apply(lambda x: (x - min_date).days)
    tick_interval = 365
    max_days = color_date.max()
    colticks = np.arange(0, (int(max_days / 1) + 1) * 1, tick_interval)
    coldatetimes = [min_date + datetime.timedelta(days=i)
                    for i in colticks.tolist()]
    coltext = [i.strftime("%d-%b-%Y") for i in coldatetimes]
    
    # Create custom data array for hover info using available columns
    customdata_list = []
    for _, row in df.iterrows():
        customdata_row = [
            row.get('ID', 'N/A'),               # 0
            row.get('Date', 'N/A'),             # 1  
            row.get('rupt_radius', 'N/A'),                # 2
            row.get('clust_labels', 'N/A'),     # 3
            '',                                 # 4
            '',                                 # 5
            '',                                 # 6
            '',                                 # 7
            '',                                 # 8
            '',                                 # 9
            row.get('kappa', 'N/A'),            # 10
            row.get('beta', 'N/A'),             # 11
            row.get('rupt_plane_azi', 'N/A'),         # 12
            row.get('rupt_plane_dip', 'N/A'),         # 13
            '',                                 # 14
            '',                                 # 15
            '',                                 # 16
            row.get('rake', 'N/A'),             # 17
            '',                                 # 18
            '',                                 # 19
            row.get('orient_cluster', 'N/A'),            # 20
        ]
        customdata_list.append(customdata_row)
    
    trace = go.Scatter3d(
        x=df['X'],
        y=df['Y'],
        z=df['Z'],
        mode='markers',
        marker=dict(
            color=color_date,
            colorscale='Rainbow',
            opacity=0.5,
            colorbar=dict(
                title='Date',
                tickvals=colticks,
                ticktext=coltext,
                xanchor='left',
                x=0
                ),
            size=3,
            showscale=True),
        customdata=customdata_list,
        hovertemplate=
            '<b>Event ID:</b> %{customdata[0]} <br>'
            '<b>Class:</b> %{customdata[20]} <br>'
            '<br>'
            '<b>Fault parameters:</b> <br>'
            '<b>Rupture radius (m):</b> %{customdata[2]:.0f} <br>'
            '<b>Fault Plane Orientation:</b> %{customdata[12]} / %{customdata[13]} <br>'
            '<b>kappa:</b> %{customdata[10]:.0f} <br>'
            '<b>beta:</b> %{customdata[11]:.0f} <br>'
            '<br>'
            '<b>Stress parameters:</b> <br>'
            '<b>Rake:</b> %{customdata[17]:.0f} <br>',
        legendgroup='hypocenter',
        name='Relocated hypocenters',
        showlegend=True,
        visible='legendonly',
        )
    fig.add_traces(trace)
    
    # Plot outliers (events with clust_labels = -1)
    if 'clust_labels' in df.columns:
        outliers = df[df['clust_labels'] == -1]
        if not outliers.empty:
            trace = go.Scatter3d(
                x=outliers['X'],
                y=outliers['Y'],
                z=outliers['Z'],
                mode='markers',
                marker=dict(
                    color='black',
                    opacity=0.5,
                    size=3,
                    showscale=True),
                legendgroup='hypocenter_outliers',
                name='Relocated hypocenters (outliers)',
                showlegend=True,
                visible='legendonly',
                )
            fig.add_traces(trace)

    ############################################################################
    # Plot error ellipsoids
    # Workaround to only show one legend entry for fault planes: create an array
    # with only the first value True with the length of the number of events
    ex_dropna = df['EX'].dropna()
    if len(ex_dropna) > 0:
        idx = ex_dropna.index[0]
        legend_show = [False for i in range(len(df))]
        legend_show[idx] = True
    else:
        legend_show = [False for i in range(len(df))]

    for i in range(len(df)):
        # Create error ellipse at the zero point
        phi = np.linspace(0, 2 * np.pi, 10)
        theta = np.linspace(-np.pi / 2, np.pi / 2, 10)
        phi, theta = np.meshgrid(phi, theta)
        x = np.cos(theta) * np.sin(phi) * df.iloc[i]['EX'] * 3
        y = np.cos(theta) * np.cos(phi) * df.iloc[i]['EY'] * 3
        z = np.sin(theta) * df.iloc[i]['EZ'] * 3
    
        # Shift error ellipse to the right xyz coordinates
        x = x + df.iloc[i]['X']
        y = y + df.iloc[i]['Y']
        z = z + df.iloc[i]['Z']
    
        trace = go.Mesh3d(x=x.flatten(),
                          y=y.flatten(),
                          z=z.flatten(),
                          color='grey',
                          opacity=0.2,
                          alphahull=0,
                          hoverinfo='none',
                          showlegend=legend_show[i],
                          name='Error ellipsoids (3σ)',
                          legendgroup='Error ellipsoids (3σ)',
                          visible='legendonly')
        fig.add_trace(trace)
    
    # Plot error ellipsoids of outliers (events with clust_labels = -1)
    if 'clust_labels' in df.columns:
        outliers = df[df['clust_labels'] == -1]
        if not outliers.empty:
            ex_outliers_dropna = outliers['EX'].dropna()
            if len(ex_outliers_dropna) > 0:
                idx = ex_outliers_dropna.index[0]
                legend_show = [False for i in range(len(outliers))]
                legend_show[idx - outliers.index[0]] = True  # Adjust index for subset
            else:
                legend_show = [False for i in range(len(outliers))]

            for i, (row_idx, row) in enumerate(outliers.iterrows()):
                # Create error ellipse at the zero point
                phi = np.linspace(0, 2 * np.pi, 10)
                theta = np.linspace(-np.pi / 2, np.pi / 2, 10)
                phi, theta = np.meshgrid(phi, theta)
                x = np.cos(theta) * np.sin(phi) * row['EX'] * 3
                y = np.cos(theta) * np.cos(phi) * row['EY'] * 3
                z = np.sin(theta) * row['EZ'] * 3
            
                # Shift error ellipse to the right xyz coordinates
                x = x + row['X']
                y = y + row['Y']
                z = z + row['Z']
            
                trace = go.Mesh3d(x=x.flatten(),
                                y=y.flatten(),
                                z=z.flatten(),
                                color='grey',
                                opacity=0.2,
                                alphahull=0,
                                hoverinfo='none',
                                showlegend=legend_show[i],
                                name='Error ellipsoids (outliers) (3σ)',
                                legendgroup='Error ellipsoids (outliers) (3σ)',
                                visible='legendonly')
                fig.add_trace(trace)

    ############################################################################
    # Plot the calculated active planes
    if 'Strike1' in df.columns:
        # Colors for focal planes
        colormap = 'RdYlGn_r'
        if 'epsilon' in df.columns:
            column = np.array(df['epsilon'])
            if np.all(np.isnan(column)):
                colors = ['gray'] * len(df)
            else:
                minval = 0
                maxval = np.nanmax(column)
                maxval = math.ceil(maxval * 10) / 10.0
                colorsteps = 40
                colors = utilities_plot.colorscale(column, colormap, minval, maxval, colorsteps)
        else:
            colors = ['gray'] * len(df)
        
        # Workaround to only show one legend entry for fault planes: create an array
        # with only the first value True with the length of the number of events
        strike1_dropna = df['Strike1'].dropna()
        if len(strike1_dropna) > 0:
            idx = strike1_dropna.index[0]
            legend_show = [False for i in range(len(df))]
            legend_show[idx] = True
        else:
            legend_show = [False for i in range(len(df))]
    
        show_pref_unpref_legend = True
        for i in range(len(df)):
            if pd.isnull(df.iloc[i]['Strike1']) is True:
                pass
            else:
                # Select the focal plane with the smaller angular difference to the
                # reconstructed fault plane
                if df.iloc[i]['pref_foc'] == 1:
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df.iloc[i]['Strike1']
                                                            + 90 % 360,
                                                            df.iloc[i]['Dip1'])
                    nor_pref = np.array([nor_x, nor_y, nor_z])
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df.iloc[i]['Strike2']
                                                            + 90 % 360,
                                                            df.iloc[i]['Dip2'])
                    nor_nonpref = np.array([nor_x, nor_y, nor_z])
                    foc_color = 'black'
        
                elif df.iloc[i]['pref_foc'] == 2:
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df.iloc[i]['Strike2']
                                                            + 90 % 360,
                                                            df.iloc[i]['Dip2'])
                    nor_pref = np.array([nor_x, nor_y, nor_z])
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df.iloc[i]['Strike1']
                                                            + 90 % 360,
                                                            df.iloc[i]['Dip1'])
                    nor_nonpref = np.array([nor_x, nor_y, nor_z])
                    foc_color = 'black'
                else:
                    continue                    
                    
                # Get XYZ coordinates of the points of the circular fault plane
                x = df.iloc[i]['X']
                y = df.iloc[i]['Y']
                z = df.iloc[i]['Z']
                p = [x, y, z]
                r = df.iloc[i]['rupt_radius']
                X_pref, Y_pref, Z_pref = utilities_plot.circleplane(p, r, nor_pref)
                X_nonpref, Y_nonpref, Z_nonpref = utilities_plot.circleplane(p, r, nor_nonpref)
        
                # Preferred focal plane
                focals_pref = go.Scatter3d(
                    x=X_pref,
                    y=Y_pref,
                    z=Z_pref,
                    mode='lines',
                    line=dict(
                        color=colors[i],
                        # color=foc_color,
                        width=10),
                    hoverinfo='none',
                    legendgroup='FM: calc. act. plane',
                    name='FM: calc. act. plane',
                    showlegend=show_pref_unpref_legend,
                    visible='legendonly'
                    )
                
                # Rake of preferred focal
                r = df.iloc[i]['rupt_radius']
                if pd.isnull(df.iloc[i]['pref_foc']):
                    # No preferred focal plane data - use default (plane 1)
                    rake = df.iloc[i]['Rake1']
                    rake_color = 'gray'
                elif df.iloc[i]['pref_foc'] == 1:
                    rake = df.iloc[i]['Rake1']
                    rake_color = 'black'
                elif df.iloc[i]['pref_foc'] == 2:
                    rake = df.iloc[i]['Rake2']
                    rake_color = 'black'
                else:
                    rake = df.iloc[i]['Rake1']
                    rake_color = 'lightgrey'
                u, v, w = utilities_plot.slipvector_3D(p, r, nor_pref, rake)
                xx = [x + (x - u), u]
                yy = [y + (y - v), v]
                zz = [z + (z - w), w]
                trace = go.Scatter3d(x=xx, y=yy, z=zz,
                                      mode='lines',
                                      line=dict(
                                          color=rake_color,
                                          width=10),
                                      legendgroup='FM: calc. act. plane (Slip vectors)',
                                      hoverinfo='none',
                                      name='FM: calc. act. plane (Slip vectors)',
                                      showlegend=show_pref_unpref_legend,
                                      visible='legendonly'
                                      )
                fig.add_trace(trace)

                # Non-preferred focal plane
                focals_nonpref = go.Scatter3d(
                    x=X_nonpref,
                    y=Y_nonpref,
                    z=Z_nonpref,
                    mode='lines',
                    line=dict(
                        color=foc_color,
                        width=5),
                    hoverinfo='none',
                    legendgroup='FM: calc. non-act. plane',
                    name='FM: calc. non-act. plane',
                    showlegend=show_pref_unpref_legend,
                    visible='legendonly'
                    )
                fig.add_traces([focals_pref, focals_nonpref])

                show_pref_unpref_legend = False

    ############################################################################
    # Plot the known active planes
    if 'Strike1' in df.columns and 'A' in df.columns:
    
        do_known = False
        show_known_legend = True
        show_unk_legend = True
        for i in range(len(df)):
            if pd.isnull(df.iloc[i]['A']) is True:
                pass
            else:
                if df.iloc[i]['A'] == 1:
                    known_strike = 'Strike1'
                    known_dip = 'Dip1'
                    foc_color = 'black'
                    do_known = True
        
                elif df.iloc[i]['A'] == 2:
                    known_strike = 'Strike2'
                    known_dip = 'Dip2'
                    foc_color = 'black'
                    do_known = True

                if do_known:
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df.iloc[i][known_strike]
                                                            + 90 % 360,
                                                            df.iloc[i][known_dip])
                    nor_known = np.array([nor_x, nor_y, nor_z])
                    foc_color = 'black'                    
                        
                    # Get XYZ coordinates of the points of the circular fault plane
                    x = df.iloc[i]['X']
                    y = df.iloc[i]['Y']
                    z = df.iloc[i]['Z']
                    p = [x, y, z]
                    r = df.iloc[i]['rupt_radius']
                    X_known, Y_known, Z_known = utilities_plot.circleplane(p, r, nor_known)
            
                    # Known focal plane
                    focals_known = go.Scatter3d(
                        x=X_known,
                        y=Y_known,
                        z=Z_known,
                        mode='lines',
                        line=dict(
                            color=foc_color,
                            width=10),
                        hoverinfo='none',
                        legendgroup='FM: known act. plane',
                        name='FM: known act. plane',
                        showlegend=show_known_legend,
                        visible='legendonly'
                        )
                    
                    fig.add_traces([focals_known])

                    # Rake of known focal
                    r = df.iloc[i]['rupt_radius']
                    if df.iloc[i]['A'] == 1:
                        rake = df.iloc[i]['Rake1']
                        rake_color = 'black'
                    elif df.iloc[i]['A'] == 2:
                        rake = df.iloc[i]['Rake2']
                        rake_color = 'black'
                    u, v, w = utilities_plot.slipvector_3D(p, r, nor_known, rake)
                    xx = [x + (x - u), u]
                    yy = [y + (y - v), v]
                    zz = [z + (z - w), w]
                    trace = go.Scatter3d(x=xx, y=yy, z=zz,
                                        mode='lines',
                                        line=dict(
                                            color=rake_color,
                                            width=30),
                                        legendgroup='FM: known act. plane (Slip vectors)',
                                        hoverinfo='none',
                                        name='FM: known act. plane (Slip vectors)',
                                        showlegend=show_known_legend,
                                        visible='legendonly'
                                        )
                    fig.add_trace(trace)

                    do_known = False
                    show_known_legend = False

                if df.iloc[i]['A'] == 0:
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df.iloc[i]['Strike1']
                                                            + 90 % 360,
                                                            df.iloc[i]['Dip1'])
                    nor_unk1 = np.array([nor_x, nor_y, nor_z])
                    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(df.iloc[i]['Strike2']
                                                            + 90 % 360,
                                                            df.iloc[i]['Dip2'])
                    nor_unk2 = np.array([nor_x, nor_y, nor_z])
                        
                    # Get XYZ coordinates of the points of the circular fault plane
                    x = df.iloc[i]['X']
                    y = df.iloc[i]['Y']
                    z = df.iloc[i]['Z']
                    p = [x, y, z]
                    r = df.iloc[i]['rupt_radius']
                    X_unk1, Y_unk1, Z_unk1 = utilities_plot.circleplane(p, r, nor_unk1)
                    X_unk2, Y_unk2, Z_unk2 = utilities_plot.circleplane(p, r, nor_unk2)
            
                    foc_color = 'grey'
                    # Unknown focal plane 1
                    focals_unk1 = go.Scatter3d(
                        x=X_unk1,
                        y=Y_unk1,
                        z=Z_unk1,
                        mode='lines',
                        line=dict(
                            color=foc_color,
                            dash='dashdot',
                            width=10),
                        hoverinfo='none',
                        legendgroup='FM: unk. act. plane (1)',
                        name='FM: unk. act. plane (1)',
                        showlegend=show_unk_legend,
                        visible='legendonly'
                        )
                    
                    # Unknown focal plane 2
                    focals_unk2 = go.Scatter3d(
                        x=X_unk2,
                        y=Y_unk2,
                        z=Z_unk2,
                        mode='lines',
                        line=dict(
                            color=foc_color,
                            dash='dot',
                            width=10),
                        hoverinfo='none',
                        legendgroup='FM: unk. act. plane (2)',
                        name='FM: unk. act. plane (2)',
                        showlegend=show_unk_legend,
                        visible='legendonly'
                        )
                    fig.add_traces([focals_unk1, focals_unk2])

                    show_unk_legend = False

    ############################################################################
    # Plot the fault planes
    # Prioritize final_cluster_id (orientation + spatial) over orient_cluster (orientation only)
    if 'final_cluster_id' in df.columns and not df['final_cluster_id'].isna().all():
        # Use combined orientation + spatial clustering
        column = df['final_cluster_id']
        
        # Convert string IDs to numeric for coloring (handle FS0001, F1_0, etc.)
        numeric_ids = []
        unique_ids = {}
        counter = 0
        for val in column:
            if pd.isna(val):
                numeric_ids.append(np.nan)
            else:
                str_val = str(val)
                if str_val not in unique_ids:
                    unique_ids[str_val] = counter
                    counter += 1
                numeric_ids.append(unique_ids[str_val])
        
        column = pd.Series(numeric_ids)
        cmap = 'gnuplot'
        minval = np.nanmin(column) - 1.1
        maxval = np.nanmax(column) + 0.1
        colorsteps = len(unique_ids)
        colors = utilities_plot.colorscale(column, cmap, minval, maxval, colorsteps, cmap_reverse=False)
    elif 'orient_cluster' in df.columns and not df['orient_cluster'].isna().all():
        # Fallback to orientation-only clustering
        column = df['orient_cluster']
        cmap = 'gnuplot'
        minval = np.nanmin(df['orient_cluster']) - 1.1
        maxval = np.nanmax(df['orient_cluster']) + 0.1
        colorsteps = len(df['orient_cluster'].unique())
        colors = utilities_plot.colorscale(column, cmap, minval, maxval, colorsteps, cmap_reverse=False)
    else:
        colors = ['black'] * len(df)

    column = df['kappa']
    minval = 0
    # maxval = np.nanmax(column)/2
    maxval = 100000
    opac = utilities_plot.opacity(column, minval, maxval, 1000)

    # Workaround to only show one legend entry for fault planes: create an array
    # with only the first value True with the length of the number of events
    legend_show = [False for i in range(len(df))]
    idx = df['rupt_plane_azi'].dropna()
    try:
        idx = idx.index[0]   
        legend_show[idx] = True
    except IndexError:
        print("indexerror")
        pass
    
    for i in range(len(df)):
        # Get XYZ coordinates of the points of the circular fault plane around the
        # hypocenter (point p)
        p = [df.iloc[i]['X'],
              df.iloc[i]['Y'],
              df.iloc[i]['Z']
              ]
        r = df.iloc[i]['rupt_radius']
        nor = np.array([df.iloc[i]['nor_x_mean'],
                        df.iloc[i]['nor_y_mean'],
                        df.iloc[i]['nor_z_mean']])
        X, Y, Z = utilities_plot.circleplane(p, r, nor)

        faults = go.Scatter3d(
            x=X,
            y=Y,
            z=Z,
            mode='lines',
            line=dict(
                color=colors[i],
                width=6),
            opacity=0.8,
            hoverinfo='none',
            legendgroup='3D Fault Model',
            name='3D Fault Model',
            showlegend=legend_show[i],
            visible='legendonly',
            )
        fig.add_traces(faults)

    df_k = df[df['rupt_plane_dip'].isna()]
    df_k = df_k[df_k['clust_labels'] != -1]     # Remove outliers from df_k (-1 in clust_labels)
    df_k = df_k.reset_index(drop=True)
    trace = go.Scatter3d(
        x=df_k['X'],
        y=df_k['Y'],
        z=df_k['Z'],
        mode='markers',
        marker=dict(
            color='rgba(0, 0, 0, 0.2)',
            size=2,
            showscale=False),
        legendgroup='3D Fault Model',
        showlegend=False,
        visible='legendonly',
        )
    fig.add_traces(trace)

    ############################################################################
    # Plot stress states
    if 'instab' in df.columns:
        # Plot the fault planes with fault instability
        colormap = 'plasma'
        column = np.array(df['instab'])
        minval = 0
        # maxval = np.nanmax(column)
        maxval = 1
        colorsteps = 50
        colors = utilities_plot.colorscale(column, colormap, minval, maxval, colorsteps)
        
        for i in range(len(df)):
            # Get XYZ coordinates of the points of the circular fault plane around the
            # hypocenter (point p)
            p = [df.iloc[i]['X'],
                  df.iloc[i]['Y'],
                  df.iloc[i]['Z']
                  ]
            r = df.iloc[i]['rupt_radius']
            nor = np.array([df.iloc[i]['nor_x_mean'],
                            df.iloc[i]['nor_y_mean'],
                            df.iloc[i]['nor_z_mean']])
            X, Y, Z = utilities_plot.circleplane(p, r, nor)
        
            faults = go.Scatter3d(
                x=X,
                y=Y,
                z=Z,
                mode='lines',
                line=dict(
                    color=colors[i],
                    width=6),
                hoverinfo='none',
                legendgroup='Fault instability I',
                name='Fault instability I',
                showlegend=legend_show[i],
                visible='legendonly'
                )
            fig.add_traces(faults)
            
        df_k = df[df['rupt_plane_dip'].isna()]
        df_k = df_k.reset_index(drop=True)
        trace = go.Scatter3d(
            x=df_k['X'],
            y=df_k['Y'],
            z=df_k['Z'],
            mode='markers',
            marker=dict(
                color='rgba(0, 0, 0, 0.2)',
                size=2,
                showscale=False),
            legendgroup='Fault instability I',
            name='Fault instability I',
            hoverinfo='none',
            showlegend=False,
            visible='legendonly',
            )
        fig.add_traces(trace)

        
        # Plot the slip vectors
        colormap = 'twilight_shifted'
        column = np.array(df['rake'])
        minval = -180
        maxval = 180
        colorsteps = 360
        colors = utilities_plot.colorscale(column, colormap, minval, maxval, colorsteps)
        
        for i in range(len(df)):
            if df.iloc[i]['rake'] == np.nan:
                pass
            else:
                x = df.iloc[i]['X']
                y = df.iloc[i]['Y']
                z = df.iloc[i]['Z']
                p = [x, y, z]
                r = df.iloc[i]['rupt_radius']
                nor = np.array([df.iloc[i]['nor_x_mean'],
                                df.iloc[i]['nor_y_mean'],
                                df.iloc[i]['nor_z_mean']])
                if df.iloc[i]['rake'] < 0:
                    rake = df.iloc[i]['rake']
                else:
                    rake = df.iloc[i]['rake']
                u, v, w = utilities_plot.slipvector_3D(p, r, nor, rake)
                xx = [x + (x - u), u]
                yy = [y + (y - v), v]
                zz = [z + (z - w), w]
                trace = go.Scatter3d(x=xx, y=yy, z=zz,
                                      mode='lines',
                                      line=dict(
                                          color=colors[i],
                                          width=6),
                                      legendgroup='Slip vectors',
                                      name='Slip vectors',
                                      hoverinfo='none',
                                      showlegend=legend_show[i],
                                      visible='legendonly'
                                      )
                fig.add_trace(trace)
        
        df_k = df[df['rupt_plane_dip'].isna()]
        df_k = df_k.reset_index(drop=True)
        trace = go.Scatter3d(
            x=df_k['X'],
            y=df_k['Y'],
            z=df_k['Z'],
            mode='markers',
            marker=dict(
                color='rgba(0, 0, 0, 0.2)',
                size=2,
                showscale=False),
            legendgroup='Slip vectors',
            name='Slip vectors',
            hoverinfo='none',
            showlegend=False,
            visible='legendonly',
            )
        fig.add_traces(trace)
        
    else:
        pass

    ############################################################################
    # Plot interpolated fault planes (if available)
    try:
        # Check if interpolated planes exist
        out_path = input_params.get('out_dir', './output')
        vtp_dir = os.path.join(out_path, 'vtp_export')
        
        if os.path.exists(vtp_dir):
            # Look for individual fault plane VTP files
            VTP_files = [f for f in os.listdir(vtp_dir) if f.startswith('fault_') and f.endswith('.vtp')]
            
            if VTP_files:
                print(f"Adding {len(VTP_files)} interpolated fault planes to 3D model...")
                
                # Color palette for different fault planes
                interp_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
                
                # Track legend entries to avoid duplicates
                legend_shown = {'interpolated_planes': False}
                
                for i, VTP_file in enumerate(VTP_files):
                    try:
                        VTP_path = os.path.join(vtp_dir, VTP_file)
                        
                        # Load VTP mesh
                        mesh = pv.read(VTP_path)
                        
                        if mesh.n_points > 0:
                            # Get vertices and faces
                            vertices = mesh.points
                            
                            # Extract triangle faces for plotly
                            if mesh.n_cells > 0:
                                # Get triangular faces
                                faces = []
                                for cell_id in range(mesh.n_cells):
                                    cell = mesh.get_cell(cell_id)
                                    if cell.n_points == 3:  # Triangle
                                        faces.append(cell.point_ids)
                                
                                if faces:
                                    faces = np.array(faces)
                                    
                                    # Create mesh3d trace
                                    color = interp_colors[i % len(interp_colors)]
                                    
                                    # Extract orientation info from filename
                                    file_info = VTP_file.replace('fault_plane_', '').replace('.vtp', '')
                                    
                                    trace = go.Mesh3d(
                                        x=vertices[:, 0],
                                        y=vertices[:, 1], 
                                        z=vertices[:, 2],
                                        i=faces[:, 0],
                                        j=faces[:, 1],
                                        k=faces[:, 2],
                                        color=color,
                                        opacity=0.6,
                                        name='Interpolated Fault Planes',
                                        legendgroup='interpolated_planes',
                                        showlegend=not legend_shown['interpolated_planes'],
                                        hovertemplate=f'<b>Interpolated Fault Plane</b><br>{file_info}<br><extra></extra>',
                                        visible='legendonly'
                                    )
                                    
                                    fig.add_trace(trace)
                                    legend_shown['interpolated_planes'] = True
                                    
                    except Exception as e:
                        print(f"Warning: Could not load interpolated plane {VTP_file}: {e}")
                        continue
                
                print("✓ Interpolated fault planes added to 3D model")
            else:
                print("No individual fault plane VTP files found")
        else:
            print("No interpolated_planes directory found")
            
    except Exception as e:
        print(f"Warning: Could not add interpolated planes to 3D model: {e}")

    ############################################################################
    # Layout settings
    # Calculate the "empty white box" to ensure equal axes of the 3D plot
    # Work-around to ensure equal axes of the 3D plot using all data_input
    x_range, y_range, z_range = utilities_plot.equal_axes(df['X'],
                                            df['Y'],
                                            df['Z'])

    # Cameraview standard (Top view)
    eye = dict(x=0, y=-0.1, z=2)    
    
    # Define the figure layout parameters
    fig.update_layout(
        template='plotly_white',
        title=f'Hypocenter-Based Imaging of Active Faults (Truttmann et al. 2023): {input_params.get("project_title", "Fault Imaging Analysis")}',
        hovermode=None,
        showlegend=True,
        legend={'itemclick': 'toggle'},
        scene=dict(
            xaxis_title='Easting [m]',
            yaxis_title='Northing [m]',
            zaxis_title='Depth [m]',
            xaxis=dict(
                range=x_range,
                tickformat='d',
                separatethousands=True,
                showspikes=False,
                showgrid=True,
                zeroline=True
                ),
            yaxis=dict(
                range=y_range,
                tickformat='d',
                separatethousands=True,
                showspikes=False,
                showgrid=True,
                zeroline=True
                ),
            zaxis=dict(
                range=z_range,
                tickformat='d',
                separatethousands=True,
                showspikes=False,
                showgrid=True,
                zeroline=True
                ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=eye)
            ),
        margin=dict(
            l=0,
            r=20,
            b=20,
            t=40)
        )
    
    fig.update_xaxes(title_standoff=20)
    
    # Save output
    out_path = input_params['out_dir']
    os.makedirs(out_path, exist_ok=True)

    fig.write_html(out_path + '/3D_model.html')
    print(f"3D model saved to: {out_path}/3D_model.html")

    return


