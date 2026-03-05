#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS: Fault Network Reconstruction Module

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
import scipy as sp
import numba
import multiprocessing as mp
from itertools import zip_longest
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from ..utils.spherical_stats import kent_me
from ..utils import utilities

@numba.njit
def hypo_perturbation(n_mc, _X, _Y, _Z, EX, EY, EZ):
    """
    Create perturbed hypocenter dataset for Monte Carlo simulation.

    Parameters
    ----------
    n_mc : int
        Number of perturbations. If n_mc=1, original coordinates are returned
        without any perturbation (Monte Carlo simulation disabled).
    _X : array
        X coordinates.
    _Y : array
        Y coordinates.
    _Z : array
        Z coordinates.
    EX : array
        Error in X direction.
    EY : array
        Error in Y direction.
    EZ : array
        Error in Z direction.

    Returns
    -------
    tuple of arrays
        Perturbed XYZ hypocenter coordinates in LV95/CH1903+.
        Shape: (n_events, n_mc) for each coordinate array.
        When n_mc=1, returns original coordinates without perturbation.

    """
    
    # If n_mc is set to 1, no perturbation is performed (MC simulation turned off)
    # Use original coordinates directly without any random generation
    if n_mc == 1:
        per_X = _X.reshape(-1, 1).astype(np.float64)  # Ensure float64 type for consistency
        per_Y = _Y.reshape(-1, 1).astype(np.float64)
        per_Z = _Z.reshape(-1, 1).astype(np.float64)

    # Else, perturbation is performed (MC simulation turned on)
    else:
        # Create empty arrays for the perturbed hypocenter locations
        per_X = np.empty((_X.shape[0], n_mc), dtype=np.float64)
        per_X[:] = np.nan
        per_Y = np.empty((_Y.shape[0], n_mc), dtype=np.float64)
        per_Y[:] = np.nan
        per_Z = np.empty((_Z.shape[0], n_mc), dtype=np.float64)
        per_Z[:] = np.nan
        
        # Calculate all perturbed hypocenters
        # Assumption: NORMAL distribution within the error
        # Note: scale parameter (= sigma) is approximately 1/3 of the total
        # error (then it catches 99.7 % of the data according to normal
        # distribution properties)
        for i in range(_X.shape[0]):
            per_X[i,:] = np.random.normal(loc=_X[i], scale=EX[i], size=n_mc)
            per_Y[i,:] = np.random.normal(loc=_Y[i], scale=EY[i], size=n_mc)
            per_Z[i,:] = np.random.normal(loc=_Z[i], scale=EZ[i], size=n_mc)
        
    return per_X, per_Y, per_Z


def nearestneighbors(p, X, r):
    """
    Find the indices and distances of the nearest neighbors to point p in dataset X within a radius r.
    
    Parameters
    ----------
    p : int
        Master event to apply nearest neighbor search on.
    X : array
        Point cloud to be analyzed.
    r : int
        Search radius.

    Returns
    -------
    Indices and distances of points within radius r to point p.

    """
    # Define nearest neighbor parameters
    neigh = NearestNeighbors(radius=r, algorithm='brute', metric='euclidean')

    # Fit the nearest neighbors estimator to the dataset X
    neigh.fit(X)

    # Find the neighbors within a given radius of point p
    rng = neigh.radius_neighbors([p])

    # Save the indices of the points within radius r into NN_idx
    NN_idx = np.asarray(rng[1][0])
    NN_dist = np.asarray(rng[0][0])

    return(NN_idx, NN_dist)


def generate_focal_mechanism_points(df_row, radius_interval=10.0, point_density_meters=10.0):
    """
    Generate synthetic points along the active focal mechanism plane using systematic radial distribution.
    
    This function generates points with the same systematic approach as _generate_fault_plane_points:
    1. Center point (hypocenter)
    2. Complete circles at fixed radius intervals
    3. Edge circle at the full radius
    
    Parameters
    ----------
    df_row : pandas.Series
        Row from dataframe containing focal mechanism and location data
    radius_interval : float, default=10.0
        Fixed interval in meters between concentric circles
    point_density_meters : float, default=10.0
        Target distance in meters between points on each circle circumference
        
    Returns
    -------
    np.array or None
        Array of shape (num_points, 3) with X, Y, Z coordinates of synthetic points,
        or None if no valid active plane is available
    """
    
    # Check if we have valid focal mechanism data with known active plane
    if pd.isna(df_row.get('A', np.nan)) or df_row.get('A', 0) not in [1, 2]:
        return None
    
    # Get the active plane parameters
    active_plane = int(df_row['A'])
    if active_plane == 1:
        strike = df_row.get('Strike1', np.nan)
        dip = df_row.get('Dip1', np.nan)
    elif active_plane == 2:
        strike = df_row.get('Strike2', np.nan)
        dip = df_row.get('Dip2', np.nan)
    
    # Check if we have valid strike and dip values
    if pd.isna(strike) or pd.isna(dip):
        return None
    
    # Validate parameter ranges
    if not (0 <= strike <= 360 and 0 <= dip <= 90):
        return None
    
    # Get hypocenter location (center of circular plane)
    p = np.array([df_row.get('X', 0.0), df_row.get('Y', 0.0), df_row.get('Z', 0.0)])

    # Calculate fault radius - use a reasonable estimate based on typical values
    # If rupture radius is available from other calculations, use it; otherwise estimate
    if 'rupt_radius' in df_row and not pd.isna(df_row['rupt_radius']):
        r = df_row['rupt_radius']
    else:
        # Calculate radius based on magnitude using Leonard (2014) scaling relationship
        if 'MAG' in df_row and not pd.isna(df_row['MAG']):
            mag = float(df_row['MAG'])  # Convert to float for numba function
            
            # Convert to Mw using the same ML_to_MW function used elsewhere in the codebase
            Mw = ML_to_MW(mag)
            
            # Calculate rupture area and radius using Leonard (2014) scaling
            A, r = faultscalingL14_Mag_A(Mw)
        else:
            raise ValueError("Cannot estimate rupture radius: missing 'rupt_radius' and 'MAG' in df_row")
    
    # Convert strike and dip to normal vector using correct geological convention
    # Strike: measured clockwise from North (0° = North, 90° = East)
    # Dip: measured downward from horizontal (0° = horizontal, 90° = vertical)
    # Convert strike to dip direction azimuth (same as in model_validation.py)
    dip_azimuth = (strike + 90) % 360
    
    # Use the same normal vector calculation as in utilities.plane_azidip_to_normal
    from ..utils import utilities
    nx, ny, nz = utilities.plane_azidip_to_normal(dip_azimuth, dip)
    
    # Normalize the normal vector (should already be normalized from utilities function)
    nor = np.array([nx, ny, nz])
    nor = nor / np.linalg.norm(nor)
    
    # Ensure normal vector points downward (z-component negative) for consistency
    if nor[2] > 0:
        nor = -nor
    
    # Create two orthonormal vectors in the plane (same method as _generate_fault_plane_points)
    if abs(nor[2]) < 0.9:
        v1 = np.cross(nor, [0, 0, 1])
    else:
        v1 = np.cross(nor, [1, 0, 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(nor, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    plane_points = []
    
    # 1. Always include the center point (hypocenter)
    plane_points.append(p.copy())
    
    # 2. Generate full circles at fixed radius intervals (same as _generate_fault_plane_points)
    # Calculate radii at fixed intervals up to the fault radius
    radii = []
    current_radius = radius_interval
    while current_radius <= r:
        radii.append(current_radius)
        current_radius += radius_interval
    
    # Always include the edge (full radius) if it's not already included
    if len(radii) == 0 or radii[-1] < r:
        radii.append(r)
    
    # Generate complete circles at each radius
    for ring_radius in radii:
        # Calculate number of points for this circle based on its circumference
        circumference = 2 * np.pi * ring_radius
        n_points_circle = max(8, int(circumference / point_density_meters))  # Minimum 8 points per circle
        
        # Generate points evenly around the complete circle
        angles = np.linspace(0, 2*np.pi, n_points_circle, endpoint=False)
        
        for angle in angles:
            # Point in local plane coordinates
            local_point = ring_radius * (np.cos(angle) * v1 + np.sin(angle) * v2)
            # Transform to global coordinates
            global_point = p + local_point
            plane_points.append(global_point)
    
    # Convert to numpy array
    points = np.array(plane_points)
    
    return points


def pca_planefit(X, Y, Z):
    """
    Calculate best fit plane of XYZ point dataset using PCA.

    Parameters
    ----------
    X : array
        X coordinates of points.
    Y : array
        Y coordinates of points.
    Z : array
        Z coordinates of points.

    Returns
    -------
    Principal vectors from PCA (e.g. normal vector) and quality parameters.

    """

    # Best-fit plane
    # Assign the first three rows (XYZ) from the input data to a temporary
    # array for best plane fitting (G)
    G = np.concatenate((X[:, None], Y[:, None], Z[:, None]),
                       axis=1).astype('float64')

    # Calculate the covariance matrix
    cov_matrix = np.cov(G.T)

    # Get the eigenvalues and eigenvectors from the covariance matrix
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

    # Calculate the normal unit vector of the plane, which corresponds to
    # the smallest eigenvector (with the minimal eigenvalue)
    nor = eig_vecs[:, np.argmin(eig_vals)]

    # Calculate the unit vector inside the plane (belonging to lambda 1 & 2)
    v1 = eig_vecs[:, np.argmax(eig_vals)]
    v2 = eig_vecs[:, np.argsort(eig_vals)[1]]
    
    # Convert all normal vector to the lower hemisphere, which means that
    # all z-components of the eigenvectors have to be negative
    # ('point downwards')
    if nor[2] > 0:
        nor = nor * -1

    # Ensure that all normal vectors are normalized to a length of 1
    nor = nor / np.linalg.norm(nor)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Plane Fit Robustness Evaluation (based on Jones 2015)
    lam1 = np.max(eig_vals)
    lam2 = np.median(eig_vals)
    lam3 = np.min(eig_vals)
    rat_lam23 = lam2 / (lam3 + 1e-12)    # Prevent division by zero by adding small epsilon
    tot_var = lam1 + lam2 + lam3

    # Define the output of the function
    return (nor, v1, v2, lam1, lam2, lam3, rat_lam23, tot_var)


@numba.njit
def ML_to_MW(ML):
    """
    Convert magnitudes from ML to Mw (after Allmann et al. 2010).

    Parameters
    ----------
    ML : float
        Magnitude ML.

    Returns
    -------
    Magnitude Mw.

    """
    # Use empirical relations from Goertz-Allmann et al. (2011) for Mag transformation
    if ML <= 2:
        Mw = 0.594 * ML + 0.985
    elif ML <= 4:
        Mw = 1.327 + 0.253 * ML + 0.085 * (ML**2)
    elif ML > 4:
        Mw = ML - 0.3

    return Mw


@numba.njit
def faultscalingL14_Mag_A(Mw):
    """
    Convert magnitude to rupture area after Leonard (2014).

    Mw = a + b * log(A)

    Parameters
    ----------
    Mw : int
        Moment magnitude Mw.

    Returns
    -------
    Fault rupture area A (in km2) and the radius r for a cirular fault plane
    (in m).

    """
    # Define the constants a and b for the tectonic setting
    # Constants for SCR SS earthquakes (Leonard 2014, Table 4)
    a = 4.18
    b = 1

    # Calculate the area A from Mw
    A = 10 ** ((Mw - a) / b)

    # Calculate the radius of a circular fault plane
    r = np.sqrt(A / np.pi)     # in km
    r = r * 1000               # in m

    return(A, r)


@numba.njit
def faultscalingL14_Mag_D(Mw):
    """
    Convert magnitude to displacement area after Leonard (2014).

    Mw = a + b * log(D)

    Parameters
    ----------
    Mw : int
        Moment magnitude Mw.

    Returns
    -------
    Fault displacement D (in m).
    """
    # Define the constants a and b for the tectonic setting
    # Constants for SCR SS earthquakes (Leonard 2014, Table 4)
    a = 3.71
    b = 2.0

    # Calculate the magnitude Mw from the fault area
    D = 10 ** ((Mw - a) / b)

    return(D)


@numba.njit
def faultscalingWC94_Mag_A(Mw):
    """
    Convert magnitude to rupture area after Wells & Coppersmith (1994).

    Mw = a + b * log(RA)

    Parameters
    ----------
    Mw : int
        Moment magnitude Mw.

    Returns
    -------
    Fault rupture area A (in km2) and the radius r for a cirular fault plane
    (in m).

    """
    # Define the constants a and b for the tectonic setting
    # Constants for SCR SS earthquakes (Wells & Coppersmith 1994, Table 2A)
    a = 3.98
    b = 1.02

    # Calculate the area A from Mw
    A = 10 ** ((Mw - a) / b)

    # Calculate the radius of a circular fault plane
    r = np.sqrt(A / np.pi)     # in km
    r = r * 1000               # in m

    return(A, r)


@numba.njit
def faultscalingT17_Mag_A(Mw):
    """
    Convert magnitude to rupture area after Thingbaijam (2017).

    Mw = a + b * log(RA)

    Parameters
    ----------
    Mw : int
        Moment magnitude Mw.

    Returns
    -------
    Fault rupture area A (in km2) and the radius r for a cirular fault plane
    (in m).

    """
    # Define the constants a and b for the tectonic setting
    # Constants for SCR SS earthquakes (Thingbaijam 2017, Table 1)
    a = 3.486
    b = 0.942

    # Calculate the area A from Mw
    A = 10 ** ((Mw - a) / b)

    # Calculate the radius of a circular fault plane
    r = np.sqrt(A / np.pi)     # in km
    r = r * 1000               # in m

    return(A, r)


def faultplanes3D(ID, date, X, Y, Z, EX, EY, EZ, r_nn, dt_nn, df_focal=None, use_focal_constraints=False, id_mapping=None):
    """
    Calculate indiv. fault planes from hypocenter with NN and PCA.
    Optionally enhance point cloud with focal mechanism constraints.

    Parameters
    ----------
    X : array
        X coordinates in CH1903+ (m).
    Y : array
        Y coordinates in CH1903+ (m).
    Z : array
        Z coordinates in CH1903+ (m).
    r_nn : int
        Search radius for NN search (m).
    dt_nn : int
        Time window for NN search (+- h).
    df_focal : DataFrame, optional
        Dataframe containing focal mechanism data with columns including A, Strike1, Dip1, Strike2, Dip2
    use_focal_constraints : bool, default=False
        Whether to use focal mechanism data to enhance point cloud for PCA
    id_mapping : dict, optional
        Mapping from sequential IDs to original event IDs for focal mechanism lookup
    p_threshold : int
        Minimum number of hypocenters to allow for fault plane calculation.
    lam2_threshold : int
        Collinearity (λ₂) threshold (according to Jones et al. 2015)
    rat_lam23_threshold : int
        Planarity (λ₂ / λ₃) threshold (according to Jones et al. 2015)

    Returns
    -------
    DataFrame with the orientations of the calculated indiv. fault planes.
    """
    
    ###########################################################################
    # Enhance point cloud with focal mechanism constraints if available
    enhanced_X, enhanced_Y, enhanced_Z = X.copy(), Y.copy(), Z.copy()
    enhanced_ID = ID.copy()
    enhanced_date = date.copy()
    focal_point_indices = []  # Track which indices are focal mechanism points
    
    if use_focal_constraints and df_focal is not None and id_mapping is not None:
        print("Enhancing point cloud with focal mechanism constraints before NN search...")
        
        # For each event, check if it has focal mechanism data and add focal points
        focal_points_added = 0
        for i in range(len(X)):
            # Map sequential ID back to original ID using the provided mapping
            if ID[i] in id_mapping:
                original_id = id_mapping[ID[i]]
                
                if original_id in df_focal.index:
                    focal_row = df_focal.loc[original_id]
                    
                    # Add hypocenter coordinates to focal row for point generation
                    focal_row_enhanced = focal_row.copy()
                    focal_row_enhanced['X'] = X[i]
                    focal_row_enhanced['Y'] = Y[i] 
                    focal_row_enhanced['Z'] = Z[i]
                    focal_row_enhanced['MAG'] = focal_row.get('MAG', np.nan)  # Include magnitude if available
                    
                    focal_points = generate_focal_mechanism_points(
                        focal_row_enhanced, 
                        radius_interval=25.0,  # Use smaller intervals for denser point cloud
                        point_density_meters=25.0  # Adaptive point density based on search radius
                    )
                    
                    if focal_points is not None and len(focal_points) > 1:  # Skip center point (hypocenter)
                        # Skip the first point (center/hypocenter) since it's already in the dataset
                        new_focal_points = focal_points[1:]  
                        
                        # Add focal mechanism points to the enhanced arrays
                        enhanced_X = np.concatenate([enhanced_X, new_focal_points[:, 0]])
                        enhanced_Y = np.concatenate([enhanced_Y, new_focal_points[:, 1]])
                        enhanced_Z = np.concatenate([enhanced_Z, new_focal_points[:, 2]])
                        
                        # Extend ID and date arrays for focal points (use same ID and date as parent event)
                        enhanced_ID = np.concatenate([enhanced_ID, np.full(len(new_focal_points), ID[i])])
                        enhanced_date = np.concatenate([enhanced_date, np.full(len(new_focal_points), date[i])])
                        
                        # Track which indices are focal mechanism points (starting from current length)
                        start_idx = len(enhanced_X) - len(new_focal_points)
                        focal_point_indices.extend(range(start_idx, len(enhanced_X)))
                        
                        focal_points_added += len(new_focal_points)
                        print(f"Event {original_id}: Added {len(new_focal_points)} focal mechanism points")
        
        print(f"Total focal mechanism points added: {focal_points_added}")
        print(f"Enhanced point cloud: {len(X)} original + {focal_points_added} focal = {len(enhanced_X)} total points")
    
    # Use enhanced coordinates for nearest neighbor search
    search_X, search_Y, search_Z = enhanced_X, enhanced_Y, enhanced_Z
    search_ID, search_date = enhanced_ID, enhanced_date
    
    ###########################################################################
    # Search nearest neighbors
    NN_idx_list = []
    NN_dist_list = []
    for i in range(len(X)):  # Still iterate over original events only
        # Execute nearest neighbor search using enhanced point cloud
        NN_idx, NN_dist = nearestneighbors([X[i], Y[i], Z[i]],
                                           np.column_stack((search_X, search_Y, search_Z)),
                                           r_nn)

        # Store indices of nearest neighbors
        NN_idx_list.append(NN_idx)
        NN_dist_list.append(NN_dist)

    # Delete nearest neighbors outside dt_nn threshold
    neigh = []
    for i in range(len(X)):  # Still iterate over original events only
        # Extract the information from the dataset of the respective rows by
        # the
        NN_idx_i = NN_idx_list[i]
        
        # Build neighbor data from enhanced arrays
        neigh_x = search_X[NN_idx_i]
        neigh_y = search_Y[NN_idx_i]
        neigh_z = search_Z[NN_idx_i]
        neigh_date = search_date[NN_idx_i]
        
        # Get the date of event i (master) - use original event date
        date_i = np.datetime64(date[i])
            
        # Calculate time delta between event i and its neighbors
        date_j = np.array(neigh_date, dtype='datetime64[s]')
        dt_nn_ij = np.abs(date_i - date_j)
        dt_nn_ij = dt_nn_ij.astype('timedelta64[h]')
        dt_nn_ij = dt_nn_ij / np.timedelta64(1, 'h')
        
        # Get the indices of the events j outside the dt_nn threshold
        idx_del = np.where(dt_nn_ij > dt_nn)[0]

        # Delete the neighbouring events j outside dt_nn
        neigh_x = np.delete(neigh_x, idx_del, 0)
        neigh_y = np.delete(neigh_y, idx_del, 0)
        neigh_z = np.delete(neigh_z, idx_del, 0)
        neigh_date = np.delete(neigh_date, idx_del, 0)
        
        # Store neighbor coordinates as array
        neigh_coords = np.column_stack((neigh_x, neigh_y, neigh_z))

        # Append the nearest neighbors of point i to the list 'neigh'
        neigh.append(neigh_coords)

    ###########################################################################
    # PCA Plane Fitting

    # Define the plane fitting reliability parameters
    # Minimum number of points to allow for best fit plane calculation
    p_threshold = 5
    # Collinearity (λ₂) (according to Jones et al. 2015)
    # Use the squared mean horizontal (EX, EY) relocation error, but cap it at
    # (r_nn/3)² so the threshold never exceeds what is physically achievable
    # within the search radius. When location errors are large relative to r_nn
    # (e.g. large EZ from sparse networks), an uncapped error-based threshold
    # would be larger than any possible λ₂ value and reject every plane.
    lam2_error_based = np.array([EX.mean(), EY.mean()]).mean() ** 2
    lam2_rnn_cap = (r_nn / 5.0) ** 2
    lam2_threshold = min(lam2_error_based, lam2_rnn_cap)

    # Planarity (λ₂ / λ₃) (according to Jones et al. 2015)
    rat_lam23_threshold = 5

    # Create an empty array
    plane_fit = np.empty((len(ID), 16))
    plane_fit[:] = np.nan

    # Pre-allocate arrays
    X_pca = np.empty(p_threshold)
    Y_pca = np.empty(p_threshold)
    Z_pca = np.empty(p_threshold)
    nor = np.empty(3)
    
    # Loop through all clusters and calculate the best fit plane for each
    # cluster individually
    for i in range(len(ID)):
        data_i = neigh[i]

        # Calculate best fit plane if the number of events within the cluster
        # is larger than p_threshold, otherwise insert NaN
        if len(data_i) >= p_threshold:
            X_pca, Y_pca, Z_pca = data_i[:, 0], data_i[:, 1], data_i[:, 2]
            
            # Perform PCA on the neighbor point cloud (which now includes focal mechanism points if enabled)
            nor, v1, v2, lam1, lam2, lam3, rat_lam23, tot_var = pca_planefit(X_pca, Y_pca, Z_pca)
            
            # Convert all normal vector to the lower hemisphere, which means
            # that all z-components of the eigenvectors have to be negative
            # ('point downwards')
            nor = nor * np.where(nor[2] > 0, -1, 1)

            values = np.hstack([i, nor, v1, v2, lam1, lam2, lam3, rat_lam23, tot_var, len(data_i)])

            # Check whether the plane meets the defined quality criteria
            # !!! IMPORTANT: This may cause troubles in regional analysis!
            # NOTE: maybe the thresholds could be adjusted (less restrictive)
            if not (lam2 > lam2_threshold and rat_lam23 > rat_lam23_threshold):
                values[1:15] = np.nan
        else:
            values = np.hstack([i, [np.nan] * 14, len(data_i)])
        plane_fit[i] = values

    return(plane_fit)


def DBSCAN_outlier_detection(df_hyfi):
    """
    Detect outliers using DBSCAN clustering with automatically calculated parameters.
    Events with valid focal mechanism data (A==1 or A==2) are exempt from outlier removal.
    
    Parameters
    ----------
    df_hyfi : DataFrame
        Single dataframe with hypocenter data including X, Y, Z coordinates
        
    Returns
    -------
    df_hyfi : DataFrame
        Input dataframe with added clust_labels column (-1 for outliers)
    """
    print('\n--- DBSCAN OUTLIER DETECTION ---')
    
    # Calculate parameters based on data distribution
    min_samples = 5

    from sklearn.neighbors import NearestNeighbors
    k = min_samples + 1
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(df_hyfi[['X', 'Y', 'Z']])
    distances, indices = nbrs.kneighbors(df_hyfi[['X', 'Y', 'Z']])

    base_dist = np.percentile(distances[:, -1], 75)
    max_dist = base_dist * 1.5  # 50% more inclusive than base distance

    print(f'  DBSCAN parameters:')
    print(f'    eps (max distance): {max_dist:.2f}')
    print(f'    min_samples: {min_samples}')
    
    # Apply DBSCAN clustering
    clust_alg = 'auto'
    leaf_size = 30
    clust = DBSCAN(eps=max_dist, min_samples=min_samples, metric='euclidean',
                   algorithm=clust_alg, leaf_size=leaf_size, n_jobs=-1)
    clustering = clust.fit(df_hyfi[['X', 'Y', 'Z']])
    
    # Add cluster labels to dataframe
    df_hyfi['clust_labels'] = clustering.labels_
    
    # Protect events with valid focal mechanism data from being marked as outliers
    if 'A' in df_hyfi.columns:
        focal_mask = df_hyfi['A'].notna() & df_hyfi['A'].isin([1, 2])
        focal_outliers = df_hyfi[focal_mask & (df_hyfi['clust_labels'] == -1)]
        
        if len(focal_outliers) > 0:
            print(f'  Protecting {len(focal_outliers)} events with focal mechanisms from outlier removal:')
            for idx, row in focal_outliers.iterrows():
                print(f'    - {row["ID"]}: focal plane A={row["A"]}')
            
            # Assign these events to the largest cluster instead of marking as outliers
            cluster_counts = df_hyfi[df_hyfi['clust_labels'] != -1]['clust_labels'].value_counts()
            if len(cluster_counts) > 0:
                largest_cluster = cluster_counts.index[0]
                df_hyfi.loc[focal_mask & (df_hyfi['clust_labels'] == -1), 'clust_labels'] = largest_cluster
                print(f'    Reassigned focal mechanism events to cluster {largest_cluster}')
            else:
                # If no valid clusters exist, create a new cluster (0) for focal events
                df_hyfi.loc[focal_mask & (df_hyfi['clust_labels'] == -1), 'clust_labels'] = 0
                print(f'    Created new cluster 0 for focal mechanism events')
    
    # Count outliers for reporting (after focal mechanism protection)
    num_outliers = len(df_hyfi[df_hyfi['clust_labels'] == -1])
    num_clusters = len(np.unique(df_hyfi['clust_labels'][df_hyfi['clust_labels'] != -1]))
    print(f'  Final outliers (after focal mechanism protection): {num_outliers}')
    print(f'  Final outlier percentage: {100*num_outliers/len(df_hyfi):.1f}%')
    print(f'  Number of clusters: {num_clusters}')
    
    return df_hyfi


def LOF_outlier_detection(df_hyfi, n_neighbors=20, contamination='auto'):
    """
    Detect outliers using Local Outlier Factor (LOF) algorithm.
    Events with valid focal mechanism data (A==1 or A==2) are exempt from outlier removal.
    
    Parameters
    ----------
    df_hyfi : DataFrame
        Single dataframe with hypocenter data including X, Y, Z coordinates
    n_neighbors : int, default=20
        Number of neighbors to use for LOF computation.
        Higher values smooth out local density variations.
    contamination : float or 'auto', default='auto'
        The proportion of outliers in the dataset.
        - 'auto': automatically determine threshold (not supported for novelty detection)
        - float: expected proportion of outliers (e.g., 0.1 for 10%)
        
    Returns
    -------
    df_hyfi : DataFrame
        Input dataframe with added clust_labels column (-1 for outliers, 0 for inliers)
        
    Notes
    -----
    LOF measures local density deviation of a point with respect to its neighbors.
    Points with substantially lower density than their neighbors are considered outliers.
    Unlike DBSCAN, LOF does not create multiple clusters - it's purely for outlier detection.
    """
    print('\n--- LOCAL OUTLIER FACTOR (LOF) OUTLIER DETECTION ---')
    
    # Automatically tune n_neighbors based on dataset size if not specified
    dataset_size = len(df_hyfi)
    if n_neighbors is None:
        # Rule of thumb: sqrt(N) but capped between 10 and 50
        n_neighbors = int(min(50, max(10, np.sqrt(dataset_size))))
    
    print(f'  LOF parameters:')
    print(f'    n_neighbors: {n_neighbors}')
    print(f'    contamination: {contamination}')
    print(f'    dataset size: {dataset_size}')
    
    # Apply Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, 
                             contamination=contamination,
                             metric='euclidean',
                             n_jobs=-1)
    
    # Fit and predict (-1 for outliers, 1 for inliers)
    outlier_labels = lof.fit_predict(df_hyfi[['X', 'Y', 'Z']])
    
    # Convert to cluster labels format: -1 for outliers, 0 for inliers
    df_hyfi['clust_labels'] = np.where(outlier_labels == -1, -1, 0)
    
    # Get LOF scores (negative outlier factor - more negative = more outlier-like)
    df_hyfi['lof_score'] = lof.negative_outlier_factor_
    
    # Protect events with valid focal mechanism data from being marked as outliers
    if 'A' in df_hyfi.columns:
        focal_mask = df_hyfi['A'].notna() & df_hyfi['A'].isin([1, 2])
        focal_outliers = df_hyfi[focal_mask & (df_hyfi['clust_labels'] == -1)]
        
        if len(focal_outliers) > 0:
            print(f'  Protecting {len(focal_outliers)} events with focal mechanisms from outlier removal:')
            for idx, row in focal_outliers.iterrows():
                print(f'    - {row["ID"]}: focal plane A={row["A"]}, LOF score={row["lof_score"]:.3f}')
            
            # Reassign these events to inliers (cluster 0)
            df_hyfi.loc[focal_mask & (df_hyfi['clust_labels'] == -1), 'clust_labels'] = 0
            print(f'    Reassigned focal mechanism events to inliers (cluster 0)')
    
    # Count outliers for reporting (after focal mechanism protection)
    num_outliers = len(df_hyfi[df_hyfi['clust_labels'] == -1])
    num_inliers = len(df_hyfi[df_hyfi['clust_labels'] == 0])
    
    print(f'  Final outliers (after focal mechanism protection): {num_outliers}')
    print(f'  Final outlier percentage: {100*num_outliers/len(df_hyfi):.1f}%')
    print(f'  Inliers: {num_inliers}')
    
    # Print statistics about LOF scores
    outlier_scores = df_hyfi[df_hyfi['clust_labels'] == -1]['lof_score']
    inlier_scores = df_hyfi[df_hyfi['clust_labels'] == 0]['lof_score']
    
    if len(outlier_scores) > 0:
        print(f'  LOF score statistics:')
        print(f'    Outliers - mean: {outlier_scores.mean():.3f}, min: {outlier_scores.min():.3f}, max: {outlier_scores.max():.3f}')
        if len(inlier_scores) > 0:
            print(f'    Inliers  - mean: {inlier_scores.mean():.3f}, min: {inlier_scores.min():.3f}, max: {inlier_scores.max():.3f}')
    
    return df_hyfi


def IsolationForest_outlier_detection(df_hyfi, n_estimators=100, contamination=0.05, max_samples='auto', random_state=42):
    """
    Detect outliers using Isolation Forest algorithm.
    Events with valid focal mechanism data (A==1 or A==2) are exempt from outlier removal.
    
    Parameters
    ----------
    df_hyfi : DataFrame
        Single dataframe with hypocenter data including X, Y, Z coordinates
    n_estimators : int, default=100
        The number of base estimators (isolation trees) in the ensemble.
        Higher values increase accuracy but also computation time.
    contamination : float or 'auto', default=0.05
        The proportion of outliers in the dataset.
        - 'auto': automatically determine threshold based on data (can be aggressive)
        - float: expected proportion of outliers (default: 0.05 = 5%, conservative)
        Note: Default set to 0.05 (5%) for balanced outlier detection
    max_samples : int, float or 'auto', default='auto'
        Number of samples to draw to train each isolation tree.
        - 'auto': min(256, n_samples)
        - int: exact number of samples
        - float: proportion of samples
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    df_hyfi : DataFrame
        Input dataframe with added clust_labels column (-1 for outliers, 0 for inliers)
        
    Notes
    -----
    Isolation Forest works by randomly selecting features and split values, isolating
    observations in a tree structure. Outliers are easier to isolate (require fewer
    splits) than normal points. The algorithm is particularly effective for high-dimensional
    data and does not rely on distance metrics.
    
    Key advantages:
    - Fast and scalable
    - Works well with high-dimensional data
    - Does not assume any particular distribution
    - Less sensitive to parameter tuning than density-based methods
    
    Conservative default: contamination=0.05 (5%) to avoid over-detection of outliers.
    Increase to 0.1 (10%) for more aggressive outlier removal, decrease to 0.01-0.02 for 
    very conservative removal, or use 'auto' for automatic threshold determination 
    (may be too aggressive for seismic data).
    """
    print('\n--- ISOLATION FOREST OUTLIER DETECTION ---')
    
    dataset_size = len(df_hyfi)
    
    # If contamination is 'auto', warn user and suggest conservative value
    if contamination == 'auto':
        print(f'  WARNING: contamination="auto" can be aggressive. Consider using 0.01-0.1 instead.')
    
    print(f'  Isolation Forest parameters:')
    print(f'    n_estimators: {n_estimators}')
    print(f'    contamination: {contamination} {"(conservative: expects ~5% outliers)" if contamination == 0.05 else ""}')
    print(f'    max_samples: {max_samples}')
    print(f'    random_state: {random_state}')
    print(f'    dataset size: {dataset_size}')
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(n_estimators=n_estimators,
                                  contamination=contamination,
                                  max_samples=max_samples,
                                  random_state=random_state,
                                  n_jobs=-1)
    
    # Fit and predict (-1 for outliers, 1 for inliers)
    outlier_labels = iso_forest.fit_predict(df_hyfi[['X', 'Y', 'Z']])
    
    # Convert to cluster labels format: -1 for outliers, 0 for inliers
    df_hyfi['clust_labels'] = np.where(outlier_labels == -1, -1, 0)
    
    # Get anomaly scores (negative = outlier, positive = inlier)
    # More negative = stronger outlier
    df_hyfi['isolation_score'] = iso_forest.score_samples(df_hyfi[['X', 'Y', 'Z']])
    
    # Protect events with valid focal mechanism data from being marked as outliers
    if 'A' in df_hyfi.columns:
        focal_mask = df_hyfi['A'].notna() & df_hyfi['A'].isin([1, 2])
        focal_outliers = df_hyfi[focal_mask & (df_hyfi['clust_labels'] == -1)]
        
        if len(focal_outliers) > 0:
            print(f'  Protecting {len(focal_outliers)} events with focal mechanisms from outlier removal:')
            for idx, row in focal_outliers.iterrows():
                print(f'    - {row["ID"]}: focal plane A={row["A"]}, isolation score={row["isolation_score"]:.3f}')
            
            # Reassign these events to inliers (cluster 0)
            df_hyfi.loc[focal_mask & (df_hyfi['clust_labels'] == -1), 'clust_labels'] = 0
            print(f'    Reassigned focal mechanism events to inliers (cluster 0)')
    
    # Count outliers for reporting (after focal mechanism protection)
    num_outliers = len(df_hyfi[df_hyfi['clust_labels'] == -1])
    num_inliers = len(df_hyfi[df_hyfi['clust_labels'] == 0])
    
    print(f'  Final outliers (after focal mechanism protection): {num_outliers}')
    print(f'  Final outlier percentage: {100*num_outliers/len(df_hyfi):.1f}%')
    print(f'  Inliers: {num_inliers}')
    
    # Print statistics about isolation scores
    outlier_scores = df_hyfi[df_hyfi['clust_labels'] == -1]['isolation_score']
    inlier_scores = df_hyfi[df_hyfi['clust_labels'] == 0]['isolation_score']
    
    if len(outlier_scores) > 0:
        print(f'  Isolation score statistics:')
        print(f'    Outliers - mean: {outlier_scores.mean():.3f}, min: {outlier_scores.min():.3f}, max: {outlier_scores.max():.3f}')
        if len(inlier_scores) > 0:
            print(f'    Inliers  - mean: {inlier_scores.mean():.3f}, min: {inlier_scores.min():.3f}, max: {inlier_scores.max():.3f}')
    
    return df_hyfi


def faultnetwork3D(input_params):
    """
    Calculate 3D fault network from hypocenters.

    Parameters
    ----------
    hypo_file : str
        Path of hypoDD input file.
    hypo_sep : str
        Separator for hypoDD input file.
    out_dir : str
        Path for output folder.
    n_mc : int
        Number of MC simulations.
    r_nn : int
        Search radius for nearest neighbor search [m].
    dt_nn : int
        Search time for nearest neighbor search [h].
    validation_bool : bool
        If True: perform model validation calculations.
    foc_file : str
        Path for focal mechanism catalog.
    foc_sep : str
        Separator for focal mechanism catalog.
    stress_bool : bool
        If True: perform fault stress analysis.
    S1_mag : int
        Maximum principal stress magnitude [MPa].
    S2_mag : int
        Intermediate principal stress magnitude [MPa].
    S3_mag : int
        Minimum principal stress magnitude [MPa].
    PP : int
        Pore fluid pressure [MPa].
    S1_trend : int
        Trend of maximum principal stress direction.
    S1_plunge : int
        Plunge of maximum principal stress direction.
    S3_trend : int
        Trend of minimum principal stress direction.
    S3_plunge : int
        Plunge of minimum principal stress direction.
    stress_R : float
        Stress shape ratio R.
    fric_coeff : float
        Friction coefficient.
    autoclass_bool : bool
        If True: perform autoclassification.
    mag_type : str
        Type of magnitude (ML, Mw)

    Returns
    -------
    DataFrames with the parameters of the input parameters (with/without detected outliers), the full 3D fault
    network model and the MC hypocenter locations.
    """
    
    print('\n')
    print('='*50)
    print('FAULT NETWORK RECONSTRUCTION')
    print('='*50)
    
    # Unpack input parameters from dictionary
    r_nn = input_params.get('r_nn')
    dt_nn = input_params.get('dt_nn')
    n_mc = input_params.get('n_mc')
    mag_type = input_params.get('mag_type', 'ML')
    hypo_file = input_params.get('hypo_file')
    hypo_sep = input_params.get('hypo_sep', ',')
    out_dir = input_params.get('out_dir', '.')
    use_focal_constraints = input_params.get('use_focal_constraints', False)
    
    # Validate input file before loading
    from ..utils.input_validation import InputFileValidator
    
    validator = InputFileValidator()
    hypo_validation = validator.validate_hypocenter_file(hypo_file, hypo_sep)
    
    if not hypo_validation['valid']:
        error_msg = f"Hypocenter file validation failed: {hypo_validation.get('error', 'Unknown error')}"
        if hypo_validation.get('missing_columns'):
            error_msg += f"\nMissing columns: {', '.join(hypo_validation['missing_columns'])}"
        if hypo_validation.get('recommendations'):
            error_msg += f"\nRecommendations: {'; '.join(hypo_validation['recommendations'])}"
        raise ValueError(error_msg)
    
    print("✓ Hypocenter file format validation passed.")
    
    # Data import - Load into single dataframe that will be enriched throughout pipeline
    # Ensure ID is read as string to preserve format (e.g., 'KP200307161108')
    # Only load the first 24 columns (standard hypoDD format)
    df_hyfi = pd.read_csv(hypo_file, sep=hypo_sep, dtype={'ID': str}, usecols=range(24))
    
    # Extract the date and time information of the hypocenters
    df_hyfi['Date'] = pd.to_datetime(pd.DataFrame({'year': df_hyfi['YR'],
                            'month': df_hyfi['MO'],
                            'day': df_hyfi['DY'],
                            'hour': df_hyfi['HR'],
                            'minute': df_hyfi['MI'],
                            'second': df_hyfi['SC']}))
        
    # Handle missing error values by providing reasonable defaults
    # Default to 0m spatial error (no error) when not provided
    default_error = 0.0
    if 'EX' not in df_hyfi.columns or df_hyfi['EX'].isna().all():
        df_hyfi['EX'] = default_error
        print(f"Warning: EX column missing or empty, using default error of {default_error}m")
    else:
        df_hyfi['EX'] = df_hyfi['EX'].fillna(default_error)
    
    if 'EY' not in df_hyfi.columns or df_hyfi['EY'].isna().all():
        df_hyfi['EY'] = default_error
        print(f"Warning: EY column missing or empty, using default error of {default_error}m")
    else:
        df_hyfi['EY'] = df_hyfi['EY'].fillna(default_error)
    
    if 'EZ' not in df_hyfi.columns or df_hyfi['EZ'].isna().all():
        df_hyfi['EZ'] = default_error
        print(f"Warning: EZ column missing or empty, using default error of {default_error}m")
    else:
        df_hyfi['EZ'] = df_hyfi['EZ'].fillna(default_error)
    
    print(f"Successfully loaded {len(df_hyfi)} events from hypocenter catalog")
    
    ###########################################################################
    # Merge focal mechanism data early (if available)
    # This makes focal mechanism data available for use in fault network calculations
    if 'foc_file' in input_params and input_params.get('foc_file'):
        from .model_validation import merge_focal_mechanisms_early
        df_hyfi = merge_focal_mechanisms_early(df_hyfi, input_params)
    else:
        print("No focal mechanism file specified in input parameters")
        # Add empty focal mechanism columns for consistency
        focal_cols = ['Strike1', 'Dip1', 'Rake1', 'Strike2', 'Dip2', 'Rake2', 'A']
        for col in focal_cols:
            df_hyfi[col] = np.nan

    ###########################################################################
    # Outlier removal
    # Apply outlier detection algorithm if specified
    # Check if outlier detection is defined
    if 'remove_outliers' in input_params:
        remove_outliers = input_params['remove_outliers']
    else:
        remove_outliers = False
    
    # Get outlier detection method (default to DBSCAN for backward compatibility)
    outlier_method = input_params.get('outlier_method', 'DBSCAN')
    
    if remove_outliers:
        if outlier_method.upper() == 'DBSCAN':
            df_hyfi = DBSCAN_outlier_detection(df_hyfi)
        elif outlier_method.upper() == 'LOF':
            # Get LOF-specific parameters if provided
            lof_n_neighbors = input_params.get('lof_n_neighbors', None)
            lof_contamination = input_params.get('lof_contamination', 'auto')
            df_hyfi = LOF_outlier_detection(df_hyfi, 
                                           n_neighbors=lof_n_neighbors,
                                           contamination=lof_contamination)
        elif outlier_method.upper() == 'ISOLATIONFOREST' or outlier_method.upper() == 'IFOREST':
            # Get Isolation Forest-specific parameters if provided
            if_n_estimators = input_params.get('if_n_estimators', 100)
            if_contamination = input_params.get('if_contamination', 0.05)  # Default to 5% (conservative)
            if_max_samples = input_params.get('if_max_samples', 'auto')
            if_random_state = input_params.get('if_random_state', 42)
            df_hyfi = IsolationForest_outlier_detection(df_hyfi,
                                                        n_estimators=if_n_estimators,
                                                        contamination=if_contamination,
                                                        max_samples=if_max_samples,
                                                        random_state=if_random_state)
        else:
            print(f'Warning: Unknown outlier detection method "{outlier_method}". Using DBSCAN.')
            df_hyfi = DBSCAN_outlier_detection(df_hyfi)
    else:
        # Add default cluster labels (all events belong to cluster 0)
        df_hyfi['clust_labels'] = 0
        print('No outlier detection applied - all events assigned to cluster 0')

    ###########################################################################
    # MC Simulation preparation
    # Filter out outliers (clust_labels == -1) for plane fitting calculations
    # but keep them in the final output dataset
    if 'clust_labels' in df_hyfi.columns:
        non_outlier_mask = df_hyfi['clust_labels'] != -1
        data_for_plane_fitting = df_hyfi[non_outlier_mask].copy().reset_index(drop=True)
        # Store the IDs for mapping results back
        non_outlier_ids = data_for_plane_fitting['ID'].tolist()
        print(f"Excluding {len(df_hyfi[df_hyfi['clust_labels'] == -1])} outliers from plane fitting calculations")
        print(f"Using {len(data_for_plane_fitting)} events for plane fitting")
    else:
        data_for_plane_fitting = df_hyfi.copy().reset_index(drop=True)
        non_outlier_ids = data_for_plane_fitting['ID'].tolist()
    
    # Create n_mc perturbed hypocenter datasets for Monte Carlo simulation
    # Use filtered data (without outliers) for plane fitting
    per_X, per_Y, per_Z = hypo_perturbation(n_mc,
                                            np.array(data_for_plane_fitting['X']),
                                            np.array(data_for_plane_fitting['Y']),
                                            np.array(data_for_plane_fitting['Z']),
                                            np.array(data_for_plane_fitting['EX']),
                                            np.array(data_for_plane_fitting['EY']),
                                            np.array(data_for_plane_fitting['EZ']))

    # Create dataframes for per_X, per_Y, and per_Z
    df_per_X = pd.concat([data_for_plane_fitting['ID'], pd.DataFrame(per_X)], axis=1)
    df_per_Y = pd.concat([data_for_plane_fitting['ID'], pd.DataFrame(per_Y)], axis=1)
    df_per_Z = pd.concat([data_for_plane_fitting['ID'], pd.DataFrame(per_Z)], axis=1)

    ###########################################################################
    # Fault Plane Orientations for each hypocenter dataset

    # Create empty lists to store all possible plane orientations for each
    # event i
    plane_fit_list = []

    # Calculate the fault plane orientations for all perturbated hypocenter locations k
    # Convert IDs to numeric format (remove 'KP' prefix if present)
    def convert_id_to_numeric(id_val):
        """Convert ID to numeric format, handling string IDs like 'KP200307161108' or 'KP200307161108_1'"""
        if isinstance(id_val, str):
            # Remove common prefixes like 'KP' but preserve suffixes like '_1', '_2'
            # First remove non-alphanumeric characters except underscores
            cleaned = id_val.replace('KP', '').replace('kp', '')
            
            # Extract all digits (this will concatenate main ID and suffix)
            # E.g., 'KP200307161108_1' -> '200307161108' + '1' -> '2003071611081'
            numeric_part = ''.join(filter(str.isdigit, cleaned))
            
            if numeric_part:
                return int(numeric_part)
            else:
                # If no digits found, create a hash-based numeric ID
                return abs(hash(id_val)) % (10**12)  # Ensure it fits in reasonable range
        else:
            return int(id_val)
    
    # Convert IDs to simple sequential integers for plane fitting calculations
    # This avoids issues with complex ID formats while maintaining mapping
    data_for_plane_fitting = data_for_plane_fitting.reset_index(drop=True)
    data_for_plane_fitting['ID_sequential'] = range(len(data_for_plane_fitting))
    
    # Create mapping from sequential ID to original ID for focal mechanism lookup
    id_mapping = dict(zip(data_for_plane_fitting['ID_sequential'], data_for_plane_fitting['ID']))
    
    it = len(data_for_plane_fitting)
    ID = np.array(data_for_plane_fitting['ID_sequential'], dtype=np.int64)  # Use sequential IDs for calculations
    # Format date as ISO string that numpy datetime64 can parse
    date = np.array(data_for_plane_fitting['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S'))
    EX = np.array(data_for_plane_fitting['EX'])
    EY = np.array(data_for_plane_fitting['EY'])
    EZ = np.array(data_for_plane_fitting['EZ'])
    print(f"Using nearest neighbor search radius r_nn = {r_nn} m and time window dt_nn = ±{dt_nn} h")
    
    # Prepare focal mechanism data for enhanced PCA if enabled
    df_focal_indexed = None
    if use_focal_constraints and any(col in df_hyfi.columns for col in ['Strike1', 'Dip1', 'A']):
        # Create a focal mechanism dataframe indexed by ID for fast lookup
        focal_cols = ['X', 'Y', 'Z', 'Strike1', 'Dip1', 'Strike2', 'Dip2', 'A', 'MAG']
        available_cols = [col for col in focal_cols if col in df_hyfi.columns]
        if available_cols:
            df_focal_indexed = df_hyfi[available_cols].set_index(df_hyfi['ID'])
            print(f"Focal mechanism constraints enabled: {len(df_focal_indexed[df_focal_indexed['A'].notna()])} events with active plane info")
        else:
            print("Focal mechanism constraints requested but no focal data available")
    elif use_focal_constraints:
        print("Focal mechanism constraints requested but focal data not found in dataframe")
    
    for i in range(n_mc):
        # Extract XYZ locations for each perturbation k
        X = per_X[:, i]
        Y = per_Y[:, i]
        Z = per_Z[:, i]

        # Apply the fault plane calculation function for the respective XYZ
        # dataset and save the results to the list
        plane_fit = faultplanes3D(ID, date, X, Y, Z, EX, EY, EZ, r_nn, dt_nn, 
                                  df_focal=df_focal_indexed, use_focal_constraints=use_focal_constraints,
                                  id_mapping=id_mapping)
        plane_fit_list.append(plane_fit)

    ###########################################################################
    # Directional Statistic

    def dirstats(plane_fit_list, it, n_mc):
        # Extract the normal unit vectors and rat_lam23 values from all k perturbations for each
        # event i and save them in separate lists
        nor_x_i_list = [[] for _ in range(it)]
        nor_y_i_list = [[] for _ in range(it)]
        nor_z_i_list = [[] for _ in range(it)]
        rat_lam23_i_list = [[] for _ in range(it)]
        for k in range(n_mc):
            temp = plane_fit_list[k]
            for i in range(it):
                nor_x_i_list[i].append(temp[i, 1])
                nor_y_i_list[i].append(temp[i, 2])
                nor_z_i_list[i].append(temp[i, 3])
                rat_lam23_i_list[i].append(temp[i, 13])  # rat_lam23 is at position 13
                
        # Calculate the directional statistics parameters of the fault planes
        # from the perturbed hypocenter data
        dirstats_output = np.empty((it, 10))  # Increased to 10 to include rat_lam23
        dirstats_output[:] = np.nan
        for i in range(it):
            nor_x = np.array(nor_x_i_list[i])
            nor_y = np.array(nor_y_i_list[i])
            nor_z = np.array(nor_z_i_list[i])
            rat_lam23 = np.array(rat_lam23_i_list[i])
            # Nan for all if there are no fits
            if np.isnan(nor_x).all():
                nr_fits = np.nan
                mean_vector = np.array([np.nan, np.nan, np.nan])
                R = np.nan
                N = np.nan
                RN = np.nan
                kappa = np.nan
                beta = np.nan
                mean_rat_lam23 = np.nan
            # Nan for statistical parameters if there are less than 80% fits
            elif np.count_nonzero(~np.isnan(nor_x)) / n_mc < 0.8:
                nr_fits = np.count_nonzero(~np.isnan(nor_x)) / n_mc
                mean_vector = np.array([np.nan, np.nan, np.nan])
                R = np.nan
                N = np.nan
                RN = np.nan
                kappa = np.nan
                beta = np.nan
                mean_rat_lam23 = np.nanmean(rat_lam23)  # Calculate mean even with <80% fits
            # Directional statistics for all events with more than 80% fits
            else:
                # Calculate direction of the pole to the first plane
                if np.isnan(nor_x).all():
                    v1 = [np.nan, np.nan, np.nan]
                else:
                    v1 = np.array([nor_x[np.isfinite(nor_x)][0], nor_y[np.isfinite(nor_y)][0], nor_z[np.isfinite(nor_z)][0]])
                    v1 /= np.linalg.norm(v1)
                # Check every point in the dataset and swap direction if it
                # lies on the other side of the stereoplot
                # (angular difference larger than 90 degrees)
                for j in range(len(nor_x)):
                    vj = [nor_x[j], nor_y[j], nor_z[j]]
                    vj = vj / np.linalg.norm(vj)
                    if np.linalg.norm(v1 - vj) == 0:
                        angle_deg = np.nan
                    else:
                        angle_deg = np.degrees(np.arccos(np.dot(v1, vj)))
                    if angle_deg > 90:
                        nor_x[j] = nor_x[j] * -1
                        nor_y[j] = nor_y[j] * -1
                        nor_z[j] = nor_z[j] * -1
                
                # Calculate the number of fitted models
                nr_fits = np.count_nonzero(~np.isnan(nor_x)) / n_mc
                # Calculate R and R/N
                v_sum = np.nansum(np.array([nor_x, nor_y, nor_z]), axis=1)
                R = np.linalg.norm(v_sum)
                N = n_mc
                RN = R / N

                # Calculate FB5 (Kent) distribution parameters
                vectors = np.array([nor_x, nor_y, nor_z]).T
                vectors = vectors[~np.isnan(vectors).any(axis=1)]
                if n_mc > 1:
                    try:
                        G = kent_me(vectors)
                        mean_vector = G.gamma1 / np.linalg.norm(G.gamma1)
                        kappa = int(G.kappa)
                        beta = int(G.beta)
                    except AssertionError:
                        mean_vector = np.nanmean(vectors, axis=0)
                        kappa = -999
                        beta = -999
                elif n_mc == 1:
                    mean_vector = np.nanmean(vectors, axis=0)
                    kappa = -999
                    beta = -999
                
                # Calculate mean rat_lam23 for this event
                mean_rat_lam23 = np.nanmean(rat_lam23)
                
                # Check if mean lies on upper hemisphere and turn back to lower
                if mean_vector[2] > 0:
                    mean_vector = [-x for x in mean_vector]
                    
                # Loop through plane vectors and make all lower hemisphere again
                for j in range(len(nor_x)):
                    if nor_z[j] > 0:
                        nor_x[j] = -nor_x[j]
                        nor_y[j] = -nor_y[j]
                        nor_z[j] = -nor_z[j]
                        
            # Save the statistical parameters in an array
            dirstats_output[i, :3] = mean_vector
            dirstats_output[i, 3:9] = [nr_fits, R, N, RN, kappa, beta]
            dirstats_output[i, 9] = mean_rat_lam23  # Add rat_lam23 at position 9
    
        return(dirstats_output, nor_x_i_list, nor_y_i_list, nor_z_i_list)


    # Use the correct count for dirstats - the number of events that had plane fitting done
    plane_fitting_count = len(data_for_plane_fitting)
    dirstats_output, nor_x_i_list, nor_y_i_list, nor_z_i_list = dirstats(plane_fit_list, plane_fitting_count, n_mc)

    # Now add computed plane fitting results back to main dataframe
    # Initialize all plane fitting columns with NaN for all events
    plane_fitting_columns = ['nor_x_mean', 'nor_y_mean', 'nor_z_mean', 'nr_fits', 'R', 'N', 'R/N', 'kappa', 'beta', 'lambda_2_3']
    for col in plane_fitting_columns:
        df_hyfi[col] = np.nan
    
    # Fill in plane fitting results only for non-outlier events
    if 'clust_labels' in df_hyfi.columns:
        
        # Ensure we don't have more results than expected
        if len(non_outlier_ids) != dirstats_output.shape[0]:
            print(f"Warning: Mismatch between non-outlier IDs count ({len(non_outlier_ids)}) and dirstats results ({dirstats_output.shape[0]})")
        
        # Map dirstats results using event IDs
        max_results = min(len(non_outlier_ids), dirstats_output.shape[0])
        for i in range(max_results):
            event_id = non_outlier_ids[i]
            # Find the row in df_hyfi that matches this ID
            mask = df_hyfi['ID'] == event_id
            if mask.any():
                df_hyfi.loc[mask, 'nor_x_mean'] = dirstats_output[i, 0]
                df_hyfi.loc[mask, 'nor_y_mean'] = dirstats_output[i, 1]
                df_hyfi.loc[mask, 'nor_z_mean'] = dirstats_output[i, 2]
                df_hyfi.loc[mask, 'nr_fits'] = dirstats_output[i, 3]
                df_hyfi.loc[mask, 'R'] = dirstats_output[i, 4]
                df_hyfi.loc[mask, 'N'] = dirstats_output[i, 5]
                df_hyfi.loc[mask, 'R/N'] = dirstats_output[i, 6]
                df_hyfi.loc[mask, 'kappa'] = dirstats_output[i, 7]
                df_hyfi.loc[mask, 'beta'] = dirstats_output[i, 8]
                df_hyfi.loc[mask, 'lambda_2_3'] = dirstats_output[i, 9]
            else:
                print(f"Warning: Could not find event ID {event_id} in output dataframe")
    else:
        # If no clustering was performed, all events have plane fitting results
        df_hyfi['nor_x_mean'] = dirstats_output[:, 0]
        df_hyfi['nor_y_mean'] = dirstats_output[:, 1]
        df_hyfi['nor_z_mean'] = dirstats_output[:, 2]
        df_hyfi['nr_fits'] = dirstats_output[:, 3]
        df_hyfi['R'] = dirstats_output[:, 4]
        df_hyfi['N'] = dirstats_output[:, 5]
        df_hyfi['R/N'] = dirstats_output[:, 6]
        df_hyfi['kappa'] = dirstats_output[:, 7]
        df_hyfi['lambda_2_3'] = dirstats_output[:, 9]

    # Set back the negative (=very well-defined) kappa value to the largest value obtained
    # Only for non-NaN values
    non_nan_kappa = df_hyfi['kappa'].dropna()
    if len(non_nan_kappa) > 0:
        max_kappa = non_nan_kappa.max()
        df_hyfi['kappa'].replace(-999, max_kappa, inplace=True)
    
    # Convert the normal unit vector to azimuth and dip of the mean plane
    # Only for events that have plane fitting results (non-NaN values)
    nor_columns = ['nor_x_mean', 'nor_y_mean', 'nor_z_mean']
    azi_list, dip_list = [], []
    for nor_x, nor_y, nor_z in zip_longest(*(df_hyfi[col] for col in nor_columns)):        
        # Check if vector of event i is NaN; if yes skip event i, else
        # calculate azimuth and dip of the mean plane
        if all(map(np.isnan, [nor_x, nor_y, nor_z])):
            azi, dip = np.nan, np.nan
        else:
            azi, dip = utilities.plane_normal_to_azidip(nor_x, nor_y, nor_z)
        azi_list.append(azi)
        dip_list.append(dip)
    df_hyfi['rupt_plane_azi'] = azi_list
    df_hyfi['rupt_plane_dip'] = dip_list

    ###########################################################################
    # Magnitude - Fault Area Scaling

    # Transform ML to Mw (after Allmann et al. 2010)
    if mag_type == 'ML':
        df_hyfi['Mw'] = df_hyfi['MAG'].apply(ML_to_MW)
    elif mag_type == 'Mw':
        df_hyfi['Mw'] = df_hyfi['MAG']
    else:
        print('ERROR: no Magnitude type specified')

    # Sanity-check Mw values: physically plausible range is roughly -3 to 10.
    # Values outside this range (e.g. Mw=32 from a typo like 3.2 entered as 32)
    # produce nonsensical rupture radii that cause downstream OOM crashes.
    _MW_MIN, _MW_MAX = -3.0, 10.0
    _mw_bad = ~df_hyfi['Mw'].between(_MW_MIN, _MW_MAX) & df_hyfi['Mw'].notna()
    if _mw_bad.any():
        _bad_vals = df_hyfi.loc[_mw_bad, 'Mw'].values
        print(f"\n⚠ WARNING: {_mw_bad.sum()} event(s) have Mw outside plausible range "
              f"[{_MW_MIN}, {_MW_MAX}]: {_bad_vals}")
        print(f"  These events will have their Mw set to NaN to prevent OOM errors.")
        print(f"  Please check the input data for magnitude typos (e.g. 32 instead of 3.2).\n")
        df_hyfi.loc[_mw_bad, 'Mw'] = np.nan
    # Calculate the rupture area A (in km2) and the diameter r (in m) of a
    # circular rupture plane (after Leonard 2014)
    A, r = np.vectorize(faultscalingL14_Mag_A)(df_hyfi['Mw'])
    df_hyfi['rupt_area'] = A  # Changed from 'A' to avoid conflict with focal mechanism 'A' column
    df_hyfi['rupt_radius'] = r  # Changed from 'r' to 'rupt_radius' for clarity

    ###########################################################################

    # Print the number of fitted fault planes
    num_fitted_planes = df_hyfi['nr_fits'].notna().sum()
    print(f'Number of events with fitted fault planes: {num_fitted_planes} out of {len(df_hyfi)} total events')

    # Return single enriched dataframe along with MC perturbation data
    return df_hyfi, df_per_X, df_per_Y, df_per_Z
