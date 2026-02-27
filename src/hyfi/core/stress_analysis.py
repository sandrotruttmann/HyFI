#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS: Fault Stress Analysis Module

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
import mplstereonet
from ..utils import utilities
import geopandas as gpd


def load_stress_field_from_shapefile(shapefile_path):
    """
    Load stress field polygons from a shapefile.
    
    Parameters
    ----------
    shapefile_path : str
        Path to the shapefile containing stress field polygons.
        Expected columns: s1_trend, s1_plunge, s3_trend, s3_plunge, R (stress ratio)
        
    Returns
    -------
    gdf : GeoDataFrame
        GeoDataFrame containing the stress field polygons with their attributes.
    """
    
    try:
        gdf = gpd.read_file(shapefile_path)
        
        # Validate required columns
        required_columns = ['s1_trend', 's1_plunge', 's3_trend', 's3_plunge', 'R']
        missing_columns = [col for col in required_columns if col not in gdf.columns]
        
        if missing_columns:
            raise ValueError(
                f"Shapefile is missing required columns: {missing_columns}. "
                f"Expected columns: {required_columns}"
            )
        
        print(f"Loaded stress field shapefile with {len(gdf)} polygons")
        return gdf
        
    except Exception as e:
        raise RuntimeError(f"Failed to load stress field shapefile: {e}")


def get_stress_field_for_point(x, y, stress_gdf, source_crs=None):
    """
    Query stress field parameters for a given point coordinate.
    
    Parameters
    ----------
    x : float
        X coordinate in source_crs
    y : float
        Y coordinate in source_crs
    stress_gdf : GeoDataFrame
        GeoDataFrame containing stress field polygons
    source_crs : str, optional
        CRS of the input coordinates (e.g. 'EPSG:21781'). If the shapefile
        has a different CRS the point is reprojected automatically.
        
    Returns
    -------
    dict or None
        Dictionary with stress field parameters (S1_trend, S1_plunge, S3_trend, S3_plunge, stress_R)
        or None if point is not within any polygon
    """
    try:
        from shapely.geometry import Point
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "shapely and geopandas are required for spatial queries."
        )
    
    # Create point geometry
    point = Point(x, y)

    # Reproject query point to shapefile CRS if CRS info is available and differs
    if source_crs and stress_gdf.crs is not None:
        try:
            point_gdf = gpd.GeoDataFrame(geometry=[point], crs=source_crs)
            point_gdf = point_gdf.to_crs(stress_gdf.crs)
            point = point_gdf.geometry.iloc[0]
            print(f"  Reprojected query point from {source_crs} to {stress_gdf.crs}")
        except Exception as _e:
            print(f"  Warning: Could not reproject point ({_e}), using raw coordinates.")

    # Find which polygon contains this point
    mask = stress_gdf.contains(point)
    
    if mask.any():
        # Get the first matching polygon
        row = stress_gdf[mask].iloc[0]
        result = {
            'S1_trend': row['s1_trend'],
            'S1_plunge': row['s1_plunge'],
            'S3_trend': row['s3_trend'],
            'S3_plunge': row['s3_plunge'],
            'stress_R': row['R']
        }
        # Add domain name if available
        if 'Domain' in row.index:
            result['domain'] = row['Domain']
        return result
    else:
        return None


def stress_on_plane_I(S1_mag, S2_mag, S3_mag,
                    S1_trend, S1_plunge,
                    S3_trend, S3_plunge,
                    strike, dip,
                    PP, fric_coeff):
    """
    Calculate the stresses on a plane.

    Parameters
    ----------
    S1_mag : int
        Magnitude of maximum principal stress [MPa].
    S2_mag : int
        Magnitude of intermediate principal stress [MPa].
    S3_mag : int
        Magnitude of minimum principal stress [MPa].
    S1_trend : int
        Trend of maximum principal stress.
    S1_plunge : int
        Plunge of maximum principal stress.
    S2_trend : int
        Trend of intermediate principal stress.
    S2_plunge : int
        Plunge of intermediate principal stress.
    S3_trend : int
        Trend of minimum principal stress.
    S3_plunge : int
        Plunge of minimum principal stress.
    strike : int
        Strike of the plane.
    dip : int
        Dip of the plane.
    PP : int
        Pore fluid pressure [MPa].
    fric_coeff : float
        Friction coefficient.

    Returns
    -------
    Stresses on the plane: effective normal stress, shear stress, rake,
    slip tendency, dilation tendency, trend and plunge of S2.

    """
    # Vectorize S1 and S3
    lon, lat = mplstereonet.line(S1_plunge, S1_trend)
    S1_vec = np.asarray(mplstereonet.stereonet2xyz(lon, lat))[:, 0]
    lon, lat = mplstereonet.line(S3_plunge, S3_trend)
    S3_vec = np.asarray(mplstereonet.stereonet2xyz(lon, lat))[:, 0]

    # Calculate S2 (vectorized and trend/plunge)
    S2_vec = np.cross(S1_vec, S3_vec).round(10)
    S2_plunge, S2_trend = mplstereonet.vector2plunge_bearing(S2_vec[0],
                                                             S2_vec[1],
                                                             S2_vec[2])
    S2_trend = S2_trend[0].round(10)
    S2_plunge = S2_plunge[0].round(10)

    # Define the principal stress coordinate system
    PS = np.array([[S1_mag, 0, 0],
                   [0, S2_mag, 0],
                   [0, 0, S3_mag]
                   ])

    # Convert principal stresses from geographic to xyz orientations
    lon, lat = mplstereonet.line(S1_plunge, S1_trend)
    S1_vec = np.asarray(mplstereonet.stereonet2xyz(lon, lat))[:, 0]
    lon, lat = mplstereonet.line(S2_plunge, S2_trend)
    S2_vec = np.asarray(mplstereonet.stereonet2xyz(lon, lat))[:, 0]
    lon, lat = mplstereonet.line(S3_plunge, S3_trend)
    S3_vec = np.asarray(mplstereonet.stereonet2xyz(lon, lat))[:, 0]

    # Invert z-axis (to make both coordinate systems right-handed)
    S1_vec[2] *= -1
    S2_vec[2] *= -1
    S3_vec[2] *= -1
    
    # Construct the transformation matrix A to convert from principal
    # stress coordinates to geographical coordinate system
    A = np.array([[S1_vec[1], S1_vec[0], S1_vec[2]],
                  [S2_vec[1], S2_vec[0], S2_vec[2]],
                  [S3_vec[1], S3_vec[0], S3_vec[2]]
                  ])
    A = A.round(10)

    # Calculate the stress tensor within the geographic coordinate system and
    # round to 10 decimal numbers
    SG = (A.T @ PS @ A)

    # Define the fault plane coordinate system from fault strike and dip
    strike = np.radians(strike)
    dip = np.radians(dip)
    # Plane normal vector
    nn = np.array([-np.sin(strike) * np.sin(dip),
                   np.cos(strike) * np.sin(dip),
                   -np.cos(dip)
                   ])
    # Plane strike vector
    ns = np.array([np.cos(strike),
                   np.sin(strike),
                   0
                   ])
    # Plane dip vector
    nd = np.array([-np.sin(strike) * np.cos(dip),
                   np.cos(strike) * np.cos(dip),
                   np.sin(dip)
                   ])

    # Project the stress tensor (geographical coordinate system) onto the
    # normal vector of the fault
    t = (SG @ nn)

    # Calculate the normal and shear stresses on the fault plane
    # Total normal stress
    Sn_tot = np.dot(t, nn)
    # Effective normal stress
    Sn_eff = (Sn_tot - PP)
    # Absolute shear stress
    Tau_d = np.dot(t, nd)
    Tau_s = np.dot(t, ns)
    Tau = np.sqrt(Tau_d**2 + Tau_s**2)

    # Calculate the rake (direction of expected fault movement within the fault
    # plane)
    # Negative rake means normal movement, positive rake means reverse movement
    rake = np.arctan2(Tau_d, -Tau_s)
    rake = np.degrees(rake).round(10)
    
    ##############################################################################
    
    # Calculate fault instability (Vavrycuk et al. (2014))
    I = (Tau - fric_coeff*(Sn_eff - 1)) / (fric_coeff + np.sqrt(1 + fric_coeff**2))
    
    ##############################################################################
    
    
    return(Sn_eff, Tau, rake, I, S2_trend, S2_plunge)

def stress_on_plane_slipdilatend(
                    S1_trend, S1_plunge,
                    S3_trend, S3_plunge,
                    strike, dip,
                    PP, fric_coeff, stress_R):
    """
    Calculate the stresses on a plane.

    Parameters
    ----------
    S1_trend : int
        Trend of maximum principal stress.
    S1_plunge : int
        Plunge of maximum principal stress.
    S2_trend : int
        Trend of intermediate principal stress.
    S2_plunge : int
        Plunge of intermediate principal stress.
    S3_trend : int
        Trend of minimum principal stress.
    S3_plunge : int
        Plunge of minimum principal stress.
    strike : int
        Strike of the plane.
    dip : int
        Dip of the plane.
    PP : int
        Pore fluid pressure [MPa].
    fric_coeff : float
        Friction coefficient.

    Returns
    -------
    Stresses on the plane: effective normal stress, shear stress, rake,
    slip tendency, dilation tendency, trend and plunge of S2.

    """
    
    S1_mag, S2_mag, S3_mag, PP = utilities.reduced_stress_tens(fric_coeff, stress_R)
    
    # Vectorize S1 and S3
    lon, lat = mplstereonet.line(S1_plunge, S1_trend)
    S1_vec = np.asarray(mplstereonet.stereonet2xyz(lon, lat))[:, 0]
    lon, lat = mplstereonet.line(S3_plunge, S3_trend)
    S3_vec = np.asarray(mplstereonet.stereonet2xyz(lon, lat))[:, 0]

    # Calculate S2 (vectorized and trend/plunge)
    S2_vec = np.cross(S1_vec, S3_vec).round(10)
    S2_plunge, S2_trend = mplstereonet.vector2plunge_bearing(S2_vec[0],
                                                             S2_vec[1],
                                                             S2_vec[2])
    S2_trend = S2_trend[0].round(10)
    S2_plunge = S2_plunge[0].round(10)

    # Define the principal stress coordinate system
    PS = np.array([[S1_mag, 0, 0],
                   [0, S2_mag, 0],
                   [0, 0, S3_mag]
                   ])

    # Convert principal stresses from geographic to xyz orientations
    lon, lat = mplstereonet.line(S1_plunge, S1_trend)
    S1_vec = np.asarray(mplstereonet.stereonet2xyz(lon, lat))[:, 0]
    lon, lat = mplstereonet.line(S2_plunge, S2_trend)
    S2_vec = np.asarray(mplstereonet.stereonet2xyz(lon, lat))[:, 0]
    lon, lat = mplstereonet.line(S3_plunge, S3_trend)
    S3_vec = np.asarray(mplstereonet.stereonet2xyz(lon, lat))[:, 0]

    # Invert z-axis (to make both coordinate systems right-handed)
    S1_vec[2] *= -1
    S2_vec[2] *= -1
    S3_vec[2] *= -1
    
    # Construct the transformation matrix A to convert from principal
    # stress coordinates to geographical coordinate system
    A = np.array([[S1_vec[1], S1_vec[0], S1_vec[2]],
                  [S2_vec[1], S2_vec[0], S2_vec[2]],
                  [S3_vec[1], S3_vec[0], S3_vec[2]]
                  ])
    A = A.round(10)

    # Calculate the stress tensor within the geographic coordinate system and
    # round to 10 decimal numbers
    SG = (A.T @ PS @ A)

    # Define the fault plane coordinate system from fault strike and dip
    strike = np.radians(strike)
    dip = np.radians(dip)
    # Plane normal vector
    nn = np.array([-np.sin(strike) * np.sin(dip),
                   np.cos(strike) * np.sin(dip),
                   -np.cos(dip)
                   ])
    # Plane strike vector
    ns = np.array([np.cos(strike),
                   np.sin(strike),
                   0
                   ])
    # Plane dip vector
    nd = np.array([-np.sin(strike) * np.cos(dip),
                   np.cos(strike) * np.cos(dip),
                   np.sin(dip)
                   ])

    # Project the stress tensor (geographical coordinate system) onto the
    # normal vector of the fault
    t = (SG @ nn)

    # Calculate the normal and shear stresses on the fault plane
    # Total normal stress
    Sn_tot = np.dot(t, nn)
    # Effective normal stress
    Sn_eff = (Sn_tot - PP)
    # Absolute shear stress
    Tau_d = np.dot(t, nd)
    Tau_s = np.dot(t, ns)
    Tau = np.sqrt(Tau_d**2 + Tau_s**2)

    # Calculate the rake (direction of expected fault movement within the fault
    # plane)
    # Negative rake means normal movement, positive rake means reverse movement
    rake = np.arctan2(Tau_d, -Tau_s)
    rake = np.degrees(rake).round(10)
    
    ##############################################################################
    
    # Calculate the slip tendency (Morris et al. (1996))
    sliptend = (Tau / Sn_eff).round(10)

    # Calculate the dilation tendency (Ferrill et al. (1999))
    dilatend = ((S1_mag - Sn_eff) / (S1_mag - S3_mag)).round(10)
    
    ##############################################################################
    
    
    return(Sn_eff, Tau, rake, sliptend, dilatend, S2_trend, S2_plunge)


def fault_stress(df_hyfi, input_params):
    """
    Calculate the stress parameters for each individual earthquake.

    Parameters
    ----------
    df_hyfi : DataFrame
        Single dataframe containing fault plane data and results.
    input_params : dict
        Input parameters for stress analysis.

    Returns
    -------
    df_hyfi : DataFrame
        Input dataframe with added stress analysis columns.
    S2_trend : float
        Trend of intermediate principal stress.
    S2_plunge : float
        Plunge of intermediate principal stress.
    """
    
    # Unpack input parameters from dictionary
    stress_bool = input_params.get('stress_bool', True)
    PP = input_params.get('PP', 0)
    fric_coeff = input_params.get('fric_coeff', 0.75)
    
    # Check if stress field shapefile should be used
    use_shapefile_stress = input_params.get('use_shapefile_stress', False)
    stress_field_shapefile = input_params.get('stress_field_shapefile', None)
    
    if stress_bool:
       
        print('\n')
        print('='*50)
        print('FAULT STRESS ANALYSIS')
        print('='*50)
        
        # Initialize with fixed parameters (used as defaults/fallback)
        S1_trend = input_params.get('S1_trend', np.nan)
        S1_plunge = input_params.get('S1_plunge', np.nan)
        S3_trend = input_params.get('S3_trend', np.nan)
        S3_plunge = input_params.get('S3_plunge', np.nan)
        stress_R = input_params.get('stress_R', np.nan)
        
        # Load stress field from shapefile if enabled and provided
        use_shapefile = False
        
        if use_shapefile_stress and stress_field_shapefile is not None:
            try:
                stress_gdf = load_stress_field_from_shapefile(stress_field_shapefile)
                use_shapefile = True
                print(f"Using spatially-varying stress field from: {stress_field_shapefile}")
                
                # Calculate center coordinates of all hypocenters
                center_x = df_hyfi['X'].mean()
                center_y = df_hyfi['Y'].mean()
                print(f"Hypocenter center coordinates: X={center_x:.1f}m, Y={center_y:.1f}m")
                
                # Query stress field for the center point
                # Pass source CRS so the point can be reprojected to the shapefile CRS if needed
                source_crs = input_params.get('coordinate_system', 'EPSG:21781')
                stress_params = get_stress_field_for_point(center_x, center_y, stress_gdf,
                                                           source_crs=source_crs)

                if stress_params is None:
                    print(f"  ⚠️  WARNING: Center point (X={center_x:.1f}, Y={center_y:.1f}) not within any stress field polygon.")
                    print(f"  ⚠️  Source CRS: {source_crs} | Shapefile CRS: {stress_gdf.crs}")
                    print("  ⚠️  Falling back to fixed stress field parameters from config.")
                    use_shapefile = False
                else:
                    S1_trend = stress_params['S1_trend']
                    S1_plunge = stress_params['S1_plunge']
                    S3_trend = stress_params['S3_trend']
                    S3_plunge = stress_params['S3_plunge']
                    stress_R = stress_params['stress_R']
                    domain_name = stress_params.get('domain', 'Unknown')
                    print(f"Stress field parameters from shapefile (Domain: {domain_name}):")
                    print(f"  σ1: trend={S1_trend}°, plunge={S1_plunge}°")
                    print(f"  σ3: trend={S3_trend}°, plunge={S3_plunge}°")
                    print(f"  R={stress_R}")
                    
            except Exception as e:
                print(f"\n  ⚠️  WARNING: Failed to load stress field from shapefile: {e}")
                print(f"  ⚠️  Shapefile path used: {stress_field_shapefile}")
                print("  ⚠️  Falling back to fixed stress field parameters from config.")
                use_shapefile = False
        
        # Display stress parameters being used
        if not use_shapefile:
            # Check if fixed parameters are available (check for None or NaN)
            if any(param is None or (isinstance(param, float) and np.isnan(param)) 
                   for param in [S1_trend, S1_plunge, S3_trend, S3_plunge, stress_R]):
                raise ValueError(
                    "Stress field parameters are not defined. Either:\n"
                    "  1. Enable shapefile: set use_shapefile=true and provide valid shapefile_path, OR\n"
                    "  2. Provide fixed parameters: sigma1_trend_degrees, sigma1_plunge_degrees, "
                    "sigma3_trend_degrees, sigma3_plunge_degrees, stress_shape_ratio"
                )
            print(f"Using fixed stress field parameters from configuration:")
            print(f"  σ1: trend={S1_trend}°, plunge={S1_plunge}°")
            print(f"  σ3: trend={S3_trend}°, plunge={S3_plunge}°")
            print(f"  R={stress_R}")
        
        # Define relative stress magnitudes after Vavrycuk et al. (2014)
        S1_mag = 1
        S2_mag = 1 - (2*stress_R)
        S3_mag = -1
            
        # Calculate the stresses on each fault plane
        strike_arr = np.array((df_hyfi['rupt_plane_azi'] - 90) % 360)
        dip_arr = np.array(df_hyfi['rupt_plane_dip'])
        Sn_eff_list = []
        Tau_list = []
        rake_list = []
        I_list = []
        sliptend_list = []
        dilatend_list = []
        
        for i in range(len(df_hyfi)):
            # Skip events without fault plane data
            if pd.isna(df_hyfi.loc[i, 'rupt_plane_azi']) or pd.isna(df_hyfi.loc[i, 'rupt_plane_dip']):
                Sn_eff_list.append(np.nan)
                Tau_list.append(np.nan)
                rake_list.append(np.nan)
                I_list.append(np.nan)
                sliptend_list.append(np.nan)
                dilatend_list.append(np.nan)
                continue
                
            strike = strike_arr[i]
            dip = dip_arr[i]
            
            try:
                Sn_eff, Tau, rake, I, S2_trend, S2_plunge = stress_on_plane_I(
                                                                        S1_mag,
                                                                        S2_mag,
                                                                        S3_mag,
                                                                        S1_trend,
                                                                        S1_plunge,
                                                                        S3_trend,
                                                                        S3_plunge,
                                                                        strike, dip,
                                                                        PP, fric_coeff)
                Sn_eff_list.append(Sn_eff)
                Tau_list.append(Tau)
                rake_list.append(rake)
                I_list.append(I)
                
                Sn_eff, Tau, rake, sliptend, dilatend, S2_trend, S2_plunge = stress_on_plane_slipdilatend(
                                                            S1_trend,
                                                            S1_plunge,
                                                            S3_trend,
                                                            S3_plunge,
                                                            strike, dip,
                                                            PP, fric_coeff, stress_R)
                sliptend_list.append(sliptend)
                dilatend_list.append(dilatend)
                
            except Exception as e:
                print(f"Error calculating stress for event {i}: {e}")
                Sn_eff_list.append(np.nan)
                Tau_list.append(np.nan)
                rake_list.append(np.nan)
                I_list.append(np.nan)
                sliptend_list.append(np.nan)
                dilatend_list.append(np.nan)
                
        # Add stress analysis columns to the dataframe
        df_hyfi['Sn_eff'] = Sn_eff_list
        df_hyfi['Tau'] = Tau_list
        df_hyfi['rake'] = rake_list
        df_hyfi['instab'] = I_list
        df_hyfi['sliptend'] = sliptend_list
        df_hyfi['dilatend'] = dilatend_list
        
        print(f"Stress analysis completed for {len([x for x in Sn_eff_list if not pd.isna(x)])} fault planes")
        
    else:
        S2_trend = np.nan
        S2_plunge = np.nan
        # Initialize columns with NaN if stress analysis is disabled
        stress_columns = ['Sn_eff', 'Tau', 'rake', 'instab', 'sliptend', 'dilatend']
        for col in stress_columns:
            df_hyfi[col] = np.nan

    return df_hyfi, S2_trend, S2_plunge


def calculate_mesh_stress(mesh, stress_params):
    """
    Calculate stress parameters for each face of an interpolated mesh.
    
    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh with faces (triangles) representing fault surface
    stress_params : dict
        Stress field parameters with keys:
        - 'S1_trend', 'S1_plunge': σ₁ orientation
        - 'S3_trend', 'S3_plunge': σ₃ orientation  
        - 'stress_R': stress shape ratio
        - 'PP': pore pressure
        - 'fric_coeff': friction coefficient
        
    Returns
    -------
    mesh : pyvista.PolyData
        Input mesh with added cell data arrays for stress parameters
    """
    try:
        import pyvista as pv
    except ImportError:
        print("Warning: pyvista not available, cannot calculate mesh stress")
        return mesh
    
    # Extract stress parameters
    S1_trend = stress_params.get('S1_trend')
    S1_plunge = stress_params.get('S1_plunge')
    S3_trend = stress_params.get('S3_trend')
    S3_plunge = stress_params.get('S3_plunge')
    stress_R = stress_params.get('stress_R')
    PP = stress_params.get('PP', 0.0)
    fric_coeff = stress_params.get('fric_coeff', 0.75)
    
    # Check if stress parameters are available
    if any(param is None or (isinstance(param, float) and np.isnan(param)) 
           for param in [S1_trend, S1_plunge, S3_trend, S3_plunge, stress_R]):
        print("Warning: Stress parameters not available for mesh stress calculation")
        return mesh
    
    # Calculate face normals if not present
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False)
    
    # Get face normals
    face_normals = mesh.cell_data['Normals']
    n_faces = len(face_normals)
    
    # Define relative stress magnitudes
    S1_mag = 1
    S2_mag = 1 - (2*stress_R)
    S3_mag = -1
    
    # Initialize arrays for stress parameters
    Sn_eff_array = np.zeros(n_faces)
    Tau_array = np.zeros(n_faces)
    rake_array = np.zeros(n_faces)
    I_array = np.zeros(n_faces)
    sliptend_array = np.zeros(n_faces)
    dilatend_array = np.zeros(n_faces)
    
    # Calculate stress for each face
    for i in range(n_faces):
        # Get normal vector
        normal = face_normals[i]
        
        # Calculate strike and dip from normal vector
        # Normal vector points perpendicular to the plane
        # We need to convert this to strike and dip
        nx, ny, nz = normal[0], normal[1], normal[2]
        
        # Ensure normal points upward (positive Z component)
        if nz < 0:
            nx, ny, nz = -nx, -ny, -nz
        
        # Calculate dip (angle from horizontal)
        dip = np.degrees(np.arccos(nz))
        
        # Calculate strike (azimuth of dip direction minus 90°)
        # Dip direction azimuth
        dip_dir = np.degrees(np.arctan2(nx, ny)) % 360
        # Strike is 90° to the left of dip direction
        strike = (dip_dir - 90) % 360
        
        try:
            # Calculate stress parameters using existing functions
            Sn_eff, Tau, rake, I, S2_trend, S2_plunge = stress_on_plane_I(
                S1_mag, S2_mag, S3_mag,
                S1_trend, S1_plunge,
                S3_trend, S3_plunge,
                strike, dip,
                PP, fric_coeff
            )
            
            _, _, _, sliptend, dilatend, _, _ = stress_on_plane_slipdilatend(
                S1_trend, S1_plunge,
                S3_trend, S3_plunge,
                strike, dip,
                PP, fric_coeff, stress_R
            )
            
            Sn_eff_array[i] = Sn_eff
            Tau_array[i] = Tau
            rake_array[i] = rake
            I_array[i] = I
            sliptend_array[i] = sliptend
            dilatend_array[i] = dilatend
            
        except Exception as e:
            # If calculation fails, set to NaN
            Sn_eff_array[i] = np.nan
            Tau_array[i] = np.nan
            rake_array[i] = np.nan
            I_array[i] = np.nan
            sliptend_array[i] = np.nan
            dilatend_array[i] = np.nan
    
    # Add stress parameters as cell data
    mesh.cell_data['Sn_eff'] = Sn_eff_array
    mesh.cell_data['Tau'] = Tau_array
    mesh.cell_data['rake'] = rake_array
    mesh.cell_data['instab'] = I_array
    mesh.cell_data['sliptend'] = sliptend_array
    mesh.cell_data['dilatend'] = dilatend_array
    
    valid_count = np.sum(~np.isnan(Sn_eff_array))
    print(f"✓ Calculated stress for {valid_count}/{n_faces} mesh faces")
    
    return mesh
