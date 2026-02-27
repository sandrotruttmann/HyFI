#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS: Model Validation Module

Please cite: Truttmann et al. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps.

@author: Sandro Truttmann
@contact: sandro.truttmann@gmail.com
@license: GPL-3.0
@date: April 2023
@version: 0.1.1
"""

# Import modules
import csv
import numpy as np
import pandas as pd
from ..utils import utilities
from obspy.imaging.beachball import aux_plane


def merge_focal_mechanisms_early(df_hyfi, input_params):
    """
    Merge focal mechanism data with hypocenter data early in the pipeline.
    
    This function can be called from fault_network module to make focal mechanism
    data available for subsequent processing steps.
    
    Parameters
    ----------
    df_hyfi : DataFrame
        Hypocenter dataframe
    input_params : dict
        Input parameters containing focal mechanism file information
    
    Returns
    -------
    df_hyfi : DataFrame
        Dataframe with focal mechanism data merged in
    """
    
    # Check if focal mechanism parameters are available
    foc_file = input_params.get('foc_file')
    foc_sep = input_params.get('foc_sep', ';')
    foc_mag_check = input_params.get('foc_mag_check', True)
    foc_loc_check = input_params.get('foc_loc_check', True)
    max_mag_diff = input_params.get('foc_max_mag_diff', 0.2)
    max_distance_km = input_params.get('foc_max_dist_km', 1.0)
    
    if not foc_file:
        print("No focal mechanism file specified, continuing without focal mechanisms")
        # Add empty focal mechanism columns for consistency
        focal_cols = ['Strike1', 'Dip1', 'Rake1', 'Strike2', 'Dip2', 'Rake2', 'A']
        for col in focal_cols:
            df_hyfi[col] = np.nan
        return df_hyfi
    
    try:
        print("Merging focal mechanism data...")
        
        # Create temporary data_input from relevant columns of df_hyfi for compatibility
        data_input_cols = ['ID', 'X', 'Y', 'Z', 'Date', 'EX', 'EY', 'EZ']
        if 'MAG' in df_hyfi.columns:
            data_input_cols.append('MAG')
            
        temp_data_input = df_hyfi[data_input_cols].copy()
        
        # Add underscore columns temporarily for compatibility with legacy function
        temp_data_input['_X'] = temp_data_input['X']
        temp_data_input['_Y'] = temp_data_input['Y'] 
        temp_data_input['_Z'] = temp_data_input['Z']
        
        # Call the existing function
        matched_data_input = match_hypoDD_focals_proc(temp_data_input, foc_file, foc_sep, 
                                                       foc_mag_check, foc_loc_check,
                                                       max_mag_diff, max_distance_km)
        
        # Add the focal mechanism columns to df_hyfi (including 'A' for active plane)
        focal_cols = ['Strike1', 'Dip1', 'Rake1', 'Strike2', 'Dip2', 'Rake2', 'A']
        for col in focal_cols:
            if col in matched_data_input.columns:
                df_hyfi[col] = matched_data_input[col]
            else:
                df_hyfi[col] = np.nan
        
        # Count successful matches
        nr_match = df_hyfi['Strike1'].count()
        print(f"Successfully merged {nr_match} focal mechanisms with hypocenter data")
        
        return df_hyfi
        
    except Exception as e:
        print(f"Error merging focal mechanisms: {e}")
        print("Continuing without focal mechanism data...")
        # Add empty focal mechanism columns for consistency
        focal_cols = ['Strike1', 'Dip1', 'Rake1', 'Strike2', 'Dip2', 'Rake2', 'A']
        for col in focal_cols:
            df_hyfi[col] = np.nan
        return df_hyfi


def match_hypoDD_focals_proc(data_input, foc_file, foc_sep, foc_mag_check, foc_loc_check, 
                             max_mag_diff=0.2, max_distance_km=1.0):
    """
    Load and match hypoDD and focal data (using event time and magnitude).

    Parameters
    ----------
    data_input : DataFrame
        Input hypocenter dataframe with coordinates and metadata
    foc_file : str
        Path of focal mechanism file.
    foc_sep : str
        Separator in focal mechanism file.
    foc_mag_check : bool
        Whether to perform magnitude consistency check
    foc_loc_check : bool
        Whether to perform location consistency check
    max_mag_diff : float, default=0.5
        Maximum allowed magnitude difference for matching events (magnitude units)
    max_distance_km : float, default=5.0
        Maximum allowed distance for location consistency check (km for depth, 
        converted to degrees for lat/lon: ~0.045 degrees ≈ 5 km)

    Returns
    -------
    DataFrame with hypoDD events and matched focal mechanisms.

    """
    
    # Validate focal mechanism file before loading
    from ..utils.input_validation import InputFileValidator
    
    validator = InputFileValidator()
    focal_validation = validator.validate_focal_mechanism_file(foc_file, foc_sep)
    
    if not focal_validation['valid']:
        error_msg = f"Focal mechanism file validation failed: {focal_validation.get('error', 'Unknown error')}"
        if focal_validation.get('missing_columns'):
            error_msg += f"\nMissing columns: {', '.join(focal_validation['missing_columns'])}"
        if focal_validation.get('recommendations'):
            error_msg += f"\nRecommendations: {'; '.join(focal_validation['recommendations'])}"
        raise ValueError(error_msg)
    
    print("✓ Focal mechanism file format validation passed.")
    
    # Import the focal file (only first 28 columns - standard focal mechanism format)
    focal_import = pd.read_csv(foc_file, sep=foc_sep, usecols=range(28))

        
    # Handle different time formats based on validation results
    time_format = focal_validation.get('time_format', ['HR', 'MI', 'SC'])
    
    if time_format == ['Hr:Mi']:
        # Combined time format - use original logic
        # Delete the last columns of the focals - but be more flexible
        if focal_import.shape[1] > 27:  # If more than expected columns
            focal_import = focal_import.iloc[:, 0:27]  # Keep first 27 columns
        
        # Check if focal_import['Hr:Mi'] incorporates seconds
        if len(focal_import['Hr:Mi'][0]) >= 8:
            datestring = pd.Series(focal_import['Hr:Mi'].str.split('.', expand=True).astype(str).values.T[0])
            Hr, Mi, Sec = datestring.str.split(':', expand=True).astype(int).values.T
            df_date = pd.DataFrame({'year': focal_import['Yr'],
                                    'month': focal_import['Mo'],
                                    'day': focal_import['Dy'],
                                    'hour': Hr,
                                    'minute': Mi,
                                    'second': Sec})
        elif len(focal_import['Hr:Mi'][0]) == 5:
            Hr, Mi = focal_import['Hr:Mi'].str.split(':', expand=True).astype(int).values.T
            df_date = pd.DataFrame({'year': focal_import['Yr'],
                                    'month': focal_import['Mo'],
                                    'day': focal_import['Dy'],
                                    'hour': Hr,
                                    'minute': Mi})
        else:
            raise ValueError('Please check the time format of the focal file!')
    else:
        # Separate time columns (HR, MI, SC or Hr, Mi, Sc)
        # Delete extra columns but keep enough for the expected structure
        if focal_import.shape[1] > 28:  # More flexible column handling
            focal_import = focal_import.iloc[:, 0:28]
        
        # Handle different capitalizations
        hr_col = 'HR' if 'HR' in focal_import.columns else 'Hr'
        mi_col = 'MI' if 'MI' in focal_import.columns else 'Mi'
        sc_col = 'SC' if 'SC' in focal_import.columns else 'Sc'
        
        df_date = pd.DataFrame({'year': focal_import['YR'],
                                'month': focal_import['MO'],
                                'day': focal_import['DY'],
                                'hour': focal_import[hr_col],
                                'minute': focal_import[mi_col],
                                'second': focal_import[sc_col] if sc_col in focal_import.columns else 0})
    
    focal_import['Date'] = pd.to_datetime(df_date)
    focal_import = focal_import.sort_values(by='Date')

   # First, try to merge on ID if both datasets have matching IDs
    if 'ID' in data_input.columns and 'ID' in focal_import.columns:
        # Ensure both ID columns have the same data type for merging
        data_input_copy = data_input.copy()
        focal_import_copy = focal_import.copy()
        data_input_copy['ID'] = data_input_copy['ID'].astype(str)
        focal_import_copy['ID'] = focal_import_copy['ID'].astype(str)
        
        df = pd.merge(data_input_copy, focal_import_copy, on='ID', how='left', suffixes=('', '_fm'))

        # Check if we got any matches by counting non-null focal mechanism values
        focal_cols_check = ['Strike1', 'Dip1', 'Rake1']
        matched_count = 0
        for col in focal_cols_check:
            if col in df.columns:
                matched_count += (~df[col].isna()).sum()

        if matched_count == 0:
            print("No matches found with Focal ID merge, falling back to time-based merge...")
            # Fall back to time-based merge
            df = pd.merge_asof(data_input.sort_values('Date'), focal_import.sort_values('Date'), 
                             on='Date', tolerance=pd.Timedelta('60s'), suffixes=('', '_fm'))
        else:
            print(f"Successfully merged focal mechanisms by ID (total non-null values: {matched_count})")
    else:
        print("Merging on Date (temporal matching)...")
        df = pd.merge_asof(data_input, focal_import, on='Date',
                         tolerance=pd.Timedelta('60s'), suffixes=('', '_fm'))
        
    # Optional: magnitude cross-check
    if foc_mag_check:
        # Check if required magnitude columns exist
        # After merge, hypocenter MAG should be 'MAG' and focal mechanism MAG should be 'MAG_fm'
        hypo_mag_col = None
        focal_mag_col = None
        
        # Find hypocenter magnitude column
        if 'MAG' in df.columns:
            hypo_mag_col = 'MAG'
        elif 'Mag' in df.columns:
            hypo_mag_col = 'Mag'
            
        # Find focal mechanism magnitude column
        if 'MAG_fm' in df.columns:
            focal_mag_col = 'MAG_fm'
        elif 'Mag_fm' in df.columns:
            focal_mag_col = 'Mag_fm'
        
        if hypo_mag_col and focal_mag_col:
            # Check if magnitudes of merged data fits
            # Calculate magnitude difference
            mag_diff = np.abs(df[hypo_mag_col] - df[focal_mag_col])
            # Find indexes with missfitting magnitudes (using configurable threshold)
            error_idx = np.where(mag_diff > max_mag_diff)[0]
            # Delete the focal data for the respective missfited hypocenter datasets
            if len(error_idx) > 0:
                print(f"Found {len(error_idx)} events with magnitude misfit > {max_mag_diff}, setting focal data to NaN")
                focal_cols = ['Strike1', 'Dip1', 'Rake1', 'Strike2', 'Dip2', 'Rake2', 'Pazim', 'Pdip', 'Tazim', 'Tdip']
                for col in focal_cols:
                    if col in df.columns:
                        df.loc[error_idx, col] = np.nan
        else:
            print("Warning: Cannot perform magnitude cross-check - required magnitude columns not found")
            print(f"Looking for: hypocenter mag ({hypo_mag_col}), focal mag ({focal_mag_col})")
            print(f"Available magnitude columns: {[col for col in df.columns if 'MAG' in col.upper() or 'Mag' in col]}")

    # Optional: Location cross-check
    if foc_loc_check:
        # Check if required location columns exist
        required_cols = ['LAT', 'LON', 'DEPTH']
        focal_location_cols = []
        
        # Find corresponding focal mechanism location columns
        for col in required_cols:
            if col in df.columns:
                # Look for focal mechanism equivalent (might be 'Lat', 'Lon', 'Z_y', etc.)
                focal_col = None
                if col == 'LAT' and 'Lat' in df.columns:
                    focal_col = 'Lat'
                elif col == 'LON' and 'Lon' in df.columns:
                    focal_col = 'Lon'
                elif col == 'DEPTH' and 'Z_y' in df.columns:
                    focal_col = 'Z_y'
                elif col == 'DEPTH' and 'DEPTH' in focal_import.columns:
                    focal_col = 'DEPTH'
                
                if focal_col:
                    focal_location_cols.append((col, focal_col))
        
        if focal_location_cols:
            # Check if lat & lon & depth are fitting (using configurable thresholds)
            # Convert km to degrees for lat/lon: ~1 km ≈ 0.009 degrees at mid-latitudes
            degrees_per_km = 0.009
            for i, row in df.iterrows():
                for hypo_col, focal_col in focal_location_cols:
                    try:
                        if hypo_col == 'DEPTH':
                            # Use km threshold for depth
                            threshold = max_distance_km
                        else:
                            # Convert km to degrees for lat/lon
                            threshold = max_distance_km * degrees_per_km
                        
                        if not pd.isna(row[hypo_col]) and not pd.isna(row[focal_col]):
                            if abs(row[hypo_col] - row[focal_col]) > threshold:
                                print(f"Please check: Missfit in {hypo_col} for event with ID {row['ID']} "
                                      f"(diff: {abs(row[hypo_col] - row[focal_col]):.3f}, threshold: {threshold:.3f})")
                    except (KeyError, TypeError) as e:
                        print(f"Warning: Cannot check location fit for {hypo_col}: {e}")
        else:
            print("Warning: Cannot perform location cross-check - required location columns not found")


    return(df)


def focal_validation(df_hyfi, input_params):
    """
    Validate the fault network model with focal mechanism data using single dataframe approach.

    Parameters
    ----------
    df_hyfi : DataFrame
        Single dataframe containing hypocenter data and computed fault plane parameters
    input_params : dict
        Input parameters including validation settings and focal mechanism file info

    Returns
    -------
    df_hyfi : DataFrame
        Input dataframe with added focal mechanism validation columns
    """
    
    # Extract validation parameters
    validation_bool = input_params.get('validation_bool', False)
    
    if not validation_bool:
        # No validation requested - initialize columns for compatibility
        df_hyfi['epsilon'] = np.nan
        df_hyfi['pref_foc'] = np.nan
        focal_cols = ['Strike1', 'Dip1', 'Rake1', 'Strike2', 'Dip2', 'Rake2']
        for col in focal_cols:
            if col not in df_hyfi.columns:
                df_hyfi[col] = np.nan
        return df_hyfi
        
    print('\n')
    print('='*50)
    print('FAULT NETWORK VALIDATION')
    print('='*50)

    try:
        # Extract focal mechanism parameters
        foc_file = input_params.get('foc_file')
        foc_sep = input_params.get('foc_sep', ';')
        foc_mag_check = input_params.get('foc_mag_check', True)
        foc_loc_check = input_params.get('foc_loc_check', True)
        max_mag_diff = input_params.get('foc_max_mag_diff', 0.2)
        max_distance_km = input_params.get('foc_max_dist_km', 1.0)
        
        if not foc_file:
            print("Warning: No focal mechanism file specified, skipping validation")
            df_hyfi['epsilon'] = np.nan
            df_hyfi['pref_foc'] = np.nan
            return df_hyfi
        
        # Check if focal mechanisms are already merged (from fault network module)
        focal_cols_to_check = ['Strike1', 'Dip1', 'Rake1']
        already_merged = all(col in df_hyfi.columns for col in focal_cols_to_check)
        
        if already_merged:
            # Check if we have any actual focal mechanism data (not just NaN columns)
            has_focal_data = any(df_hyfi[col].notna().any() for col in focal_cols_to_check)
            
            if has_focal_data:
                print("✓ Focal mechanisms already merged in fault network module, proceeding with validation...")
                nr_existing_match = df_hyfi['Strike1'].count()
                print(f"Found {nr_existing_match} existing focal mechanism matches")
            else:
                print("Focal mechanism columns exist but contain no data, attempting to merge...")
                # Match hypocenter data with focal mechanisms
                df_hyfi = match_hypoDD_focals(df_hyfi, foc_file, foc_sep, foc_mag_check, foc_loc_check,
                                              max_mag_diff, max_distance_km)
        else:
            print("Focal mechanisms not yet merged, merging now...")
            # Match hypocenter data with focal mechanisms
            df_hyfi = match_hypoDD_focals(df_hyfi, foc_file, foc_sep, foc_mag_check, foc_loc_check,
                                          max_mag_diff, max_distance_km)

        # Calculate auxiliary plane from Strike1, Dip1, Rake1
        for k in range(len(df_hyfi)):
            try:
                strike1 = df_hyfi.loc[k, 'Strike1']
                dip1 = df_hyfi.loc[k, 'Dip1']
                rake1 = df_hyfi.loc[k, 'Rake1']
                
                # Check if focal mechanism parameters are valid
                if pd.isna(strike1) or pd.isna(dip1) or pd.isna(rake1):
                    df_hyfi.loc[k, ['Strike2', 'Dip2', 'Rake2']] = np.nan
                    continue
                
                # Check parameter ranges
                if not (0 <= strike1 <= 360 and 0 <= dip1 <= 90 and -180 <= rake1 <= 180):
                    print(f"Warning: Invalid focal mechanism parameters for event {df_hyfi.loc[k, 'ID']}: "
                        f"Strike1={strike1}, Dip1={dip1}, Rake1={rake1}")
                    df_hyfi.loc[k, ['Strike2', 'Dip2', 'Rake2']] = np.nan
                    continue
                
                auxiliary_plane = aux_plane(strike1, dip1, rake1)
                
                if np.isnan(auxiliary_plane[0]):
                    df_hyfi.loc[k, ['Strike2', 'Dip2', 'Rake2']] = np.nan
                else:
                    df_hyfi.loc[k, ['Strike2', 'Dip2', 'Rake2']] = [int(auxiliary_plane[0]), int(auxiliary_plane[1]), int(auxiliary_plane[2])]
                    
            except Exception as e:
                print(f"Error calculating auxiliary plane for event {df_hyfi.loc[k, 'ID']}: {e}")
                df_hyfi.loc[k, ['Strike2', 'Dip2', 'Rake2']] = np.nan

        # Initialize validation columns for all events
        df_hyfi['epsilon'] = np.nan
        df_hyfi['pref_foc'] = np.nan
    
        # Calculate the angular difference between fault plane orientations and focal mechanisms
        for i in range(len(df_hyfi)):
            try:
                # Skip if no fault plane was calculated
                if pd.isnull(df_hyfi['rupt_plane_azi'][i]):
                    continue
                    
                # Check if focal mechanism data is available
                if (pd.isna(df_hyfi['Strike1'][i]) or pd.isna(df_hyfi['Dip1'][i]) or 
                    pd.isna(df_hyfi['Strike2'][i]) or pd.isna(df_hyfi['Dip2'][i])):
                    continue
                
                # Calculate the normal unit vector of the computed fault plane
                nor_fau = [df_hyfi['nor_x_mean'][i],
                          df_hyfi['nor_y_mean'][i],
                          df_hyfi['nor_z_mean'][i]]
                
                # Calculate the normal unit vectors of the two focal plane solutions
                # Strike to dip azimuth conversion: dip direction is 90° clockwise from strike
                azi1 = (df_hyfi['Strike1'][i] + 90) % 360  # Convert strike to dip azimuth
                dip1 = df_hyfi['Dip1'][i]
                azi2 = (df_hyfi['Strike2'][i] + 90) % 360  # Convert strike to dip azimuth
                dip2 = df_hyfi['Dip2'][i]
                
                nor_x1, nor_y1, nor_z1 = utilities.plane_azidip_to_normal(azi1, dip1)
                nor_x2, nor_y2, nor_z2 = utilities.plane_azidip_to_normal(azi2, dip2)
                nor_foc1 = [nor_x1, nor_y1, nor_z1]
                nor_foc2 = [nor_x2, nor_y2, nor_z2]
    
                # Calculate the angle between the computed fault plane and the focal planes
                angle1 = utilities.angle_between(nor_fau, nor_foc1)
                angle2 = utilities.angle_between(nor_fau, nor_foc2)
                angle1 = angle1 if angle1 < 90 else 180 - angle1
                angle2 = angle2 if angle2 < 90 else 180 - angle2    
                
                # PRIORITIZE the active plane indicator "A" if available
                if 'A' in df_hyfi.columns and not pd.isnull(df_hyfi['A'][i]):
                    active_plane = int(df_hyfi['A'][i])
                    if active_plane in [1, 2]:
                        # Use the specified active plane
                        df_hyfi.loc[i, 'pref_foc'] = active_plane
                        # Store the angular difference for the active plane
                        df_hyfi.loc[i, 'epsilon'] = angle1 if active_plane == 1 else angle2
                    else:
                        # A=0 or invalid A value: use geometric selection (minimum angular difference)
                        angle12 = [angle1, angle2]
                        df_hyfi.loc[i, 'epsilon'] = min(angle12)
                        df_hyfi.loc[i, 'pref_foc'] = angle12.index(min(angle12)) + 1
                else:
                    # No active plane specified, choose based on minimal angular difference
                    angle12 = [angle1, angle2]
                    df_hyfi.loc[i, 'epsilon'] = min(angle12)
                    if not pd.isnull(df_hyfi['Strike1'][i]):
                        df_hyfi.loc[i, 'pref_foc'] = angle12.index(min(angle12)) + 1
                        
            except Exception as e:
                print(f"Error calculating angular differences for event {i}: {e}")
                df_hyfi.loc[i, 'epsilon'] = np.nan
                df_hyfi.loc[i, 'pref_foc'] = np.nan
    
        # Count the number of matched focals
        nr_match = df_hyfi['Strike1'].count()
        print('Number of matched focal mechanisms: ', nr_match)
        
        # Report on active plane usage and automatically determined planes
        has_A_column = 'A' in df_hyfi.columns
        
        if has_A_column:
            nr_with_active_plane = df_hyfi['A'].notna().sum()
            if nr_with_active_plane > 0:
                print(f'Number of events with active plane indicator (A): {nr_with_active_plane}')
                print('✓ Active plane information WILL BE USED for preferred plane selection')
            else:
                print('No active plane indicators found - using geometric selection only')
        else:
            print('No active plane column found - using geometric selection only')
        
        # Count preferred planes by determination method
        nr_pref_determined = df_hyfi['pref_foc'].notna().sum()
        
        if nr_pref_determined > 0:
            print('\n--- PREFERRED FOCAL PLANE DETERMINATION ---')
            
            # Separate events by how plane was determined
            if has_A_column:
                # Events with A=1 or A=2 (pre-specified)
                prespecified_mask = df_hyfi['A'].isin([1, 2]) & df_hyfi['pref_foc'].notna()
                nr_prespecified = prespecified_mask.sum()
                
                # Events with A=0 (unknown, newly determined)
                A0_determined_mask = (df_hyfi['A'] == 0) & df_hyfi['pref_foc'].notna()
                nr_A0_determined = A0_determined_mask.sum()
                
                # Events with no A or NaN A (newly determined)
                noA_determined_mask = df_hyfi['A'].isna() & df_hyfi['pref_foc'].notna()
                nr_noA_determined = noA_determined_mask.sum()
                
                # Events with A=1/2 but no pref_foc (no rupture plane)
                A_but_no_plane_mask = df_hyfi['A'].isin([1, 2]) & df_hyfi['pref_foc'].isna()
                nr_A_no_plane = A_but_no_plane_mask.sum()
                
                nr_newly_total = nr_A0_determined + nr_noA_determined
                
                print(f'  Pre-specified active planes (A=1 or A=2): {nr_prespecified}')
                print(f'  Newly determined preferred planes (geometric selection):')
                print(f'    - A=0 (unknown plane): {nr_A0_determined}')
                if nr_noA_determined > 0:
                    print(f'    - No A value: {nr_noA_determined}')
                print(f'    - Total newly determined: {nr_newly_total}')
                
                if nr_A_no_plane > 0:
                    print(f'\n  ⚠️  {nr_A_no_plane} events with A=1/2 have no pref_foc (no computed rupture plane)')
                
                if nr_newly_total > 0:
                    print(f'\n  ℹ️  {nr_newly_total} active planes were newly determined by geometric selection')
                    print('     (based on minimum angular difference to computed fault plane)')
                    
                    # Provide statistics on the newly determined planes
                    newly_determined = df_hyfi[A0_determined_mask | noA_determined_mask]
                    nr_plane1 = (newly_determined['pref_foc'] == 1).sum()
                    nr_plane2 = (newly_determined['pref_foc'] == 2).sum()
                    print(f'     - Plane 1 selected: {nr_plane1} events')
                    print(f'     - Plane 2 selected: {nr_plane2} events')
                    
                    if len(newly_determined) > 0:
                        mean_epsilon = newly_determined['epsilon'].mean()
                        print(f'     - Mean angular difference: {mean_epsilon:.1f}°')
            else:
                # No A column, all are newly determined
                print(f'  All {nr_pref_determined} preferred planes were newly determined (geometric selection)')
                nr_plane1 = (df_hyfi['pref_foc'] == 1).sum()
                nr_plane2 = (df_hyfi['pref_foc'] == 2).sum()
                print(f'     - Plane 1 selected: {nr_plane1} events')
                print(f'     - Plane 2 selected: {nr_plane2} events')
                
                if nr_pref_determined > 0:
                    mean_epsilon = df_hyfi[df_hyfi['pref_foc'].notna()]['epsilon'].mean()
                    print(f'     - Mean angular difference: {mean_epsilon:.1f}°')
        
        # Export detailed active plane determination summary to file
        try:
            if 'out_dir' in input_params and nr_pref_determined > 0:
                export_active_plane_summary(df_hyfi, input_params['out_dir'])
        except Exception as e:
            print(f"Warning: Could not export active plane summary: {e}")

        
    except Exception as e:
        print(f"Error during focal mechanism validation: {e}")
        print("Continuing without focal mechanism validation...")
        # Set default values for validation columns
        df_hyfi['epsilon'] = np.nan
        df_hyfi['pref_foc'] = np.nan
        # Add focal mechanism columns as NaN if they don't exist
        focal_cols = ['Strike1', 'Dip1', 'Rake1', 'Strike2', 'Dip2', 'Rake2']
        for col in focal_cols:
            if col not in df_hyfi.columns:
                df_hyfi[col] = np.nan

    return df_hyfi


def export_active_plane_summary(df_hyfi, out_dir):
    """
    Export a detailed summary of active plane determinations to a CSV file.
    
    This documents which events had pre-specified active planes vs. auto-determined ones,
    and also lists events where no preferred plane could be determined.
    
    Parameters
    ----------
    df_hyfi : DataFrame
        Main dataframe with focal mechanism and validation data
    out_dir : str
        Output directory path
    """
    import os
    
    # Filter to events with focal mechanisms (Strike1 is not NaN)
    has_focal_mask = df_hyfi['Strike1'].notna()
    focal_events_all = df_hyfi[has_focal_mask].copy()
    
    if len(focal_events_all) == 0:
        return
    
    # Determine determination method for each event
    has_A_column = 'A' in focal_events_all.columns
    
    def get_determination_method(row):
        has_pref_foc = pd.notna(row['pref_foc'])
        has_rupture_plane = pd.notna(row.get('rupt_plane_azi', np.nan))
        
        if has_pref_foc:
            # pref_foc is determined
            if has_A_column and pd.notna(row['A']):
                if row['A'] in [1, 2]:
                    # A=1 or A=2: Pre-specified active plane
                    return 'Pre-specified (A=1 or A=2)'
                elif row['A'] == 0:
                    # A=0 with pref_foc: Unknown plane, newly determined by geometry
                    return 'Newly determined (A=0, geometric selection)'
                else:
                    # Invalid A value but pref_foc exists
                    return 'Newly determined (invalid A, geometric selection)'
            else:
                # No A column, geometric selection
                return 'Newly determined (no A, geometric selection)'
        else:
            # No pref_foc determined
            if has_A_column and pd.notna(row['A']) and row['A'] in [1, 2]:
                # A=1 or A=2 but pref_foc is NaN: no rupture plane was calculated
                return 'Not determined (A specified but no rupture plane)'
            elif not has_rupture_plane:
                # No rupture plane computed
                return 'Not determined (no computed rupture plane)'
            else:
                # Other reason
                return 'Not determined (other reason)'
    
    focal_events_all['plane_determination_method'] = focal_events_all.apply(
        get_determination_method, axis=1
    )
    
    # Create summary dataframe with relevant columns
    summary_cols = ['ID', 'Date', 'X', 'Y', 'Z', 'MAG',
                    'Strike1', 'Dip1', 'Rake1', 
                    'Strike2', 'Dip2', 'Rake2',
                    'pref_foc', 'epsilon', 'plane_determination_method']
    
    if has_A_column:
        summary_cols.insert(summary_cols.index('pref_foc'), 'A')
    
    # Add mean_azi and mean_dip to show computed fault plane orientation
    if 'rupt_plane_azi' in focal_events_all.columns:
        summary_cols.append('rupt_plane_azi')
    if 'rupt_plane_dip' in focal_events_all.columns:
        summary_cols.append('rupt_plane_dip')
    
    # Only include columns that exist
    summary_cols = [col for col in summary_cols if col in focal_events_all.columns]
    
    summary_df = focal_events_all[summary_cols].copy()
    
    # Add interpretation columns (only for events with pref_foc)
    summary_df['preferred_strike'] = summary_df.apply(
        lambda row: row['Strike1'] if row['pref_foc'] == 1 
                   else (row['Strike2'] if row['pref_foc'] == 2 else np.nan),
        axis=1
    )
    summary_df['preferred_dip'] = summary_df.apply(
        lambda row: row['Dip1'] if row['pref_foc'] == 1 
                   else (row['Dip2'] if row['pref_foc'] == 2 else np.nan),
        axis=1
    )
    summary_df['preferred_rake'] = summary_df.apply(
        lambda row: row['Rake1'] if row['pref_foc'] == 1 
                   else (row['Rake2'] if row['pref_foc'] == 2 else np.nan),
        axis=1
    )
    
    # Sort by determination method, then by epsilon (best matches first)
    # Put "Not determined" events at the end
    sort_order = {
        'Pre-specified (A=1 or A=2)': 0,
        'Newly determined (A=0, geometric selection)': 1,
        'Newly determined (no A, geometric selection)': 2,
        'Newly determined (invalid A, geometric selection)': 3,
        'Not determined (A specified but no rupture plane)': 4,
        'Not determined (no computed rupture plane)': 5,
        'Not determined (other reason)': 6
    }
    summary_df['_sort_order'] = summary_df['plane_determination_method'].map(sort_order)
    summary_df = summary_df.sort_values(['_sort_order', 'epsilon'])
    summary_df = summary_df.drop(columns=['_sort_order'])
    summary_df['epsilon'] = summary_df['epsilon'].round(1)
    
    # Export to CSV — use ISO 8601 'T' separator so datetime values are never
    # split into two columns by space-delimited viewers.
    output_file = os.path.join(out_dir, 'active_plane_determination_summary.csv')
    summary_df.to_csv(output_file, index=False, date_format='%Y-%m-%dT%H:%M:%S.%f',
                       quoting=csv.QUOTE_NONNUMERIC)
    
    print(f'\n  📄 Active plane determination summary exported to: {output_file}')
    
    # Also create a brief statistics file
    stats_file = os.path.join(out_dir, 'active_plane_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write('='*60 + '\n')
        f.write('ACTIVE PLANE DETERMINATION STATISTICS\n')
        f.write('='*60 + '\n\n')
        
        f.write(f'Total events with focal mechanisms: {len(summary_df)}\n\n')
        
        # Count events by determination method
        method_counts = summary_df['plane_determination_method'].value_counts()
        
        # New categories
        nr_prespecified = method_counts.get('Pre-specified (A=1 or A=2)', 0)
        nr_newly_A0 = method_counts.get('Newly determined (A=0, geometric selection)', 0)
        nr_newly_noA = method_counts.get('Newly determined (no A, geometric selection)', 0)
        nr_newly_invalid = method_counts.get('Newly determined (invalid A, geometric selection)', 0)
        nr_newly_total = nr_newly_A0 + nr_newly_noA + nr_newly_invalid
        
        nr_not_A_no_plane = method_counts.get('Not determined (A specified but no rupture plane)', 0)
        nr_not_determined_no_plane = method_counts.get('Not determined (fault plane)', 0)
        nr_not_determined_other = method_counts.get('Not determined (other reason)', 0)
        nr_not_determined_total = nr_not_A_no_plane + nr_not_determined_no_plane + nr_not_determined_other
        
        f.write('DETERMINATION METHOD BREAKDOWN:\n')
        f.write(f'  Pre-specified active planes (A=1 or A=2): {nr_prespecified}\n')
        f.write(f'  Newly determined preferred planes (geometric selection):\n')
        f.write(f'    - A=0 (unknown plane): {nr_newly_A0}\n')
        if nr_newly_noA > 0:
            f.write(f'    - No A column: {nr_newly_noA}\n')
        if nr_newly_invalid > 0:
            f.write(f'    - Invalid A value: {nr_newly_invalid}\n')
        f.write(f'    - Total newly determined: {nr_newly_total}\n')
        f.write(f'  Not determined:\n')
        if nr_not_A_no_plane > 0:
            f.write(f'    - A specified but no rupture plane: {nr_not_A_no_plane}\n')
        f.write(f'    - No computed rupture plane: {nr_not_determined_no_plane}\n')
        if nr_not_determined_other > 0:
            f.write(f'    - Other reason: {nr_not_determined_other}\n')
        f.write(f'    - Total NOT determined: {nr_not_determined_total}\n\n')
        
        # Statistics for newly determined planes (all geometric selection methods combined)
        if nr_newly_total > 0:
            newly_df = summary_df[summary_df['plane_determination_method'].str.contains('Newly determined')]
            f.write('Newly determined plane selection breakdown:\n')
            f.write(f'  - Plane 1 selected: {(newly_df["pref_foc"] == 1).sum()} events\n')
            f.write(f'  - Plane 2 selected: {(newly_df["pref_foc"] == 2).sum()} events\n')
            f.write(f'  - Mean angular difference: {newly_df["epsilon"].mean():.1f}°\n')
            f.write(f'  - Median angular difference: {newly_df["epsilon"].median():.1f}°\n')
            f.write(f'  - Best match: {newly_df["epsilon"].min():.1f}°\n')
            f.write(f'  - Worst match: {newly_df["epsilon"].max():.1f}°\n\n')
        
        # Statistics for pre-specified planes
        if nr_prespecified > 0:
            prespec_df = summary_df[summary_df['plane_determination_method'] == 'Pre-specified (A=1 or A=2)']
            f.write('Pre-specified plane angular differences:\n')
            f.write(f'  - Mean angular difference: {prespec_df["epsilon"].mean():.1f}°\n')
            f.write(f'  - Median angular difference: {prespec_df["epsilon"].median():.1f}°\n\n')
        
        # Events where no preferred plane could be determined
        if nr_not_determined_total > 0:
            f.write('Events where NO preferred plane could be determined:\n')
            if nr_not_A_no_plane > 0:
                f.write(f'  - A specified (A=1 or A=2) but no rupture plane: {nr_not_A_no_plane} events\n')
                f.write(f'    (Focal mechanism available, but fault plane fitting failed)\n')
            if nr_not_determined_no_plane > 0:
                f.write(f'  - No computed rupture plane available: {nr_not_determined_no_plane} events\n')
                f.write(f'    (Insufficient neighbors for plane fitting)\n')
            if nr_not_determined_other > 0:
                f.write(f'  - Other reasons: {nr_not_determined_other} events\n')
            f.write(f'\n  ⚠️  Review these {nr_not_determined_total} events in the CSV file for details\n\n')
        
        f.write('='*60 + '\n')
        f.write('See active_plane_determination_summary.csv for full details\n')
        f.write('='*60 + '\n')
    
    print(f'  📊 Statistics summary exported to: {stats_file}')
    
    # Print warning if there are events without determined planes
    if nr_not_determined_total > 0:
        print(f'\n  ⚠️  Warning: {nr_not_determined_total} events with focal mechanisms have NO determined preferred plane')
        print(f'     See statistics file for breakdown by reason')


def match_hypoDD_focals(df_hyfi, foc_file, foc_sep, foc_mag_check, foc_loc_check,
                        max_mag_diff=0.5, max_distance_km=5.0):
    """
    Match hypocenter data with focal mechanism data for single dataframe approach.
    
    This is a single dataframe version of match_hypoDD_focals that adds focal mechanism
    data directly to df_hyfi instead of returning separate dataframes.
    
    Parameters
    ----------
    df_hyfi : DataFrame
        Main hypocenter dataframe
    foc_file : str
        Path to focal mechanism file
    foc_sep : str
        Separator for focal mechanism file
    foc_mag_check : bool
        Whether to perform magnitude consistency check
    foc_loc_check : bool
        Whether to perform location consistency check
    max_mag_diff : float, default=0.5
        Maximum allowed magnitude difference for matching events
    max_distance_km : float, default=5.0
        Maximum allowed distance for location consistency check (km)
    
    Returns
    -------
    df_hyfi : DataFrame
        Input dataframe with added focal mechanism columns
    """
    try:
        # Use the existing match_hypoDD_focals function but adapt for single dataframe
        # Create temporary data_input from relevant columns of df_hyfi
        data_input_cols = ['ID', 'X', 'Y', 'Z', 'Date', 'EX', 'EY', 'EZ']
        if 'MAG' in df_hyfi.columns:
            data_input_cols.append('MAG')
            
        temp_data_input = df_hyfi[data_input_cols].copy()
        
        # Add underscore columns temporarily for compatibility with legacy function
        temp_data_input['_X'] = temp_data_input['X']
        temp_data_input['_Y'] = temp_data_input['Y'] 
        temp_data_input['_Z'] = temp_data_input['Z']
        
        # Call the existing function with all parameters
        matched_data_input = match_hypoDD_focals_proc(temp_data_input, foc_file, foc_sep, 
                                                       foc_mag_check, foc_loc_check,
                                                       max_mag_diff, max_distance_km)
        
        # Add the focal mechanism columns back to df_hyfi (including 'A' for active plane)
        focal_cols = ['Strike1', 'Dip1', 'Rake1', 'Strike2', 'Dip2', 'Rake2', 'A']
        for col in focal_cols:
            if col in matched_data_input.columns:
                df_hyfi[col] = matched_data_input[col]
            else:
                df_hyfi[col] = np.nan
                
        return df_hyfi
        
    except Exception as e:
        print(f"Error matching focal mechanisms: {e}")
        # Add empty focal mechanism columns (including 'A' for active plane)
        focal_cols = ['Strike1', 'Dip1', 'Rake1', 'Strike2', 'Dip2', 'Rake2', 'A']
        for col in focal_cols:
            df_hyfi[col] = np.nan
        return df_hyfi