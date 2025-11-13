#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input File Validation Module for HyFI

This module provides utilities to validate input files (hypocenter and focal mechanism data)
to ensure they have the correct columns and format before processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class InputFileValidator:
    """
    Validator class for HyFI input files (hypocenter and focal mechanism data).
    """
    
    # Expected column names for hypocenter data (hypoDD format)
    REQUIRED_HYPO_COLUMNS = [
        'ID', 'LAT', 'LON', 'DEPTH', 'X', 'Y', 'Z', 'EX', 'EY', 'EZ',
        'YR', 'MO', 'DY', 'HR', 'MI', 'SC', 'MAG', 'NCCP', 'NCCS', 
        'NCTP', 'NCTS', 'RCC', 'RCT', 'CID'
    ]
    
    # Expected column names for focal mechanism data
    REQUIRED_FOCAL_COLUMNS = [
        'ID', 'LAT', 'LON', 'DEPTH', 'X', 'Y', 'Z', 'YR', 'MO', 'DY', 
        'HR', 'MI', 'SC', 'MAG', 'A', 'Strike1', 'Dip1', 'Rake1', 
        'Strike2', 'Dip2', 'Rake2', 'Pazim', 'Pdip', 'Tazim', 'Tdip', 
        'Q', 'Type', 'Loc'
    ]
    
    # Alternative time formats for focal mechanisms
    FOCAL_TIME_ALTERNATIVES = [
        ['HR', 'MI', 'SC'],  # Separate columns
        ['Hr:Mi'],           # Combined format with optional seconds
        ['Hr', 'Mi', 'Sc'],  # Alternative capitalization
    ]
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_hypocenter_file(self, file_path: str, separator: str = '\t') -> Dict:
        """
        Validate hypocenter data file format and columns.
        
        Parameters
        ----------
        file_path : str
            Path to the hypocenter data file
        separator : str, default '\t'
            Column separator in the file
            
        Returns
        -------
        dict
            Validation results with status, missing columns, and recommendations
        """
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                return {
                    'valid': False,
                    'error': f"File not found: {file_path}",
                    'missing_columns': [],
                    'extra_columns': [],
                    'recommendations': ['Check file path']
                }
            
            # Try to read the file
            try:
                # Read with ID column as string to preserve format
                dtype_dict = {'ID': str} if 'ID' in self.REQUIRED_HYPO_COLUMNS else {}
                df = pd.read_csv(file_path, sep=separator, nrows=5, dtype=dtype_dict)  # Read only first 5 rows for validation
            except Exception as e:
                return {
                    'valid': False,
                    'error': f"Cannot read file: {e}",
                    'missing_columns': [],
                    'extra_columns': [],
                    'recommendations': ['Check file format and separator']
                }
            
            # Check columns
            file_columns = list(df.columns)
            missing_columns = [col for col in self.REQUIRED_HYPO_COLUMNS if col not in file_columns]
            extra_columns = [col for col in file_columns if col not in self.REQUIRED_HYPO_COLUMNS]
            
            # Validate data types and ranges
            data_issues = self._validate_hypocenter_data(df)
            
            # Determine if valid
            is_valid = len(missing_columns) == 0
            
            recommendations = []
            if missing_columns:
                recommendations.append(f"Add missing columns: {', '.join(missing_columns)}")
            if extra_columns:
                recommendations.append(f"Extra columns found (will be ignored): {', '.join(extra_columns)}")
            if data_issues:
                recommendations.extend(data_issues)
            
            return {
                'valid': is_valid,
                'file_path': str(file_path),
                'columns_found': file_columns,
                'missing_columns': missing_columns,
                'extra_columns': extra_columns,
                'data_issues': data_issues,
                'recommendations': recommendations,
                'sample_data': df.head(3).to_dict('records') if is_valid else None
            }
            
        except Exception as e:
            logger.error(f"Unexpected error validating hypocenter file: {e}")
            return {
                'valid': False,
                'error': f"Unexpected error: {e}",
                'missing_columns': [],
                'extra_columns': [],
                'recommendations': ['Check file and try again']
            }
    
    def validate_focal_mechanism_file(self, file_path: str, separator: str = ';') -> Dict:
        """
        Validate focal mechanism data file format and columns.
        
        Parameters
        ----------
        file_path : str
            Path to the focal mechanism data file
        separator : str, default ';'
            Column separator in the file
            
        Returns
        -------
        dict
            Validation results with status, missing columns, and recommendations
        """
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                return {
                    'valid': False,
                    'error': f"File not found: {file_path}",
                    'missing_columns': [],
                    'extra_columns': [],
                    'recommendations': ['Check file path']
                }
            
            # Try to read the file
            try:
                # Read with ID column as string to preserve format
                dtype_dict = {'ID': str} if 'ID' in self.REQUIRED_FOCAL_COLUMNS else {}
                df = pd.read_csv(file_path, sep=separator, nrows=5, dtype=dtype_dict)
            except Exception as e:
                return {
                    'valid': False,
                    'error': f"Cannot read file: {e}",
                    'missing_columns': [],
                    'extra_columns': [],
                    'recommendations': ['Check file format and separator']
                }
            
            # Check columns - be flexible with time format
            file_columns = list(df.columns)
            
            # First check for the core required columns (excluding time variations)
            core_columns = [col for col in self.REQUIRED_FOCAL_COLUMNS 
                           if col not in ['HR', 'MI', 'SC']]
            
            missing_core = [col for col in core_columns if col not in file_columns]
            
            # Check time format variations
            time_format_found = None
            for time_format in self.FOCAL_TIME_ALTERNATIVES:
                if all(col in file_columns for col in time_format):
                    time_format_found = time_format
                    break
            
            # Calculate missing columns
            missing_columns = missing_core.copy()
            if time_format_found is None:
                missing_columns.append("Time columns (HR,MI,SC or Hr:Mi)")
            
            # Find extra columns (beyond the first 28 expected columns or so)
            expected_columns = core_columns + (time_format_found if time_format_found else ['HR', 'MI', 'SC'])
            extra_columns = [col for col in file_columns if col not in expected_columns]
            
            # Validate data
            data_issues = self._validate_focal_data(df, time_format_found)
            
            # Determine if valid
            is_valid = len(missing_columns) == 0 and time_format_found is not None
            
            recommendations = []
            if missing_columns:
                recommendations.append(f"Add missing columns: {', '.join(missing_columns)}")
            if time_format_found is None:
                recommendations.append("Add time columns: either HR,MI,SC (separate) or Hr:Mi (combined)")
            if extra_columns and len(extra_columns) > 5:  # Many extra columns
                recommendations.append(f"File has {len(extra_columns)} extra columns - these will be ignored during processing")
            if data_issues:
                recommendations.extend(data_issues)
            
            return {
                'valid': is_valid,
                'file_path': str(file_path),
                'columns_found': file_columns,
                'missing_columns': missing_columns,
                'extra_columns': extra_columns,
                'time_format': time_format_found,
                'data_issues': data_issues,
                'recommendations': recommendations,
                'sample_data': df.head(3).to_dict('records') if is_valid else None
            }
            
        except Exception as e:
            logger.error(f"Unexpected error validating focal mechanism file: {e}")
            return {
                'valid': False,
                'error': f"Unexpected error: {e}",
                'missing_columns': [],
                'extra_columns': [],
                'recommendations': ['Check file and try again']
            }
    
    def _validate_hypocenter_data(self, df: pd.DataFrame) -> List[str]:
        """Validate hypocenter data content and data types."""
        issues = []
        
        try:
            # Check ID column format and data type
            if 'ID' in df.columns:
                id_issues = self._validate_id_column(df['ID'], 'hypocenter')
                issues.extend(id_issues)
            
            # Check coordinate columns are numeric
            coordinate_cols = ['LAT', 'LON', 'DEPTH', 'X', 'Y', 'Z']
            for col in coordinate_cols:
                if col in df.columns:
                    numeric_issues = self._validate_numeric_column(df[col], col, allow_negative=(col == 'Z'))
                    issues.extend(numeric_issues)
            
            # Check error columns are numeric and positive
            error_cols = ['EX', 'EY', 'EZ']
            for col in error_cols:
                if col in df.columns:
                    error_issues = self._validate_error_column(df[col], col)
                    issues.extend(error_issues)
            
            # Check coordinate ranges
            if 'LAT' in df.columns:
                lat_range = (df['LAT'].min(), df['LAT'].max())
                if not (-90 <= lat_range[0] <= 90 and -90 <= lat_range[1] <= 90):
                    issues.append(f"Latitude values outside valid range [-90,90]: {lat_range}")
            
            if 'LON' in df.columns:
                lon_range = (df['LON'].min(), df['LON'].max())
                if not (-180 <= lon_range[0] <= 180 and -180 <= lon_range[1] <= 180):
                    issues.append(f"Longitude values outside valid range [-180,180]: {lon_range}")
            
            # Check depth (should be positive for below ground)
            if 'DEPTH' in df.columns:
                if df['DEPTH'].min() < 0:
                    issues.append("Negative depth values found (depths should be positive for below ground)")
                if df['DEPTH'].max() > 1000:
                    issues.append("Very deep events found (>1000km), please verify depth units")
            
            # Check magnitude
            if 'MAG' in df.columns:
                mag_issues = self._validate_magnitude_column(df['MAG'])
                issues.extend(mag_issues)
            
            # Check time columns
            time_issues = self._validate_time_columns(df)
            issues.extend(time_issues)
            
            # Check integer columns
            integer_cols = ['YR', 'MO', 'DY', 'HR', 'MI', 'NCCP', 'NCCS', 'NCTP', 'NCTS', 'CID']
            for col in integer_cols:
                if col in df.columns:
                    int_issues = self._validate_integer_column(df[col], col)
                    issues.extend(int_issues)
            
        except Exception as e:
            issues.append(f"Error validating data content: {e}")
        
        return issues
    
    def _validate_focal_data(self, df: pd.DataFrame, time_format: Optional[List[str]]) -> List[str]:
        """Validate focal mechanism data content and data types."""
        issues = []
        
        try:
            # Check ID column format
            if 'ID' in df.columns:
                id_issues = self._validate_id_column(df['ID'], 'focal mechanism')
                issues.extend(id_issues)
            
            # Check coordinate columns are numeric
            coordinate_cols = ['LAT', 'LON', 'DEPTH', 'X', 'Y', 'Z']
            for col in coordinate_cols:
                if col in df.columns:
                    numeric_issues = self._validate_numeric_column(df[col], col, allow_negative=(col == 'Z'))
                    issues.extend(numeric_issues)
            
            # Check strike, dip, rake ranges and formats
            focal_mechanism_cols = {
                'Strike1': (0, 360, "Strike1 values should be between 0-360 degrees"),
                'Strike2': (0, 360, "Strike2 values should be between 0-360 degrees"),
                'Dip1': (0, 90, "Dip1 values should be between 0-90 degrees"),
                'Dip2': (0, 90, "Dip2 values should be between 0-90 degrees"),
                'Rake1': (-180, 180, "Rake1 values should be between -180 to 180 degrees"),
                'Rake2': (-180, 180, "Rake2 values should be between -180 to 180 degrees"),
                'Pazim': (0, 360, "P-axis azimuth should be between 0-360 degrees"),
                'Tazim': (0, 360, "T-axis azimuth should be between 0-360 degrees"),
                'Pdip': (0, 90, "P-axis dip should be between 0-90 degrees"),
                'Tdip': (0, 90, "T-axis dip should be between 0-90 degrees")
            }
            
            for col, (min_val, max_val, error_msg) in focal_mechanism_cols.items():
                if col in df.columns:
                    # Check numeric format first
                    numeric_issues = self._validate_numeric_column(df[col], col, allow_negative=(min_val < 0))
                    issues.extend(numeric_issues)
                    
                    # Check range
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    valid_values = numeric_series.dropna()
                    
                    if len(valid_values) > 0:
                        out_of_range = ((valid_values < min_val) | (valid_values > max_val)).sum()
                        if out_of_range > 0:
                            issues.append(f"Column '{col}': {out_of_range} values outside valid range [{min_val},{max_val}]")
            
            # Check magnitude
            if 'MAG' in df.columns:
                mag_issues = self._validate_magnitude_column(df['MAG'])
                issues.extend(mag_issues)
            
            # Check coordinates (similar to hypocenter validation)
            if 'LAT' in df.columns:
                lat_range = (df['LAT'].min(), df['LAT'].max())
                if not (-90 <= lat_range[0] <= 90 and -90 <= lat_range[1] <= 90):
                    issues.append(f"Latitude values outside valid range [-90,90]: {lat_range}")
            
            if 'LON' in df.columns:
                lon_range = (df['LON'].min(), df['LON'].max())
                if not (-180 <= lon_range[0] <= 180 and -180 <= lon_range[1] <= 180):
                    issues.append(f"Longitude values outside valid range [-180,180]: {lon_range}")
            
            # Check time format specific validation
            if time_format:
                time_issues = self._validate_focal_time_format(df, time_format)
                issues.extend(time_issues)
            
            # Check integer columns
            integer_cols = ['YR', 'MO', 'DY', 'HR', 'MI', 'SC', 'Hr', 'Mi', 'Sc']
            for col in integer_cols:
                if col in df.columns:
                    int_issues = self._validate_integer_column(df[col], col)
                    issues.extend(int_issues)
            
        except Exception as e:
            issues.append(f"Error validating focal mechanism data: {e}")
        
        return issues
    
    def _validate_time_columns(self, df: pd.DataFrame) -> List[str]:
        """Validate time columns in hypocenter data."""
        issues = []
        
        try:
            if 'YR' in df.columns:
                year_range = (df['YR'].min(), df['YR'].max())
                if year_range[0] < 1900 or year_range[1] > 2030:
                    issues.append(f"Unusual year range: {year_range}")
            
            if 'MO' in df.columns:
                month_range = (df['MO'].min(), df['MO'].max())
                if not (1 <= month_range[0] <= 12 and 1 <= month_range[1] <= 12):
                    issues.append(f"Invalid month values: {month_range}")
            
            if 'DY' in df.columns:
                day_range = (df['DY'].min(), df['DY'].max())
                if not (1 <= day_range[0] <= 31 and 1 <= day_range[1] <= 31):
                    issues.append(f"Invalid day values: {day_range}")
            
            if 'HR' in df.columns:
                hour_range = (df['HR'].min(), df['HR'].max())
                if not (0 <= hour_range[0] <= 23 and 0 <= hour_range[1] <= 23):
                    issues.append(f"Invalid hour values: {hour_range}")
            
            if 'MI' in df.columns:
                min_range = (df['MI'].min(), df['MI'].max())
                if not (0 <= min_range[0] <= 59 and 0 <= min_range[1] <= 59):
                    issues.append(f"Invalid minute values: {min_range}")
            
        except Exception as e:
            issues.append(f"Error validating time columns: {e}")
        
        return issues
    
    def _validate_focal_time_format(self, df: pd.DataFrame, time_format: List[str]) -> List[str]:
        """Validate time format in focal mechanism data."""
        issues = []
        
        try:
            if time_format == ['Hr:Mi']:
                # Validate combined time format
                if 'Hr:Mi' in df.columns:
                    sample_times = df['Hr:Mi'].dropna().head(10)
                    for time_str in sample_times:
                        if ':' not in str(time_str):
                            issues.append("Hr:Mi column should contain time in format HH:MM or HH:MM:SS")
                            break
            else:
                # Validate separate time columns
                if 'HR' in time_format or 'Hr' in time_format:
                    hr_col = 'HR' if 'HR' in df.columns else 'Hr'
                    if hr_col in df.columns:
                        hour_range = (df[hr_col].min(), df[hr_col].max())
                        if not (0 <= hour_range[0] <= 23 and 0 <= hour_range[1] <= 23):
                            issues.append(f"Invalid hour values in {hr_col}: {hour_range}")
                
                if 'MI' in time_format or 'Mi' in time_format:
                    mi_col = 'MI' if 'MI' in df.columns else 'Mi'
                    if mi_col in df.columns:
                        min_range = (df[mi_col].min(), df[mi_col].max())
                        if not (0 <= min_range[0] <= 59 and 0 <= min_range[1] <= 59):
                            issues.append(f"Invalid minute values in {mi_col}: {min_range}")
            
        except Exception as e:
            issues.append(f"Error validating focal time format: {e}")
        
        return issues
    
    def validate_both_files(self, hypo_file: str, hypo_sep: str, focal_file: Optional[str] = None, 
                           focal_sep: str = ';') -> Dict:
        """
        Validate both hypocenter and focal mechanism files.
        
        Parameters
        ----------
        hypo_file : str
            Path to hypocenter file
        hypo_sep : str
            Hypocenter file separator
        focal_file : str, optional
            Path to focal mechanism file
        focal_sep : str
            Focal mechanism file separator
            
        Returns
        -------
        dict
            Combined validation results
        """
        results = {
            'hypocenter': self.validate_hypocenter_file(hypo_file, hypo_sep),
            'focal_mechanism': None,
            'overall_valid': False,
            'summary': []
        }
        
        if focal_file:
            results['focal_mechanism'] = self.validate_focal_mechanism_file(focal_file, focal_sep)
            results['overall_valid'] = (results['hypocenter']['valid'] and 
                                      results['focal_mechanism']['valid'])
        else:
            results['overall_valid'] = results['hypocenter']['valid']
        
        # Generate summary
        summary = []
        if results['hypocenter']['valid']:
            summary.append("✓ Hypocenter file is valid")
        else:
            summary.append("✗ Hypocenter file has issues")
        
        if focal_file:
            if results['focal_mechanism']['valid']:
                summary.append("✓ Focal mechanism file is valid")
            else:
                summary.append("✗ Focal mechanism file has issues")
        else:
            summary.append("• No focal mechanism file provided")
        
        results['summary'] = summary
        
        return results
    
    def print_validation_report(self, results: Dict):
        """Print a formatted validation report."""
        print("\n" + "="*60)
        print("HyFI INPUT FILE VALIDATION REPORT")
        print("="*60)
        
        # Overall status
        status = "✓ PASSED" if results['overall_valid'] else "✗ FAILED"
        print(f"\nOverall Status: {status}")
        
        # Summary
        print(f"\nSummary:")
        for item in results['summary']:
            print(f"  {item}")
        
        # Detailed results for hypocenter file
        print(f"\n{'─'*60}")
        print("HYPOCENTER FILE VALIDATION")
        print(f"{'─'*60}")
        
        hypo_results = results['hypocenter']
        if 'error' in hypo_results:
            print(f"Error: {hypo_results['error']}")
        else:
            print(f"File: {hypo_results.get('file_path', 'N/A')}")
            print(f"Status: {'✓ Valid' if hypo_results['valid'] else '✗ Invalid'}")
            print(f"Columns found: {len(hypo_results.get('columns_found', []))}")
            
            if hypo_results.get('missing_columns'):
                print(f"Missing columns: {', '.join(hypo_results['missing_columns'])}")
            
            if hypo_results.get('extra_columns'):
                print(f"Extra columns: {len(hypo_results['extra_columns'])} (will be ignored)")
            
            if hypo_results.get('data_issues'):
                print(f"Data issues:")
                for issue in hypo_results['data_issues']:
                    print(f"  • {issue}")
            
            if hypo_results.get('recommendations'):
                print(f"Recommendations:")
                for rec in hypo_results['recommendations']:
                    print(f"  • {rec}")
        
        # Detailed results for focal mechanism file
        if results['focal_mechanism']:
            print(f"\n{'─'*60}")
            print("FOCAL MECHANISM FILE VALIDATION")
            print(f"{'─'*60}")
            
            focal_results = results['focal_mechanism']
            if 'error' in focal_results:
                print(f"Error: {focal_results['error']}")
            else:
                print(f"File: {focal_results.get('file_path', 'N/A')}")
                print(f"Status: {'✓ Valid' if focal_results['valid'] else '✗ Invalid'}")
                print(f"Columns found: {len(focal_results.get('columns_found', []))}")
                print(f"Time format: {focal_results.get('time_format', 'Not detected')}")
                
                if focal_results.get('missing_columns'):
                    print(f"Missing columns: {', '.join(focal_results['missing_columns'])}")
                
                if focal_results.get('extra_columns'):
                    print(f"Extra columns: {len(focal_results['extra_columns'])} (will be ignored)")
                
                if focal_results.get('data_issues'):
                    print(f"Data issues:")
                    for issue in focal_results['data_issues']:
                        print(f"  • {issue}")
                
                if focal_results.get('recommendations'):
                    print(f"Recommendations:")
                    for rec in focal_results['recommendations']:
                        print(f"  • {rec}")
        
        print(f"\n{'='*60}")
    
    def _validate_id_column(self, id_series: pd.Series, file_type: str) -> List[str]:
        """Validate ID column format and uniqueness."""
        issues = []
        
        try:
            # Ensure ID is treated as string
            id_series = id_series.astype(str)
            
            # Check for missing values
            missing_count = id_series.isna().sum() + (id_series == 'nan').sum() + (id_series.str.strip() == '').sum()
            if missing_count > 0:
                issues.append(f"Found {missing_count} missing or empty values in ID column")
            
            # Check for duplicates
            duplicates = id_series.duplicated().sum()
            if duplicates > 0:
                issues.append(f"Found {duplicates} duplicate IDs")
            
            # Check ID format patterns
            sample_ids = id_series.dropna().head(10)
            for idx, id_val in enumerate(sample_ids):
                id_str = str(id_val).strip()
                if id_str == '' or id_str == 'nan':
                    issues.append(f"Empty or invalid ID found at row {idx}")
                    break
                # Check for reasonable ID length (not too short or too long)
                if len(id_str) < 1:
                    issues.append(f"Very short ID found: '{id_str}'")
                elif len(id_str) > 50:
                    issues.append(f"Very long ID found (length {len(id_str)}): '{id_str[:20]}...'")
            
            # Provide info about ID format
            if len(sample_ids) > 0:
                first_id = str(sample_ids.iloc[0])
                if first_id.startswith('KP'):
                    # SECOS format IDs
                    non_secos_count = (~id_series.str.startswith('KP')).sum()
                    if non_secos_count > 0:
                        issues.append(f"Found {non_secos_count} IDs that don't follow SECOS 'KP' format")
                
        except Exception as e:
            issues.append(f"Error validating ID column: {e}")
        
        return issues
    
    def _validate_numeric_column(self, series: pd.Series, col_name: str, allow_negative: bool = True) -> List[str]:
        """Validate that a column contains numeric values."""
        issues = []
        
        try:
            # Try to convert to numeric
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            # Check for non-numeric values
            non_numeric_count = numeric_series.isna().sum() - series.isna().sum()
            if non_numeric_count > 0:
                issues.append(f"Column '{col_name}' contains {non_numeric_count} non-numeric values")
            
            # Check for negative values if not allowed
            if not allow_negative and (numeric_series < 0).any():
                negative_count = (numeric_series < 0).sum()
                issues.append(f"Column '{col_name}' contains {negative_count} negative values (should be positive)")
            
            # Check for infinite values
            if np.isinf(numeric_series).any():
                inf_count = np.isinf(numeric_series).sum()
                issues.append(f"Column '{col_name}' contains {inf_count} infinite values")
                
        except Exception as e:
            issues.append(f"Error validating numeric column '{col_name}': {e}")
        
        return issues
    
    def _validate_error_column(self, series: pd.Series, col_name: str) -> List[str]:
        """Validate error columns (should be positive numeric or missing)."""
        issues = []
        
        try:
            # Handle missing error columns gracefully (they can be filled with defaults)
            if series.isna().all():
                return issues  # All missing is OK for error columns
            
            # Check numeric format
            numeric_issues = self._validate_numeric_column(series, col_name, allow_negative=False)
            issues.extend(numeric_issues)
            
            # Check for reasonable error ranges (errors should be > 0 and < 10000m typically)
            numeric_series = pd.to_numeric(series, errors='coerce')
            valid_errors = numeric_series.dropna()
            
            if len(valid_errors) > 0:
                if (valid_errors <= 0).any():
                    zero_count = (valid_errors <= 0).sum()
                    issues.append(f"Column '{col_name}' contains {zero_count} zero or negative error values")
                
                if (valid_errors > 10000).any():
                    large_count = (valid_errors > 10000).sum()
                    issues.append(f"Column '{col_name}' contains {large_count} very large error values (>10km)")
                    
        except Exception as e:
            issues.append(f"Error validating error column '{col_name}': {e}")
        
        return issues
    
    def _validate_magnitude_column(self, series: pd.Series) -> List[str]:
        """Validate magnitude column."""
        issues = []
        
        try:
            # Check numeric format
            numeric_issues = self._validate_numeric_column(series, 'MAG', allow_negative=True)
            issues.extend(numeric_issues)
            
            # Check magnitude range
            numeric_series = pd.to_numeric(series, errors='coerce')
            valid_mags = numeric_series.dropna()
            
            if len(valid_mags) > 0:
                mag_range = (valid_mags.min(), valid_mags.max())
                if mag_range[0] < -3 or mag_range[1] > 10:
                    issues.append(f"Unusual magnitude range: {mag_range} (expected range: -3 to 10)")
                    
        except Exception as e:
            issues.append(f"Error validating magnitude column: {e}")
        
        return issues
    
    def _validate_integer_column(self, series: pd.Series, col_name: str) -> List[str]:
        """Validate that a column contains integer values."""
        issues = []
        
        try:
            # Convert to numeric first
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            # Check for non-numeric values
            non_numeric_count = numeric_series.isna().sum() - series.isna().sum()
            if non_numeric_count > 0:
                issues.append(f"Column '{col_name}' contains {non_numeric_count} non-integer values")
            
            # Check if values are actually integers (no decimal parts)
            valid_numeric = numeric_series.dropna()
            if len(valid_numeric) > 0:
                non_integer_count = ((valid_numeric % 1) != 0).sum()
                if non_integer_count > 0:
                    issues.append(f"Column '{col_name}' contains {non_integer_count} non-integer values")
                    
        except Exception as e:
            issues.append(f"Error validating integer column '{col_name}': {e}")
        
        return issues


def validate_input_files(hypo_file: str, hypo_sep: str = '\t', 
                        focal_file: Optional[str] = None, focal_sep: str = ';',
                        print_report: bool = True) -> Dict:
    """
    Convenience function to validate HyFI input files.
    
    Parameters
    ----------
    hypo_file : str
        Path to hypocenter data file
    hypo_sep : str, default '\t'
        Separator for hypocenter file
    focal_file : str, optional
        Path to focal mechanism file
    focal_sep : str, default ';'
        Separator for focal mechanism file
    print_report : bool, default True
        Whether to print validation report
        
    Returns
    -------
    dict
        Validation results
    """
    validator = InputFileValidator()
    results = validator.validate_both_files(hypo_file, hypo_sep, focal_file, focal_sep)
    
    if print_report:
        validator.print_validation_report(results)
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python input_validation.py <hypocenter_file> [focal_file]")
        print("Example: python input_validation.py data.csv focals.csv")
        sys.exit(1)
    
    hypo_file = sys.argv[1]
    focal_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    validate_input_files(hypo_file, focal_file=focal_file)
