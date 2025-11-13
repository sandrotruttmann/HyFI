#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration validation utilities for HyFI.

This module provides validation functions and custom validators for configuration parameters.
"""

import re
from pathlib import Path
from typing import Union, Any, Optional


class ConfigValidationError(ValueError):
    """Custom exception for configuration validation errors."""
    pass


def validate_file_exists(file_path: Union[str, Path], required: bool = True) -> Optional[Path]:
    """
    Validate that a file exists.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file
    required : bool
        Whether the file is required to exist
        
    Returns
    -------
    Optional[Path]
        Validated Path object or None if not required and doesn't exist
        
    Raises
    ------
    ConfigValidationError
        If file doesn't exist and is required
    """
    if file_path is None or file_path == "":
        if required:
            raise ConfigValidationError("File path is required but not provided")
        return None
    
    path = Path(file_path)
    if not path.exists():
        if required:
            raise ConfigValidationError(f"File does not exist: {path}")
        return None
    
    return path


def validate_positive_number(value: Union[int, float], name: str) -> Union[int, float]:
    """
    Validate that a number is positive.
    
    Parameters
    ----------
    value : Union[int, float]
        Value to validate
    name : str
        Name of the parameter for error messages
        
    Returns
    -------
    Union[int, float]
        Validated value
        
    Raises
    ------
    ConfigValidationError
        If value is not positive
    """
    if value <= 0:
        raise ConfigValidationError(f"{name} must be positive, got {value}")
    return value


def validate_range(value: Union[int, float], min_val: float, max_val: float, name: str) -> Union[int, float]:
    """
    Validate that a value is within a specified range.
    
    Parameters
    ----------
    value : Union[int, float]
        Value to validate
    min_val : float
        Minimum allowed value
    max_val : float
        Maximum allowed value
    name : str
        Name of the parameter for error messages
        
    Returns
    -------
    Union[int, float]
        Validated value
        
    Raises
    ------
    ConfigValidationError
        If value is outside the range
    """
    if not (min_val <= value <= max_val):
        raise ConfigValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return value


def validate_choice(value: Any, choices: list, name: str) -> Any:
    """
    Validate that a value is one of the allowed choices.
    
    Parameters
    ----------
    value : Any
        Value to validate
    choices : list
        List of allowed choices
    name : str
        Name of the parameter for error messages
        
    Returns
    -------
    Any
        Validated value
        
    Raises
    ------
    ConfigValidationError
        If value is not in choices
    """
    if value not in choices:
        raise ConfigValidationError(f"{name} must be one of {choices}, got {value}")
    return value


def validate_separator(separator: str) -> str:
    """
    Validate file separator characters.
    
    Parameters
    ----------
    separator : str
        Separator character(s)
        
    Returns
    -------
    str
        Validated separator
        
    Raises
    ------
    ConfigValidationError
        If separator is invalid
    """
    valid_separators = ['\t', ',', ';', ' ', '|']
    if separator not in valid_separators:
        raise ConfigValidationError(f"Separator must be one of {valid_separators}, got '{separator}'")
    return separator


def validate_project_title(title: str) -> str:
    """
    Validate project title.
    
    Parameters
    ----------
    title : str
        Project title
        
    Returns
    -------
    str
        Validated title
        
    Raises
    ------
    ConfigValidationError
        If title is invalid
    """
    if not title or not title.strip():
        raise ConfigValidationError("Project title cannot be empty")
    
    # Check for invalid characters that might cause issues in file names
    invalid_chars = r'[<>:"/\\|?*]'
    if re.search(invalid_chars, title):
        raise ConfigValidationError(f"Project title contains invalid characters: {title}")
    
    return title.strip()


def validate_output_directory(out_dir: Union[str, Path]) -> Path:
    """
    Validate and create output directory if it doesn't exist.
    
    Parameters
    ----------
    out_dir : Union[str, Path]
        Output directory path
        
    Returns
    -------
    Path
        Validated and potentially created directory path
        
    Raises
    ------
    ConfigValidationError
        If directory cannot be created or accessed
    """
    path = Path(out_dir)
    
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise ConfigValidationError(f"Permission denied creating directory: {path}")
    except OSError as e:
        raise ConfigValidationError(f"Cannot create directory {path}: {e}")
    
    if not path.is_dir():
        raise ConfigValidationError(f"Path exists but is not a directory: {path}")
    
    # Test write permissions
    test_file = path / ".hyfi_test_write"
    try:
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError):
        raise ConfigValidationError(f"No write permission in directory: {path}")
    
    return path
