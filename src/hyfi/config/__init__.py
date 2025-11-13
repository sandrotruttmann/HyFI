#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management for fault imaging.

This module contains configuration classes, parameter management, validation,
and configuration file I/O functionality.
"""

from .parameters import ProjectConfig, FaultNetworkConfig, ModelValidationConfig, AutoClassConfig, StressAnalysisConfig
from .validation import ConfigValidationError
from .io import (
    load_config_from_json, 
    save_config_to_json,
    auto_load_config,
    create_template_config,
    load_dag_from_json,
    save_dag_to_json,
    create_template_dag,
    validate_dag_file,
    convert_legacy_to_dag
)

__all__ = [
    'ProjectConfig',
    'FaultNetworkConfig',
    'ModelValidationConfig', 
    'AutoClassConfig',
    'StressAnalysisConfig',
    'ConfigValidationError',
    'load_config_from_json',
    'save_config_to_json',
    'auto_load_config',
    'create_template_config',
    'load_dag_from_json',
    'save_dag_to_json',
    'create_template_dag',
    'validate_dag_file',
    'convert_legacy_to_dag'
]
