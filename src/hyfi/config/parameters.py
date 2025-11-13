#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration classes for fault imaging parameters.

This module provides structured configuration management using dataclasses
to replace the dictionary-based parameter system.
"""

from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path

from .validation import (
    ConfigValidationError,
    validate_file_exists,
    validate_positive_number,
    validate_range,
    validate_choice,
    validate_separator,
    validate_project_title,
    validate_output_directory
)


@dataclass
class FaultNetworkConfig:
    """Configuration for fault network reconstruction module."""
    n_mc: int = 1000                    # number of Monte Carlo simulations
    r_nn: Union[float, str] = 100.0     # search radius [m] of nearest neighbor search or 'auto'
    dt_nn: Union[float, str] = 26298.0  # search time window [h] or 'auto'
    mag_type: str = 'ML'                # magnitude type: 'ML' or 'Mw'
    
    # Parameter optimization settings
    auto_optimize_parameters: bool = False      # enable automatic parameter optimization
    optimization_method: str = 'grid_search'   # optimization method: 'grid_search', 'bayesian', 'optuna', or 'heuristic'
    optimization_grid_points: int = 25         # number of grid points per dimension for grid search
    
    # Bayesian optimization specific settings
    optimization_n_calls: int = 50              # total evaluations for Bayesian optimization
    optimization_n_initial_points: int = 10     # random initialization points for Bayesian
    optimization_acquisition_func: str = 'EI'   # acquisition function: 'EI', 'PI', 'LCB', 'gp_hedge'
    
    # Optuna optimization specific settings
    optimization_n_trials: int = 50             # total trials for Optuna optimization
    optimization_sampler: str = 'tpe'           # Optuna sampler: 'tpe', 'cmaes', 'random'
    optimization_n_startup_trials: int = 10     # random trials before sampler optimization
    
    # Pareto multi-objective optimization parameters
    optimization_pareto_sampler: str = 'nsga2'  # Pareto sampler: 'nsga2', 'nsga3', 'random'
    optimization_pareto_population: int = 50    # population size for evolutionary algorithms
    
    def validate(self):
        """Validate configuration parameters."""
        validate_positive_number(self.n_mc, "n_mc")
        
        # Validate r_nn (can be numeric or 'auto')
        if isinstance(self.r_nn, str):
            if self.r_nn != 'auto':
                raise ValueError("r_nn must be a positive number or 'auto'")
        else:
            validate_positive_number(self.r_nn, "r_nn")
        
        # Validate dt_nn (can be numeric or 'auto')
        if isinstance(self.dt_nn, str):
            if self.dt_nn != 'auto':
                raise ValueError("dt_nn must be a positive number or 'auto'")
        else:
            validate_positive_number(self.dt_nn, "dt_nn")
            
        validate_choice(self.mag_type, ['ML', 'Mw'], "mag_type")
        validate_choice(self.optimization_method, ['grid_search', 'bayesian', 'optuna', 'pareto', 'heuristic'], "optimization_method")
        validate_positive_number(self.optimization_grid_points, "optimization_grid_points")
        
        # Validate Bayesian optimization parameters
        validate_positive_number(self.optimization_n_calls, "optimization_n_calls")
        validate_positive_number(self.optimization_n_initial_points, "optimization_n_initial_points")
        validate_choice(self.optimization_acquisition_func, ['EI', 'PI', 'LCB', 'gp_hedge'], "optimization_acquisition_func")
        
        # Validate Optuna optimization parameters
        validate_positive_number(self.optimization_n_trials, "optimization_n_trials")
        validate_choice(self.optimization_sampler, ['tpe', 'cmaes', 'random'], "optimization_sampler")
        validate_positive_number(self.optimization_n_startup_trials, "optimization_n_startup_trials")
        
        # Validate Pareto optimization parameters
        validate_choice(self.optimization_pareto_sampler, ['nsga2', 'nsga3', 'random'], "optimization_pareto_sampler")
        validate_positive_number(self.optimization_pareto_population, "optimization_pareto_population")
        
        # Auto-enable optimization if 'auto' parameters are specified
        if (self.r_nn == 'auto' or self.dt_nn == 'auto') and not self.auto_optimize_parameters:
            self.auto_optimize_parameters = True


@dataclass
class ModelValidationConfig:
    """Configuration for model validation module."""
    validation_bool: bool = True
    foc_file: Optional[Union[str, Path]] = None
    foc_sep: str = ';'
    foc_mag_check: bool = True          # check focal magnitude (recommended)
    foc_loc_check: bool = True          # check focal location (recommended)
    
    def validate(self):
        """Validate configuration parameters."""
        if self.validation_bool:
            validate_file_exists(self.foc_file, required=True)
            # Also validate the focal mechanism file format
            if self.foc_file:
                from ..utils.input_validation import InputFileValidator
                try:
                    validator = InputFileValidator()
                    focal_validation = validator.validate_focal_mechanism_file(str(self.foc_file), self.foc_sep)
                    
                    if not focal_validation['valid']:
                        error_msg = f"Focal mechanism file validation failed: {focal_validation.get('error', 'Unknown error')}"
                        if focal_validation.get('missing_columns'):
                            error_msg += f"\nMissing columns: {', '.join(focal_validation['missing_columns'])}"
                        if focal_validation.get('recommendations'):
                            error_msg += f"\nRecommendations: {'; '.join(focal_validation['recommendations'])}"
                        raise ValueError(error_msg)
                    
                except ValueError as e:
                    raise ConfigValidationError(
                        f"Focal mechanism file format validation failed: {e}\n"
                        f"To disable focal mechanism validation, set 'validation_bool': false in your configuration."
                    )
        validate_separator(self.foc_sep)


@dataclass
class AutoClassConfig:
    """Configuration for automatic classification module."""
    autoclass_bool: bool = True
    n_clusters: int = 2                 # number of expected classes
    algorithm: str = 'vmf_soft'         # clustering algorithm
    rotation: bool = True               # rotate poles before analysis
    
    def validate(self):
        """Validate configuration parameters."""
        validate_positive_number(self.n_clusters, "n_clusters")
        validate_choice(self.algorithm, ['vmf_soft', 'vmf_hard', 'spherical_kmeans'], "algorithm")


@dataclass
class StressAnalysisConfig:
    """Configuration for fault stress analysis module."""
    stress_bool: bool = True
    S1_trend: float = 301               # σ1 trend
    S1_plunge: float = 23               # σ1 plunge
    S3_trend: float = 43                # σ3 trend
    S3_plunge: float = 26               # σ3 plunge
    stress_R: float = 0.35              # Stress shape ratio
    PP: float = 0                       # Pore pressure
    fric_coeff: float = 0.75            # Friction coefficient
    
    def validate(self):
        """Validate configuration parameters."""
        validate_range(self.S1_trend, 0, 360, "S1_trend")
        validate_range(self.S1_plunge, 0, 90, "S1_plunge")
        validate_range(self.S3_trend, 0, 360, "S3_trend")
        validate_range(self.S3_plunge, 0, 90, "S3_plunge")
        validate_range(self.stress_R, 0, 1, "stress_R")
        validate_positive_number(self.fric_coeff, "fric_coeff")


@dataclass
class ProjectConfig:
    """Main configuration class containing all module configurations."""
    
    # General settings
    project_title: str = "Fault Imaging Analysis"
    
    # Input files
    hypo_file: Union[str, Path] = ""
    hypo_sep: str = '\t'
    
    # Output settings
    out_dir: Union[str, Path] = field(default_factory=Path.cwd)
    
    # Module configurations
    fault_network: FaultNetworkConfig = field(default_factory=FaultNetworkConfig)
    model_validation: ModelValidationConfig = field(default_factory=ModelValidationConfig)
    auto_class: AutoClassConfig = field(default_factory=AutoClassConfig)
    stress_analysis: StressAnalysisConfig = field(default_factory=StressAnalysisConfig)
    
    def __post_init__(self):
        """Convert string paths to Path objects and validate basic properties."""
        if isinstance(self.hypo_file, str):
            self.hypo_file = Path(self.hypo_file)
        if isinstance(self.out_dir, str):
            self.out_dir = Path(self.out_dir)
        
        # Basic validation on initialization
        self.project_title = validate_project_title(self.project_title)
        self.hypo_sep = validate_separator(self.hypo_sep)
    
    def validate(self):
        """Validate all configuration parameters."""
        # Validate file paths
        validate_file_exists(self.hypo_file, required=True)
        self.out_dir = validate_output_directory(self.out_dir)
        
        # Validate all subconfigurations
        self.fault_network.validate()
        self.model_validation.validate()
        self.auto_class.validate()
        self.stress_analysis.validate()
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ProjectConfig':
        """Create ProjectConfig from dictionary (for backward compatibility)."""
        # Extract nested configurations
        fault_network_params = {
            'n_mc': config_dict.get('n_mc', 1000),
            'r_nn': config_dict.get('r_nn', 100.0),
            'dt_nn': config_dict.get('dt_nn', 26298.0),
            'mag_type': config_dict.get('mag_type', 'ML'),
            'auto_optimize_parameters': config_dict.get('auto_optimize_parameters', False),
            'optimization_method': config_dict.get('optimization_method', 'grid_search'),
            'optimization_grid_points': config_dict.get('optimization_grid_points', 25),
            'optimization_n_calls': config_dict.get('optimization_n_calls', 50),
            'optimization_n_initial_points': config_dict.get('optimization_n_initial_points', 10),
            'optimization_acquisition_func': config_dict.get('optimization_acquisition_func', 'EI'),
            'optimization_n_trials': config_dict.get('optimization_n_trials', 50),
            'optimization_sampler': config_dict.get('optimization_sampler', 'tpe'),
            'optimization_n_startup_trials': config_dict.get('optimization_n_startup_trials', 10),
            'optimization_pareto_sampler': config_dict.get('optimization_pareto_sampler', 'nsga2'),
            'optimization_pareto_population': config_dict.get('optimization_pareto_population', 50)
        }
        
        model_validation_params = {
            'validation_bool': config_dict.get('validation_bool', True),
            'foc_file': config_dict.get('foc_file'),
            'foc_sep': config_dict.get('foc_sep', ';'),
            'foc_mag_check': config_dict.get('foc_mag_check', True),
            'foc_loc_check': config_dict.get('foc_loc_check', True)
        }
        
        auto_class_params = {
            'autoclass_bool': config_dict.get('autoclass_bool', True),
            'n_clusters': config_dict.get('n_clusters', 2),
            'algorithm': config_dict.get('algorithm', 'vmf_soft'),
            'rotation': config_dict.get('rotation', True)
        }
        
        stress_analysis_params = {
            'stress_bool': config_dict.get('stress_bool', True),
            'S1_trend': config_dict.get('S1_trend', 301),
            'S1_plunge': config_dict.get('S1_plunge', 23),
            'S3_trend': config_dict.get('S3_trend', 43),
            'S3_plunge': config_dict.get('S3_plunge', 26),
            'stress_R': config_dict.get('stress_R', 0.35),
            'PP': config_dict.get('PP', 0),
            'fric_coeff': config_dict.get('fric_coeff', 0.75)
        }
        
        return cls(
            project_title=config_dict.get('project_title', 'Fault Imaging Analysis'),
            hypo_file=config_dict.get('hypo_file', ''),
            hypo_sep=config_dict.get('hypo_sep', '\t'),
            out_dir=config_dict.get('out_dir', Path.cwd()),
            fault_network=FaultNetworkConfig(**fault_network_params),
            model_validation=ModelValidationConfig(**model_validation_params),
            auto_class=AutoClassConfig(**auto_class_params),
            stress_analysis=StressAnalysisConfig(**stress_analysis_params)
        )
    
    def to_dict(self) -> dict:
        """Convert ProjectConfig to dictionary (for backward compatibility)."""
        result = {
            'project_title': self.project_title,
            'hypo_file': str(self.hypo_file),
            'hypo_sep': self.hypo_sep,
            'out_dir': str(self.out_dir),
            
            # Fault network parameters
            'n_mc': self.fault_network.n_mc,
            'r_nn': self.fault_network.r_nn,
            'dt_nn': self.fault_network.dt_nn,
            'mag_type': self.fault_network.mag_type,
            'auto_optimize_parameters': self.fault_network.auto_optimize_parameters,
            'optimization_method': self.fault_network.optimization_method,
            'optimization_grid_points': self.fault_network.optimization_grid_points,
            'optimization_n_calls': self.fault_network.optimization_n_calls,
            'optimization_n_initial_points': self.fault_network.optimization_n_initial_points,
            'optimization_acquisition_func': self.fault_network.optimization_acquisition_func,
            'optimization_n_trials': self.fault_network.optimization_n_trials,
            'optimization_sampler': self.fault_network.optimization_sampler,
            'optimization_n_startup_trials': self.fault_network.optimization_n_startup_trials,
            
            # Model validation parameters
            'validation_bool': self.model_validation.validation_bool,
            'foc_file': str(self.model_validation.foc_file) if self.model_validation.foc_file else None,
            'foc_sep': self.model_validation.foc_sep,
            'foc_mag_check': self.model_validation.foc_mag_check,
            'foc_loc_check': self.model_validation.foc_loc_check,
            
            # Auto classification parameters
            'autoclass_bool': self.auto_class.autoclass_bool,
            'n_clusters': self.auto_class.n_clusters,
            'algorithm': self.auto_class.algorithm,
            'rotation': self.auto_class.rotation,
            
            # Stress analysis parameters
            'stress_bool': self.stress_analysis.stress_bool,
            'S1_trend': self.stress_analysis.S1_trend,
            'S1_plunge': self.stress_analysis.S1_plunge,
            'S3_trend': self.stress_analysis.S3_trend,
            'S3_plunge': self.stress_analysis.S3_plunge,
            'stress_R': self.stress_analysis.stress_R,
            'PP': self.stress_analysis.PP,
            'fric_coeff': self.stress_analysis.fric_coeff
        }
        
        return result
