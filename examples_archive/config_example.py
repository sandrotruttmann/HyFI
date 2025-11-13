#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example demonstrating the enhanced configuration management in HyFI.

This script shows:
- Creating configurations programmatically
- Loading/saving configurations from files
- Configuration validation
- Using different file formats
"""

from pathlib import Path
import sys

# Add the src directory to the path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hyfi.config import (
    ProjectConfig, FaultNetworkConfig, ModelValidationConfig,
    AutoClassConfig, StressAnalysisConfig,
    save_config_to_yaml, save_config_to_json, save_config_to_ini,
    load_config_from_yaml, auto_load_config,
    create_template_config, ConfigValidationError
)
from hyfi.core.workflow import FaultImagingWorkflow


def example_programmatic_config():
    """Example of creating configuration programmatically with validation."""
    
    print("=== Programmatic Configuration Example ===")
    
    try:
        # Create a detailed configuration
        config = ProjectConfig(
            project_title="St. Leonard Analysis - Config Demo",
            hypo_file="./data_examples/StLeonard/hypoDD_StLeonard.txt",
            hypo_sep="\t",
            out_dir="./output_config_demo",
            
            fault_network=FaultNetworkConfig(
                n_mc=500,  # Reduced for demo
                r_nn=150.0,
                dt_nn=8766.0,  # 1 year
                mag_type="ML"
            ),
            
            model_validation=ModelValidationConfig(
                validation_bool=True,
                foc_file="./data_examples/StLeonard/FocalMechanisms_StLeonard.txt",
                foc_sep=";",
                foc_mag_check=True,
                foc_loc_check=True
            ),
            
            auto_class=AutoClassConfig(
                autoclass_bool=True,
                n_clusters=3,
                algorithm="vmf_soft",
                rotation=True
            ),
            
            stress_analysis=StressAnalysisConfig(
                stress_bool=True,
                S1_trend=145.0,
                S1_plunge=15.0,
                S3_trend=45.0,
                S3_plunge=20.0,
                stress_R=0.4,
                PP=0.0,
                fric_coeff=0.8
            )
        )
        
        # Validate configuration
        config.validate()
        print("✓ Configuration created and validated successfully")
        
        # Show configuration summary
        print(f"Project: {config.project_title}")
        print(f"MC simulations: {config.fault_network.n_mc}")
        print(f"Search radius: {config.fault_network.r_nn} m")
        print(f"Clusters: {config.auto_class.n_clusters}")
        
        return config
        
    except ConfigValidationError as e:
        print(f"✗ Configuration validation failed: {e}")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def example_file_formats(config):
    """Example of saving and loading different configuration file formats."""
    
    print("\n=== Configuration File Formats Example ===")
    
    # Save in different formats
    save_config_to_yaml(config, "demo_config.yaml")
    save_config_to_json(config, "demo_config.json") 
    save_config_to_ini(config, "demo_config.ini")
    
    print("✓ Configuration saved in YAML, JSON, and INI formats")
    
    # Load and validate each format
    formats = [
        ("YAML", "demo_config.yaml"),
        ("JSON", "demo_config.json"),
        ("INI", "demo_config.ini")
    ]
    
    for format_name, filename in formats:
        try:
            loaded_config = auto_load_config(filename)
            loaded_config.validate()
            print(f"✓ {format_name} configuration loaded and validated")
        except Exception as e:
            print(f"✗ {format_name} configuration failed: {e}")


def example_validation_errors():
    """Example demonstrating configuration validation."""
    
    print("\n=== Configuration Validation Example ===")
    
    # Test various validation scenarios
    validation_tests = [
        ("Negative Monte Carlo", lambda: ProjectConfig(
            project_title="Test",
            hypo_file="./data_examples/StLeonard/hypoDD_StLeonard.txt",
            fault_network=FaultNetworkConfig(n_mc=-100)
        )),
        
        ("Invalid magnitude type", lambda: ProjectConfig(
            project_title="Test", 
            hypo_file="./data_examples/StLeonard/hypoDD_StLeonard.txt",
            fault_network=FaultNetworkConfig(mag_type="MW")  # Wrong case
        )),
        
        ("Invalid stress ratio", lambda: ProjectConfig(
            project_title="Test",
            hypo_file="./data_examples/StLeonard/hypoDD_StLeonard.txt",
            stress_analysis=StressAnalysisConfig(stress_R=1.5)  # > 1
        )),
        
        ("Empty project title", lambda: ProjectConfig(
            project_title="",
            hypo_file="./data_examples/StLeonard/hypoDD_StLeonard.txt"
        )),
        
        ("Nonexistent file", lambda: ProjectConfig(
            project_title="Test",
            hypo_file="./nonexistent_file.txt"
        ))
    ]
    
    for test_name, config_func in validation_tests:
        try:
            config = config_func()
            config.validate()
            print(f"✗ {test_name}: Expected validation error but none occurred")
        except ConfigValidationError as e:
            print(f"✓ {test_name}: Correctly caught validation error - {e}")
        except Exception as e:
            print(f"? {test_name}: Unexpected error - {e}")


def example_template_creation():
    """Example of creating configuration templates."""
    
    print("\n=== Template Creation Example ===")
    
    # Create templates in different formats
    try:
        create_template_config("template_config.yaml", "yaml")
        create_template_config("template_config.json", "json")
        create_template_config("template_config.ini", "ini")
        
        print("✓ Template configurations created in all formats")
        print("Files created: template_config.yaml, template_config.json, template_config.ini")
        
    except Exception as e:
        print(f"✗ Template creation failed: {e}")


def example_workflow_with_config():
    """Example of using configuration with the workflow."""
    
    print("\n=== Workflow with Configuration Example ===")
    
    try:
        # Load configuration from file
        config = load_config_from_yaml("demo_config.yaml")
        
        # Modify for quick demo
        config.fault_network.n_mc = 10  # Very small for demo
        config.model_validation.validation_bool = False
        config.auto_class.autoclass_bool = False
        config.stress_analysis.stress_bool = False
        config.out_dir = Path("./demo_output")
        
        # Validate
        config.validate()
        
        print(f"✓ Configuration loaded and modified for demo")
        print(f"Running quick analysis with {config.fault_network.n_mc} MC simulations...")
        
        # Create workflow and run quick analysis
        workflow = FaultImagingWorkflow(config)
        results = workflow.run_fault_network_only()
        
        print(f"✓ Quick analysis completed - found {len(results[2])} fault planes")
        
    except Exception as e:
        print(f"✗ Workflow example failed: {e}")


def main():
    """Run all configuration examples."""
    
    print("HyFI Configuration Management Examples")
    print("=" * 50)
    
    # Create configuration programmatically
    config = example_programmatic_config()
    
    if config:
        # Demonstrate file formats
        example_file_formats(config)
    
    # Show validation in action
    example_validation_errors()
    
    # Create templates
    example_template_creation()
    
    # Use with workflow
    if config:
        example_workflow_with_config()
    
    print("\n" + "=" * 50)
    print("Configuration examples completed!")
    print("\nNext steps:")
    print("1. Edit demo_config.yaml to customize parameters")
    print("2. Run: python examples/config_example.py")
    print("3. Or use CLI: hyfi run --config-file demo_config.yaml")


if __name__ == "__main__":
    main()
