#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command Line Interface for HyFI: Hypocenter-based 3D Imaging of Active Faults

This module provides a command-line interface for running fault imaging analyses.
"""

import click
import sys
from pathlib import Path

from .config.parameters import ProjectConfig
from .config.io import auto_load_config, create_template_config
from .config.validation import ConfigValidationError
from .core.workflow import FaultImagingWorkflow
from .utils.input_validation import validate_input_files


@click.group()
@click.version_option()
def main():
    """
    HyFI: Hypocenter-based 3D Imaging of Active Faults
    
    A tool for 3D imaging of active faults based on relatively relocated 
    hypocenter datasets.
    """
    pass


@main.command()
@click.option('--config-file', '-c', type=click.Path(exists=True),
              help='Path to configuration file (YAML, JSON, or INI)')
@click.option('--output-dir', '-o', type=click.Path(),
              help='Output directory (overrides config file)')
@click.option('--interpolate-planes', is_flag=True,
              help='Enable Poisson surface reconstruction for fault plane interpolation')
def run(config_file, output_dir, interpolate_planes):
    """
    Run analysis from a configuration file.
    
    This command loads parameters from a configuration file and runs the analysis.
    Supports YAML, JSON, and INI formats.
    """
    
    if not config_file:
        click.echo("Error: Configuration file is required", err=True)
        click.echo("Use 'hyfi config create' to generate a template")
        sys.exit(1)
    
    try:
        # Load configuration from file
        config = auto_load_config(config_file)
        
        # Override output directory if provided
        if output_dir:
            if hasattr(config, 'out_dir'):
                config.out_dir = Path(output_dir)
            elif hasattr(config, 'output_directory'):
                config.output_directory = str(output_dir)
        
        # Override interpolation setting if flag provided
        if interpolate_planes:
            if hasattr(config, 'nodes') and 'visualization' in config.nodes:
                # DAG configuration
                config.nodes['visualization'].parameters['enable_plane_interpolation'] = True
            elif hasattr(config, 'visualization') and hasattr(config.visualization, 'parameters'):
                # Legacy configuration with separate visualization config
                config.visualization.parameters['enable_plane_interpolation'] = True
            else:
                click.echo("Warning: Could not enable interpolation - configuration format not supported")
        
        # Validate configuration
        if hasattr(config, 'validate_dag'):
            # DAG configuration
            config.validate_dag()
        elif hasattr(config, 'validate'):
            # Legacy configuration
            config.validate()
        
        # Run analysis based on config type
        from .config.schema import HyFIWorkflowDAG
        from .config.multi_sequence_config import MultiSequenceConfig
        
        if isinstance(config, HyFIWorkflowDAG):
            # Run DAG-based workflow
            from .core.dag_executor import DAGExecutor
            executor = DAGExecutor(config, config_source_file=config_file)
            results = executor.execute()
            summary = executor.get_execution_summary()
        elif isinstance(config, MultiSequenceConfig):
            # Run multi-sequence workflow
            from .core.multi_sequence_workflow import MultiSequenceWorkflow
            workflow = MultiSequenceWorkflow(config)
            results = workflow.run_full_multi_sequence_analysis()
            summary = workflow.get_multi_sequence_summary()
        else:
            # Run legacy workflow
            workflow = FaultImagingWorkflow(config)
            results = workflow.run_full_analysis()
            summary = workflow.get_results_summary()
        
        # Print summary
        summary = summary or {}
        click.echo("\n\n" + "="*50)
        click.echo("ANALYSIS SUMMARY")
        click.echo("="*50)
        
        # Print specific fields explicitly with formatting
        if 'status' in summary:
            click.echo(f"Status: {summary['status']}")
        
        if 'nodes_executed' in summary:
            click.echo(f"Nodes executed: {summary['nodes_executed']}")
        
        if 'nodes_failed' in summary:
            click.echo(f"Nodes failed: {summary['nodes_failed']}")
        
        if 'output_directory' in summary:
            click.echo(f"Output directory: {summary['output_directory']}")
        
        if 'total_duration' in summary:
            # Convert duration from seconds to min:sec format
            duration_seconds = summary['total_duration']
            if duration_seconds>= 60:
                minutes = int(duration_seconds // 60)
                seconds = int(duration_seconds % 60)
                click.echo(f"Total duration: {minutes}:{seconds:02d} min")
            else:
                click.echo(f"Total duration: {duration_seconds:.2f} s")

        # Print any other fields that weren't explicitly handled
        excluded_fields = {'status', 'nodes_executed', 'nodes_failed', 'output_directory', 'total_duration', 'workflow_name'}
        for key, value in summary.items():
            if key not in excluded_fields:
                click.echo(f"{key}: {value}")
        
        click.echo("\n\nANALYSIS COMPLETED SUCCESSFULLY.")
        
    except ConfigValidationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.group()
def config():
    """Configuration file management commands."""
    pass


@config.command()
@click.option('--output', '-o', type=click.Path(), default='hyfi_multi_sequence_config.json',
              help='Output file path for multi-sequence configuration')
@click.option('--workflow-name', default='Multi-Sequence HyFI Analysis',
              help='Name for the analysis workflow')
@click.option('--catalog-file', default='path/to/full_catalog.txt',
              help='Path to full earthquake catalog file')
@click.option('--output-dir', default='./multi_sequence_output',
              help='Output directory for analysis results')
@click.option('--clustering-method', type=click.Choice(['dbscan', 'hdbscan', 'temporal', 'spatial_temporal']),
              default='dbscan', help='Primary clustering method for catalog segmentation')
@click.option('--multi-step', is_flag=True,
              help='Create multi-step segmentation template with cascading parameters')
@click.option('--advanced', is_flag=True,
              help='Create advanced multi-step template with 5 segmentation steps')
def create_multi_sequence(output, workflow_name, catalog_file, output_dir, clustering_method, multi_step, advanced):
    """Create a template multi-sequence configuration file."""
    
    try:
        from .config.multi_sequence_templates import (create_multi_sequence_template, 
                                                      create_advanced_multi_step_template)
        
        if advanced:
            config = create_advanced_multi_step_template(
                output_file=output,
                catalog_file=catalog_file,
                project_title=workflow_name,
                output_dir=output_dir
            )
            template_type = "advanced multi-step (5 steps)"
        else:
            config = create_multi_sequence_template(
                output_file=output,
                catalog_file=catalog_file,
                project_title=workflow_name,
                output_dir=output_dir,
                clustering_method=clustering_method,
                multi_step=multi_step
            )
            template_type = "multi-step" if multi_step else "single-step"
        
        click.echo(f"✅ Multi-sequence configuration template created: {output}")
        click.echo(f"📋 Template type: {template_type} segmentation")
        click.echo(f"🔧 Number of segmentation steps: {len(config.segmentation_steps)}")
        
        for i, step in enumerate(config.segmentation_steps):
            click.echo(f"   Step {i+1}: {step.step_name} ({step.method})")
        
        click.echo("🔧 Edit the file to customize clustering parameters and workflow settings.")
        click.echo(f"🚀 Then run: hyfi multi-sequence --template-config {output}")
        click.echo(f"🚀 Or run: hyfi run --config-file {output}")
        
    except Exception as e:
        click.echo(f"❌ Error creating multi-sequence template: {e}", err=True)
        sys.exit(1)


@config.command()
@click.option('--output', '-o', type=click.Path(), default='hyfi_config.json',
              help='Output file path')
@click.option('--workflow-name', default='My HyFI Analysis',
              help='Name for the analysis workflow')
@click.option('--input-file', default='path/to/hypocenter_data.txt',
              help='Path to input hypocenter data file')
@click.option('--output-dir', default='./hyfi_output',
              help='Output directory for analysis results')
def create(output, workflow_name, input_file, output_dir):
    """Create a template configuration file."""
    
    try:
        # Create DAG configuration (JSON format)
        from .config.io import create_template_dag
        dag = create_template_dag(
            file_path=output,
            workflow_name=workflow_name,
            input_file=input_file,
            output_dir=output_dir
        )
        click.echo(f"✅ DAG configuration template created: {output}")
        click.echo("📋 This JSON file represents your analysis as a directed acyclic graph.")
        click.echo("🔧 Edit the file to customize parameters and workflow structure.")
        click.echo(f"🚀 Then run: hyfi run --config-file {output}")
        
    except Exception as e:
        click.echo(f"❌ Error creating template: {e}", err=True)
        sys.exit(1)


@config.command()
@click.argument('config_file', type=click.Path(exists=True))
def validate(config_file):
    """Validate a configuration file."""
    
    try:
        config = auto_load_config(config_file)
        
        # Check if it's a DAG configuration
        from .config.schema import HyFIWorkflowDAG
        if isinstance(config, HyFIWorkflowDAG):
            # Validate DAG structure
            config.validate_dag()
            
            # Show execution order
            execution_order = config.get_execution_order()
            enabled_nodes = config.get_enabled_nodes()
            
            click.echo(f"✅ DAG configuration is valid: {config_file}")
            click.echo(f"📊 Workflow: {config.workflow_name}")
            click.echo(f"🔄 Execution order: {' → '.join(execution_order)}")
            click.echo(f"✔️  Enabled nodes: {', '.join(enabled_nodes)}")
            click.echo(f"📁 Output directory: {config.output_directory}")
            
        else:
            # Legacy configuration
            config.validate()
            click.echo(f"✅ Configuration file is valid: {config_file}")
            
            # Print summary
            click.echo(f"\n📋 Configuration Summary:")
            click.echo(f"Project: {config.project_title}")
            click.echo(f"Input file: {config.hypo_file}")
            click.echo(f"Output directory: {config.out_dir}")
            click.echo(f"Monte Carlo simulations: {config.fault_network.n_mc}")
            click.echo(f"Search radius: {config.fault_network.r_nn} m")
            click.echo(f"Time window: {config.fault_network.dt_nn} h")
            click.echo(f"Validation enabled: {config.model_validation.validation_bool}")
            click.echo(f"Classification enabled: {config.auto_class.autoclass_bool}")
            click.echo(f"Stress analysis enabled: {config.stress_analysis.stress_bool}")
            click.echo("💡 Consider migrating to JSON DAG format for better workflow management.")
        
    except Exception as e:
        click.echo(f"❌ Configuration validation failed: {e}", err=True)
        sys.exit(1)


@config.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', default=None,
              help='Output file path (default: input_file.json)')
def convert(input_file, output):
    """Convert configuration to DAG format."""
    
    try:
        from .config.io import auto_load_config, save_dag_to_json, convert_legacy_to_dag
        from .config.schema import HyFIWorkflowDAG
        from pathlib import Path
        
        # Load source configuration
        config = auto_load_config(input_file)
        
        # Determine output path
        if output is None:
            input_path = Path(input_file)
            output = input_path.with_suffix('.json')
        
        if isinstance(config, HyFIWorkflowDAG):
            # Already a DAG
            save_dag_to_json(config, output)
            click.echo(f"✅ DAG configuration saved: {output}")
        else:
            # Convert legacy to DAG
            dag = convert_legacy_to_dag(config)
            save_dag_to_json(dag, output)
            click.echo(f"✅ Converted to DAG format: {output}")
            click.echo("🎯 The configuration is now structured as a workflow DAG.")
            click.echo("🔧 You can now edit the JSON file to customize the workflow structure.")
        
    except Exception as e:
        click.echo(f"❌ Error converting configuration: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--hypo-file', '-h', required=True, type=click.Path(exists=True),
              help='Path to hypocenter data file (hypoDD format)')
@click.option('--output-dir', '-o', type=click.Path(), default='.',
              help='Output directory for results')
@click.option('--project-title', '-t', default='HyFI Analysis',
              help='Project title for the analysis')
@click.option('--separator', '-s', default='\t',
              help='Column separator in hypocenter file')
@click.option('--focal-file', '-f', type=click.Path(exists=True),
              help='Path to focal mechanism file')
@click.option('--focal-sep', default=';',
              help='Column separator in focal mechanism file')
@click.option('--n-mc', default=1000, type=int,
              help='Number of Monte Carlo simulations')
@click.option('--r-nn', default=100.0, type=float,
              help='Search radius [m] for nearest neighbor search')
@click.option('--dt-nn', default=26298.0, type=float,
              help='Search time window [h]')
@click.option('--n-clusters', default=2, type=int,
              help='Number of expected fault clusters')
@click.option('--no-validation', is_flag=True,
              help='Skip model validation with focal mechanisms')
@click.option('--no-classification', is_flag=True,
              help='Skip automatic classification')
@click.option('--no-stress', is_flag=True,
              help='Skip stress analysis')
def analyze(hypo_file, output_dir, project_title, separator, focal_file, focal_sep,
           n_mc, r_nn, dt_nn, n_clusters, no_validation, no_classification, no_stress):
    """
    Run a complete fault imaging analysis.
    
    This command runs the full HyFI workflow including fault network reconstruction,
    model validation, automatic classification, stress analysis, and visualization.
    """
    
    try:
        # Create configuration
        config = ProjectConfig(
            project_title=project_title,
            hypo_file=Path(hypo_file),
            hypo_sep=separator,
            out_dir=Path(output_dir)
        )
        
        # Configure modules
        config.fault_network.n_mc = n_mc
        config.fault_network.r_nn = r_nn  
        config.fault_network.dt_nn = dt_nn
        
        config.model_validation.validation_bool = not no_validation
        if focal_file:
            config.model_validation.foc_file = Path(focal_file)
            config.model_validation.foc_sep = focal_sep
        
        config.auto_class.autoclass_bool = not no_classification
        config.auto_class.n_clusters = n_clusters
        
        config.stress_analysis.stress_bool = not no_stress
        
        # Validate configuration
        config.validate()
        
        # Run analysis
        workflow = FaultImagingWorkflow(config)
        results = workflow.run_full_analysis()
        
        # Print summary
        summary = workflow.get_results_summary()
        click.echo("\n" + "="*50)
        click.echo("ANALYSIS SUMMARY")
        click.echo("="*50)
        for key, value in summary.items():
            click.echo(f"{key}: {value}")
        
        click.echo(f"\nResults saved to: {output_dir}")
        click.echo("Analysis completed successfully!")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--hypo-file', '-h', required=True, type=click.Path(exists=True),
              help='Path to hypocenter data file')
@click.option('--output-dir', '-o', type=click.Path(), default='.',
              help='Output directory for results')
@click.option('--separator', '-s', default='\t',
              help='Column separator in hypocenter file')
@click.option('--n-mc', default=100, type=int,
              help='Number of Monte Carlo simulations (reduced for quick analysis)')
@click.option('--r-nn', default=100.0, type=float,
              help='Search radius [m] for nearest neighbor search')
@click.option('--dt-nn', default=26298.0, type=float,
              help='Search time window [h]')
def quick(hypo_file, output_dir, separator, n_mc, r_nn, dt_nn):
    """
    Run a quick fault network reconstruction (no validation or stress analysis).
    
    This command runs only the fault network reconstruction module for quick results.
    """
    
    try:
        # Create minimal configuration
        config = ProjectConfig(
            project_title='HyFI Quick Analysis',
            hypo_file=Path(hypo_file),
            hypo_sep=separator,
            out_dir=Path(output_dir)
        )
        
        # Configure for quick analysis
        config.fault_network.n_mc = n_mc
        config.fault_network.r_nn = r_nn
        config.fault_network.dt_nn = dt_nn
        
        # Disable other modules
        config.model_validation.validation_bool = False
        config.auto_class.autoclass_bool = False
        config.stress_analysis.stress_bool = False
        
        # Run analysis
        workflow = FaultImagingWorkflow(config)
        results = workflow.run_fault_network_only()
        
        click.echo(f"Quick analysis found {len(results[2])} fault planes")
        click.echo(f"Results saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--catalog-file', '-c', required=True, type=click.Path(exists=True),
              help='Path to full earthquake catalog file')
@click.option('--output-dir', '-o', type=click.Path(), default='./multi_sequence_output',
              help='Output directory for multi-sequence analysis')
@click.option('--project-title', '-t', default='Multi-Sequence Analysis',
              help='Project title for the analysis')
@click.option('--separator', '-s', default='\t',
              help='Column separator in catalog file')
@click.option('--template-config', type=click.Path(exists=True),
              help='Template configuration file for individual cluster processing')
# Primary clustering step options
@click.option('--primary-method', type=click.Choice(['dbscan', 'hdbscan', 'temporal', 'spatial_temporal']),
              default='dbscan', help='Primary clustering method')
@click.option('--primary-eps', default=1000.0, type=float,
              help='Primary DBSCAN eps parameter')
@click.option('--primary-min-samples', default=10, type=int,
              help='Primary minimum samples per cluster')
@click.option('--primary-min-cluster-size', default=20, type=int,
              help='Primary minimum cluster size to process')
# Secondary clustering step options
@click.option('--secondary-method', type=click.Choice(['dbscan', 'hdbscan', 'temporal', 'spatial_temporal']),
              help='Secondary clustering method for outliers (optional)')
@click.option('--secondary-eps', default=500.0, type=float,
              help='Secondary DBSCAN eps parameter')
@click.option('--secondary-min-samples', default=5, type=int,
              help='Secondary minimum samples per cluster')
@click.option('--secondary-min-cluster-size', default=10, type=int,
              help='Secondary minimum cluster size to process')
# Template workflow options
@click.option('--n-mc', default=1000, type=int,
              help='Number of Monte Carlo simulations for each cluster')
@click.option('--r-nn', default=100.0, type=float,
              help='Search radius [m] for nearest neighbor search')
@click.option('--no-validation', is_flag=True,
              help='Skip model validation with focal mechanisms')
@click.option('--no-classification', is_flag=True,
              help='Skip automatic classification')
@click.option('--no-stress', is_flag=True,
              help='Skip stress analysis')
def multi_sequence(catalog_file, output_dir, project_title, separator, template_config,
                  primary_method, primary_eps, primary_min_samples, 
                  primary_min_cluster_size, secondary_method, secondary_eps, 
                  secondary_min_samples, secondary_min_cluster_size, n_mc, r_nn, 
                  no_validation, no_classification, no_stress):
    """
    Run multi-sequence fault imaging analysis on a full earthquake catalog.
    
    This command segments a full earthquake catalog into clusters using multi-step
    clustering and applies the standard HyFI workflow to each cluster individually.
    """
    
    try:
        from .config.multi_sequence_config import MultiSequenceConfig, ClusteringConfig, SegmentationStep
        from .core.multi_sequence_workflow import MultiSequenceWorkflow
        
        # Load template configuration if provided
        if template_config:
            template_proj_config = auto_load_config(template_config)
        else:
            # Create default template configuration
            template_proj_config = ProjectConfig(
                project_title="Template Config",
                hypo_file=Path(""),  # Will be set dynamically for each cluster
                hypo_sep=separator,
                out_dir=Path("")  # Will be set dynamically for each cluster
            )
            
            # Configure template based on CLI options
            template_proj_config.fault_network.n_mc = n_mc
            template_proj_config.fault_network.r_nn = r_nn
            template_proj_config.model_validation.validation_bool = not no_validation
            template_proj_config.auto_class.autoclass_bool = not no_classification
            template_proj_config.stress_analysis.stress_bool = not no_stress
        
        # Create segmentation steps
        segmentation_steps = []
        
        # Primary clustering step
        primary_step = SegmentationStep(
            step_name="primary_clustering",
            description=f"Primary {primary_method} clustering",
            method=primary_method,
            features=['spatial'],
            dbscan_eps=primary_eps,
            dbscan_min_samples=primary_min_samples,
            min_cluster_size=primary_min_cluster_size,
            process_outliers=bool(secondary_method),  # Process outliers if secondary method specified
            outlier_handling='next_step' if secondary_method else 'discard'
        )
        segmentation_steps.append(primary_step)
        
        # Secondary clustering step (if specified)
        if secondary_method:
            secondary_step = SegmentationStep(
                step_name="secondary_clustering",
                description=f"Secondary {secondary_method} clustering for outliers",
                method=secondary_method,
                features=['spatial'],
                dbscan_eps=secondary_eps,
                dbscan_min_samples=secondary_min_samples,
                min_cluster_size=secondary_min_cluster_size,
                process_outliers=False,  # Final step
                outlier_handling='discard'
            )
            segmentation_steps.append(secondary_step)
        
        # Create clustering configuration
        clustering_config = ClusteringConfig(
            segmentation_steps=segmentation_steps,
            final_outlier_handling='keep',
            max_outlier_ratio=0.3
        )
        
        # Create multi-sequence configuration
        config = MultiSequenceConfig(
            project_title=project_title,
            catalog_file=Path(catalog_file),
            catalog_sep=separator,
            output_directory=Path(output_dir),
            clustering=clustering_config,
            template_config=template_proj_config
        )
        
        # Validate configuration
        config.validate()
        
        # Print configuration summary
        click.echo(f"\n📋 Multi-Sequence Configuration:")
        click.echo(f"   Catalog: {catalog_file}")
        click.echo(f"   Segmentation steps: {len(segmentation_steps)}")
        for i, step in enumerate(segmentation_steps):
            click.echo(f"     Step {i+1}: {step.step_name} ({step.method})")
        click.echo(f"   Output: {output_dir}")
        
        # Run multi-sequence analysis
        workflow = MultiSequenceWorkflow(config)
        results = workflow.run_full_multi_sequence_analysis()
        
        # Print summary
        summary = workflow.get_multi_sequence_summary()
        click.echo("\n" + "="*60)
        click.echo("MULTI-SEQUENCE ANALYSIS SUMMARY")
        click.echo("="*60)
        for key, value in summary.items():
            click.echo(f"{key}: {value}")
        
        click.echo(f"\nResults saved to: {output_dir}")
        click.echo("Multi-sequence analysis completed successfully!")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--hypo-file', '-h', required=True, type=click.Path(exists=True),
              help='Path to hypocenter data file')
@click.option('--hypo-sep', default='\t',
              help='Column separator in hypocenter file')
@click.option('--focal-file', '-f', type=click.Path(exists=True),
              help='Path to focal mechanism file (optional)')
@click.option('--focal-sep', default=';',
              help='Column separator in focal mechanism file')
@click.option('--no-report', is_flag=True,
              help='Skip printing detailed report')
def validate_input(hypo_file, hypo_sep, focal_file, focal_sep, no_report):
    """Validate input data files for correct format and columns."""
    try:
        results = validate_input_files(
            hypo_file=hypo_file,
            hypo_sep=hypo_sep,
            focal_file=focal_file,
            focal_sep=focal_sep,
            print_report=not no_report
        )
        
        if results['overall_valid']:
            click.echo("\n✓ All input files are valid and ready for processing!")
            sys.exit(0)
        else:
            click.echo("\n✗ Input file validation failed. Please fix the issues above.")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error during validation: {e}", err=True)
        sys.exit(1)


@main.command()
def info():
    """Display information about HyFI."""
    
    from . import __version__, __author__, __email__
    
    click.echo("HyFI: Hypocenter-based 3D Imaging of Active Faults")
    click.echo(f"Version: {__version__}")
    click.echo(f"Author: {__author__}")
    click.echo(f"Contact: {__email__}")
    click.echo("")
    click.echo("Scientific Publication:")
    click.echo("Truttmann, S., Diehl, T., Herwegh, M. (2023).")
    click.echo("Hypocenter-based 3D Imaging of Active Faults: Method and Applications")
    click.echo("in the Southwestern Swiss Alps. Journal of Geophysical Research: Solid Earth")
    click.echo("https://doi.org/10.1029/2023JB026352")


if __name__ == '__main__':
    main()
