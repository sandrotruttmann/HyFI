#!/usr/bin/env python3
"""
HyFI Database Query CLI
======================
Command-line interface for database queries
"""

import click
import sys
from pathlib import Path
from .database import HyFIDatabase
from .queries import HyFIQueries

@click.group()
@click.option('--database-dir', type=click.Path(exists=True), help='HyFI database directory')
@click.pass_context
def cli(ctx, database_dir):
    """HyFI Database Query CLI"""
    ctx.ensure_object(dict)
    ctx.obj['database_dir'] = Path(database_dir) if database_dir else None

@cli.command()
@click.option('--min-events', default=5, help='Minimum number of events per fault system')
@click.pass_context
def overview(ctx, min_events):
    """Show fault systems overview"""
    db_dir = ctx.obj['database_dir']
    if not db_dir:
        click.echo("Please specify --database-dir")
        sys.exit(1)
    
    with HyFIDatabase() as db:
        db.load_csv_tables(db_dir)
        queries = HyFIQueries(db)
        
        result = queries.fault_systems_overview(min_events=min_events)
        click.echo("\\nFault Systems Overview:")
        click.echo(result.to_string(index=False))

@cli.command()
@click.pass_context
def summary(ctx):
    """Show database summary statistics"""
    db_dir = ctx.obj['database_dir']
    if not db_dir:
        click.echo("Please specify --database-dir")
        sys.exit(1)
    
    with HyFIDatabase() as db:
        db.load_csv_tables(db_dir)
        queries = HyFIQueries(db)
        
        result = queries.summary_statistics()
        click.echo("\\nDatabase Summary:")
        click.echo(result.to_string(index=False))

@cli.command()
@click.option('--limit', default=10, help='Number of top fault systems to show')
@click.pass_context
def instability(ctx, limit):
    """Show fault systems with highest instability"""
    db_dir = ctx.obj['database_dir']
    if not db_dir:
        click.echo("Please specify --database-dir")
        sys.exit(1)
    
    with HyFIDatabase() as db:
        db.load_csv_tables(db_dir)
        queries = HyFIQueries(db)
        
        result = queries.high_instability_faults(limit=limit)
        click.echo(f"\\nTop {limit} High Instability Fault Systems:")
        click.echo(result.to_string(index=False))

@cli.command()
@click.option('--x', required=True, type=float, help='X coordinate (center)')
@click.option('--y', required=True, type=float, help='Y coordinate (center)')
@click.option('--radius', default=5.0, help='Search radius in kilometers')
@click.option('--limit', default=50, help='Maximum number of events to return')
@click.pass_context
def spatial(ctx, x, y, radius, limit):
    """Find events within radius of specified coordinates"""
    db_dir = ctx.obj['database_dir']
    if not db_dir:
        click.echo("Please specify --database-dir")
        sys.exit(1)
    
    with HyFIDatabase() as db:
        db.load_csv_tables(db_dir)
        queries = HyFIQueries(db)
        
        result = queries.spatial_query(x, y, radius_km=radius, limit=limit)
        click.echo(f"\\nEvents within {radius}km of ({x}, {y}):")
        click.echo(result.to_string(index=False))

@cli.command()
@click.option('--fault-id', required=True, type=str, help='Fault system ID to analyze')
@click.pass_context
def fault_detail(ctx, fault_id):
    """Get detailed information for a specific fault system"""
    db_dir = ctx.obj['database_dir']
    if not db_dir:
        click.echo("Please specify --database-dir")
        sys.exit(1)
    
    with HyFIDatabase() as db:
        db.load_csv_tables(db_dir)
        queries = HyFIQueries(db)
        
        details = queries.fault_system_details(fault_id)
        
        # Check for errors
        if 'error' in details:
            click.echo(f"\\nError: {details['error']}")
            if 'available_ids' in details:
                click.echo("\\nAvailable fault system IDs in metadata:")
                click.echo(details['available_ids'].to_string(index=False))
            return
        
        click.echo(f"\\nFault System {fault_id} Details:")
        click.echo("\\nMetadata:")
        click.echo(details['metadata'].to_string(index=False))
        
        # Show debugging info if IDs were different
        if 'used_id_for_events' in details:
            click.echo(f"\\nNote: Used '{details['used_id_for_events']}' for events (partial match)")
        if 'used_id_for_focals' in details:
            click.echo(f"Note: Used '{details['used_id_for_focals']}' for focals (partial match)")
        
        click.echo("\\nEvents (top 10 by magnitude):")
        if not details['events'].empty:
            click.echo(details['events'].head(10).to_string(index=False))
        else:
            click.echo("No events found for this fault system ID")
        
        if 'focals' in details and not details['focals'].empty:
            click.echo("\\nFocal Mechanisms:")
            click.echo(details['focals'].to_string(index=False))
        else:
            click.echo("\\nNo focal mechanisms found for this fault system ID")

@cli.command()
@click.pass_context
def list_faults(ctx):
    """List all available fault system IDs across tables"""
    db_dir = ctx.obj['database_dir']
    if not db_dir:
        click.echo("Please specify --database-dir")
        sys.exit(1)
    
    with HyFIDatabase() as db:
        db.load_csv_tables(db_dir)
        
        click.echo("\\nFault System IDs by table:")
        
        # Metadata table
        metadata_ids = db.query("""
            SELECT DISTINCT fault_system_id, n_events 
            FROM metadata 
            ORDER BY fault_system_id
        """)
        click.echo(f"\\nMetadata table ({len(metadata_ids)} fault systems):")
        click.echo(metadata_ids.to_string(index=False))
        
        # Hypocenters table
        hypo_ids = db.query("""
            SELECT DISTINCT fault_system_id, COUNT(*) as event_count
            FROM hypocenters 
            WHERE fault_system_id IS NOT NULL
            GROUP BY fault_system_id
            ORDER BY fault_system_id
        """)
        click.echo(f"\\nHypocenters table ({len(hypo_ids)} fault systems):")
        click.echo(hypo_ids.to_string(index=False))
        
        # Focals table
        focal_ids = db.query("""
            SELECT DISTINCT fault_system_id, COUNT(*) as focal_count
            FROM focals 
            WHERE fault_system_id IS NOT NULL
            GROUP BY fault_system_id
            ORDER BY fault_system_id
        """)
        click.echo(f"\\nFocals table ({len(focal_ids)} fault systems):")
        click.echo(focal_ids.to_string(index=False))

@cli.command()
@click.option('--sql', required=True, help='Custom SQL query')
@click.pass_context
def query(ctx, sql):
    """Execute custom SQL query"""
    db_dir = ctx.obj['database_dir']
    if not db_dir:
        click.echo("Please specify --database-dir")
        sys.exit(1)
    
    with HyFIDatabase() as db:
        db.load_csv_tables(db_dir)
        result = db.query(sql)
        click.echo("\\nQuery Result:")
        click.echo(result.to_string(index=False))

@cli.command()
@click.pass_context
def tables(ctx):
    """List available tables and their structure"""
    db_dir = ctx.obj['database_dir']
    if not db_dir:
        click.echo("Please specify --database-dir")
        sys.exit(1)
    
    with HyFIDatabase() as db:
        status = db.load_csv_tables(db_dir)
        tables = db.list_tables()
        
        click.echo("\\nLoaded Tables:")
        for table in tables:
            if status.get(table, False):
                click.echo(f"✓ {table}")
                info = db.get_table_info(table)
                click.echo(info.to_string(index=False))
                click.echo("")
            else:
                click.echo(f"✗ {table} (failed to load)")

if __name__ == '__main__':
    cli()