#!/usr/bin/env python3
"""
HyFI Database Query Examples
===========================
Examples of using the HyFI query system
"""

from pathlib import Path
from hyfi.query import HyFIDatabase, HyFIQueries

def basic_example():
    """Example 1: Basic database connection and queries"""
    database_dir = Path('output_SECOS_VS_filtered/HyFI_Database')
    
    print("=== Basic HyFI Database Query Example ===")
    
    with HyFIDatabase() as db:
        # Load CSV tables
        status = db.load_csv_tables(database_dir)
        print("Loaded tables:", status)
        
        # Initialize query interface
        queries = HyFIQueries(db)
        
        # Get fault systems overview
        print("\\n1. Fault Systems Overview (min 10 events):")
        fault_overview = queries.fault_systems_overview(min_events=10)
        print(fault_overview.to_string(index=False))
        
        # Get summary statistics
        print("\\n2. Database Summary:")
        summary = queries.summary_statistics()
        print(summary.to_string(index=False))
        
        # Get high instability faults
        print("\\n3. Top 5 High Instability Fault Systems:")
        high_instability = queries.high_instability_faults(limit=5)
        print(high_instability.to_string(index=False))

def analysis_example():
    """Example 2: Detailed analysis queries"""
    database_dir = Path('output_SECOS_VS_filtered/HyFI_Database')
    
    print("\\n=== Detailed Analysis Example ===")
    
    with HyFIDatabase() as db:
        db.load_csv_tables(database_dir)
        queries = HyFIQueries(db)
        
        # Events by fault system
        print("\\n1. Events by Fault System:")
        events_by_fault = queries.events_by_fault_system()
        print(events_by_fault.head(10).to_string(index=False))
        
        # Magnitude distribution
        print("\\n2. Magnitude Distribution:")
        mag_dist = queries.magnitude_distribution(bin_size=0.2)
        print(mag_dist.to_string(index=False))
        
        # Depth distribution
        print("\\n3. Depth Distribution:")
        depth_dist = queries.depth_distribution()
        print(depth_dist.head(10).to_string(index=False))

def custom_query_example():
    """Example 3: Custom analysis queries"""
    database_dir = Path('output_SECOS_VS_filtered/HyFI_Database')
    
    print("\\n=== Custom Query Example ===")
    
    with HyFIDatabase() as db:
        db.load_csv_tables(database_dir)
        
        # Custom query: fault systems with high average magnitude
        print("\\n1. Fault Systems with High Average Magnitude:")
        result = db.query("""
            SELECT 
                fault_system_id, 
                COUNT(*) as num_events,
                ROUND(AVG(MAG), 2) as avg_mag,
                ROUND(MIN(MAG), 2) as min_mag,
                ROUND(MAX(MAG), 2) as max_mag
            FROM hypocenters 
            WHERE fault_system_id IS NOT NULL
            GROUP BY fault_system_id
            HAVING AVG(MAG) > 2.0
            ORDER BY avg_mag DESC
        """)
        print(result.to_string(index=False))
        
        # Custom query: events with stress analysis data
        print("\\n2. Events with Stress Analysis Data:")
        stress_events = db.query("""
            SELECT 
                fault_system_id,
                COUNT(*) as total_events,
                COUNT(*) FILTER (WHERE instability_index IS NOT NULL) as with_instability,
                COUNT(*) FILTER (WHERE slip_tendency IS NOT NULL) as with_slip_tend,
                ROUND(AVG(instability_index), 3) as avg_instability
            FROM hypocenters
            WHERE fault_system_id IS NOT NULL
            GROUP BY fault_system_id
            ORDER BY with_instability DESC
        """)
        print(stress_events.head(10).to_string(index=False))

def spatial_query_example():
    """Example 4: Spatial queries"""
    database_dir = Path('output_SECOS_VS_filtered/HyFI_Database')
    
    print("\\n=== Spatial Query Example ===")
    
    with HyFIDatabase() as db:
        db.load_csv_tables(database_dir)
        queries = HyFIQueries(db)
        
        # Get coordinate range first
        coord_info = db.query("""
            SELECT 
                MIN(X) as min_x, MAX(X) as max_x,
                MIN(Y) as min_y, MAX(Y) as max_y,
                COUNT(*) as total_events
            FROM hypocenters
            WHERE X IS NOT NULL AND Y IS NOT NULL
        """)
        print("\\n1. Coordinate Range:")
        print(coord_info.to_string(index=False))
        
        # Use center coordinates for spatial query
        if not coord_info.empty:
            center_x = (coord_info.iloc[0]['min_x'] + coord_info.iloc[0]['max_x']) / 2
            center_y = (coord_info.iloc[0]['min_y'] + coord_info.iloc[0]['max_y']) / 2
            
            print(f"\\n2. Events within 5km of center ({center_x:.0f}, {center_y:.0f}):")
            spatial_result = queries.spatial_query(center_x, center_y, radius_km=5.0, limit=20)
            print(spatial_result.to_string(index=False))

def fault_detail_example():
    """Example 5: Detailed fault system analysis"""
    database_dir = Path('output_SECOS_VS_filtered/HyFI_Database')
    
    print("\\n=== Fault System Detail Example ===")
    
    with HyFIDatabase() as db:
        db.load_csv_tables(database_dir)
        queries = HyFIQueries(db)
        
        # Get the fault system with most events
        fault_systems = queries.events_by_fault_system()
        if not fault_systems.empty:
            top_fault_id = fault_systems.iloc[0]['fault_system_id']
            
            print(f"\\nDetailed analysis of Fault System {top_fault_id}:")
            details = queries.fault_system_details(top_fault_id)
            
            print("\\n1. Metadata:")
            print(details['metadata'].to_string(index=False))
            
            print(f"\\n2. Top 5 Events (by magnitude):")
            print(details['events'].head(5).to_string(index=False))
            
            if not details['focals'].empty:
                print("\\n3. Focal Mechanisms:")
                print(details['focals'].to_string(index=False))

def main():
    """Run all examples"""
    try:
        basic_example()
        analysis_example()
        custom_query_example()
        spatial_query_example()
        fault_detail_example()
        
        print("\\n=== All Examples Completed Successfully ===")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure the HyFI_Database directory exists with CSV files.")

if __name__ == '__main__':
    main()