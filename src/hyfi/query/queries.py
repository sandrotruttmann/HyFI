"""
HyFI Pre-built Queries
=====================
Collection of commonly used queries for HyFI database analysis
"""

from typing import Optional, Dict, Any
import pandas as pd

class HyFIQueries:
    """Collection of pre-built queries for HyFI database analysis"""
    
    def __init__(self, database):
        """
        Initialize with HyFI database connection
        
        Parameters
        ----------
        database : HyFIDatabase
            Connected HyFI database instance
        """
        self.db = database
    
    def fault_systems_overview(self, min_events: int = 5) -> pd.DataFrame:
        """Get overview of active faults"""
        return self.db.query(f"""
            SELECT 
                fault_id,
                sequence_label,
                n_events,
                ROUND(rupture_mean_azimuth, 1) as azimuth,
                ROUND(rupture_mean_dip, 1) as dip,
                ROUND(mesh_area_m2 / 1e6, 2) as area_km2,
                ROUND(max_mag, 2) as max_mag,
                ROUND(rupture_mean_instability, 3) as instability
            FROM metadata
            WHERE n_events >= {min_events}
            ORDER BY fault_id ASC
        """)
    
    def events_by_fault_system(self) -> pd.DataFrame:
        """Get event counts and statistics by fault"""
        return self.db.query("""
            SELECT 
                fault_id,
                sequence_label,
                COUNT(*) as num_events,
                ROUND(AVG(MAG), 2) as avg_magnitude,
                ROUND(MIN(MAG), 2) as min_mag,
                ROUND(MAX(MAG), 2) as max_mag,
                COUNT(*) FILTER (WHERE rupture_plane_azimuth IS NOT NULL) as with_planes,
                COUNT(*) FILTER (WHERE instability_index IS NOT NULL) as with_stress
            FROM hypocenters
            WHERE fault_id IS NOT NULL
            GROUP BY fault_id, sequence_label
            ORDER BY num_events DESC
        """)
    
    def high_instability_faults(self, limit: int = 10) -> pd.DataFrame:
        """Get faults with highest instability"""
        return self.db.query(f"""
            SELECT 
                fault_id,
                n_events,
                ROUND(rupture_mean_instability, 3) as instability,
                ROUND(rupture_mean_sliptend, 3) as slip_tendency,
                ROUND(rupture_mean_dilatend, 3) as dilation_tendency,
                ROUND(mesh_mean_instability, 3) as mesh_instability
            FROM metadata
            WHERE rupture_mean_instability IS NOT NULL
            ORDER BY rupture_mean_instability DESC
            LIMIT {limit}
        """)
    
    def magnitude_distribution(self, bin_size: float = 0.1) -> pd.DataFrame:
        """Get magnitude distribution"""
        return self.db.query(f"""
            SELECT 
                ROUND(MAG / {bin_size}) * {bin_size} as mag_bin,
                COUNT(*) as event_count,
                COUNT(*) FILTER (WHERE fault_id IS NOT NULL) as in_fault_system
            FROM hypocenters
            GROUP BY mag_bin
            ORDER BY mag_bin
        """)
    
    def spatial_query(self, center_x: float, center_y: float, 
                     radius_km: float = 5.0, limit: int = 50) -> pd.DataFrame:
        """Get events within radius of specified coordinates"""
        radius_m = radius_km * 1000
        return self.db.query(f"""
            SELECT 
                ID,
                fault_id,
                ROUND(MAG, 2) as magnitude,
                ROUND(X, 0) as x,
                ROUND(Y, 0) as y,
                ROUND(Z, 0) as z,
                ROUND(SQRT(POWER(X - {center_x}, 2) + POWER(Y - {center_y}, 2)) / 1000, 2) as distance_km
            FROM hypocenters
            WHERE X IS NOT NULL AND Y IS NOT NULL
                AND SQRT(POWER(X - {center_x}, 2) + POWER(Y - {center_y}, 2)) < {radius_m}
            ORDER BY distance_km
            LIMIT {limit}
        """)
    
    def focal_mechanisms_summary(self) -> pd.DataFrame:
        """Get focal mechanism distribution by fault"""
        return self.db.query("""
            SELECT 
                fault_id,
                segmentation_level,
                COUNT(*) as num_focals,
                ROUND(AVG(Dip1), 1) as avg_dip1,
                ROUND(AVG(Dip2), 1) as avg_dip2,
                ROUND(AVG(Rake1), 1) as avg_rake1,
                COUNT(*) FILTER (WHERE A IS NOT NULL) as with_active_plane
            FROM focals
            WHERE is_clustered = TRUE
            GROUP BY fault_id, segmentation_level
            ORDER BY num_focals DESC
        """)
    
    def summary_statistics(self) -> pd.DataFrame:
        """Get overall database summary statistics"""
        return self.db.query("""
            SELECT 
                COUNT(*) as total_fault_systems,
                SUM(n_events) as total_events_in_faults,
                ROUND(AVG(n_events), 1) as avg_events_per_fault,
                ROUND(SUM(mesh_area_m2) / 1e6, 2) as total_area_km2,
                ROUND(AVG(rupture_mean_instability), 3) as avg_instability,
                ROUND(MAX(max_mag), 2) as max_potential_mag
            FROM metadata
        """)
    
    def depth_distribution(self, min_depth: float = -30000, max_depth: float = 0) -> pd.DataFrame:
        """Get depth distribution of events"""
        return self.db.query(f"""
            SELECT 
                ROUND(Z / 1000, 1) as depth_km,
                COUNT(*) as event_count,
                COUNT(*) FILTER (WHERE fault_id IS NOT NULL) as in_fault_system,
                ROUND(AVG(MAG), 2) as avg_magnitude
            FROM hypocenters
            WHERE Z BETWEEN {min_depth} AND {max_depth}
                AND Z IS NOT NULL
            GROUP BY ROUND(Z / 1000, 1)
            ORDER BY depth_km DESC
        """)
    
    def temporal_distribution(self, time_column: str = 'ORIGIN_TIME') -> pd.DataFrame:
        """Get temporal distribution of events (if time data available)"""
        return self.db.query(f"""
            SELECT 
                DATE_PART('year', {time_column}::DATE) as year,
                DATE_PART('month', {time_column}::DATE) as month,
                COUNT(*) as event_count,
                COUNT(*) FILTER (WHERE fault_id IS NOT NULL) as in_fault_system
            FROM hypocenters
            WHERE {time_column} IS NOT NULL
            GROUP BY year, month
            ORDER BY year, month
        """)
    
    def fault_system_details(self, fault_id: str) -> Dict[str, pd.DataFrame]:
        """Get detailed information for a specific fault"""
        results = {}
        
        # First, check if the fault ID exists in metadata
        metadata_check = self.db.query(f"""
            SELECT fault_id FROM metadata 
            WHERE fault_id = '{fault_id}'
        """)
        
        if metadata_check.empty:
            # Try to find similar IDs
            similar_ids = self.db.query("""
                SELECT DISTINCT fault_id FROM metadata 
                ORDER BY fault_id
                LIMIT 10
            """)
            results['error'] = f"Fault system '{fault_id}' not found in metadata table."
            results['available_ids'] = similar_ids
            return results
        
        # Metadata
        results['metadata'] = self.db.query(f"""
            SELECT * FROM metadata
            WHERE fault_id = '{fault_id}'
        """)
        
        # Check what fault IDs exist in hypocenters table
        hypo_ids = self.db.query(f"""
            SELECT DISTINCT fault_id FROM hypocenters 
            WHERE fault_id LIKE '%{fault_id}%'
            OR fault_id = '{fault_id}'
        """)
        
        # Events - try exact match first, then partial match
        events_exact = self.db.query(f"""
            SELECT 
                ID, MAG, X, Y, Z,
                rupture_plane_azimuth, rupture_plane_dip,
                instability_index, slip_tendency, dilation_tendency
            FROM hypocenters
            WHERE fault_id = '{fault_id}'
            ORDER BY MAG DESC
        """)
        
        if events_exact.empty and not hypo_ids.empty:
            # Try with the first matching ID from hypocenters
            actual_id = hypo_ids.iloc[0]['fault_id']
            events_exact = self.db.query(f"""
                SELECT 
                    ID, MAG, X, Y, Z,
                    rupture_plane_azimuth, rupture_plane_dip,
                    instability_index, slip_tendency, dilation_tendency
                FROM hypocenters
                WHERE fault_id = '{actual_id}'
                ORDER BY MAG DESC
            """)
            results['used_id_for_events'] = actual_id
        
        results['events'] = events_exact
        results['hypo_fault_ids'] = hypo_ids
        
        # Check what fault IDs exist in focals table  
        focal_ids = self.db.query(f"""
            SELECT DISTINCT fault_id FROM focals 
            WHERE fault_id LIKE '%{fault_id}%'
            OR fault_id = '{fault_id}'
        """)
        
        # Focal mechanisms - try exact match first, then partial match
        focals_exact = self.db.query(f"""
            SELECT * FROM focals
            WHERE fault_id = '{fault_id}'
            ORDER BY segmentation_level
        """)
        
        if focals_exact.empty and not focal_ids.empty:
            # Try with the first matching ID from focals
            actual_id = focal_ids.iloc[0]['fault_id'] 
            focals_exact = self.db.query(f"""
                SELECT * FROM focals
                WHERE fault_id = '{actual_id}'
                ORDER BY segmentation_level
            """)
            results['used_id_for_focals'] = actual_id
            
        results['focals'] = focals_exact
        results['focal_fault_ids'] = focal_ids
        
        return results