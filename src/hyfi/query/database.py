"""
HyFI Database Interface
======================
Main class for connecting to and querying HyFI databases
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class HyFIDatabase:
    """Interface for HyFI database operations using DuckDB"""
    
    def __init__(self, database_path: Optional[str] = None, connection_type: str = 'memory'):
        """
        Initialize HyFI database connection
        
        Parameters
        ----------
        database_path : str, optional
            Path to database directory or file. If None, uses in-memory database.
        connection_type : str, default='memory'
            Type of connection: 'memory', 'file', or 'auto'
        """
        self.database_path = Path(database_path) if database_path else None
        self.connection_type = connection_type
        self.con = None
        self.tables_loaded = {}
        
        self._connect()
    
    def _connect(self):
        """Establish DuckDB connection"""
        if self.connection_type == 'memory' or self.database_path is None:
            self.con = duckdb.connect(':memory:')
        else:
            # For file-based connection
            db_file = self.database_path / 'hyfi.db' if self.database_path.is_dir() else self.database_path
            self.con = duckdb.connect(str(db_file))
        
        logger.info(f"Connected to DuckDB ({self.connection_type})")
    
    def load_csv_tables(self, database_dir: Path) -> Dict[str, bool]:
        """
        Load CSV files from HyFI_Database directory into DuckDB tables
        
        Parameters
        ----------
        database_dir : Path
            Path to HyFI_Database directory
            
        Returns
        -------
        Dict[str, bool]
            Status of loaded tables
        """
        expected_files = {
            'metadata': 'HyFI_database_metadata.csv',
            'hypocenters': 'HyFI_database_hypocenters.csv',
            'focals': 'HyFI_database_focals.csv'
        }
        
        status = {}
        
        for table_name, filename in expected_files.items():
            file_path = database_dir / filename
            
            if file_path.exists():
                try:
                    # Load CSV directly into DuckDB
                    self.con.execute(f"""
                        CREATE OR REPLACE TABLE {table_name} AS 
                        SELECT * FROM read_csv_auto('{file_path}')
                    """)
                    
                    # Verify table was created
                    count = self.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                    logger.info(f"Loaded {table_name}: {count} rows")
                    status[table_name] = True
                    self.tables_loaded[table_name] = file_path
                    
                except Exception as e:
                    logger.error(f"Failed to load {table_name} from {file_path}: {e}")
                    status[table_name] = False
            else:
                logger.warning(f"File not found: {file_path}")
                status[table_name] = False
        
        return status
    
    def query(self, sql: str, fetch: str = 'df') -> Any:
        """
        Execute SQL query
        
        Parameters
        ----------
        sql : str
            SQL query string
        fetch : str, default='df'
            Return format: 'df' for DataFrame, 'all' for all rows, 'one' for single row
            
        Returns
        -------
        pandas.DataFrame or tuple or list
            Query results in specified format
        """
        try:
            if fetch == 'df':
                return self.con.execute(sql).df()
            elif fetch == 'all':
                return self.con.execute(sql).fetchall()
            elif fetch == 'one':
                return self.con.execute(sql).fetchone()
            else:
                return self.con.execute(sql)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            logger.error(f"SQL: {sql}")
            raise
    
    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """Get information about table structure"""
        return self.query(f"DESCRIBE {table_name}")
    
    def list_tables(self) -> List[str]:
        """List all available tables"""
        result = self.query("SHOW TABLES", fetch='all')
        return [row[0] for row in result]
    
    def close(self):
        """Close database connection"""
        if self.con:
            self.con.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()