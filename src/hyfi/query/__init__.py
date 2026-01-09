"""
HyFI Database Query Module
=========================
Query interface for HyFI database analysis using DuckDB
"""

from .database import HyFIDatabase
from .queries import HyFIQueries

__all__ = ['HyFIDatabase', 'HyFIQueries']