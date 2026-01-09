#!/usr/bin/env python3
"""
HyFI Database Query Entry Point
==============================
CLI entry point for HyFI database queries
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hyfi.query.cli import cli

if __name__ == '__main__':
    cli()