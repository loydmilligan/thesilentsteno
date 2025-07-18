#!/usr/bin/env python3
"""
Database Initialization Script for The Silent Steno

This script creates the database tables and initializes the database
with the proper schema. It should be run once to set up the database.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import initialize_database, get_database
from src.data.models import create_models

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Initialize the database with all required tables"""
    try:
        logger.info("Starting database initialization...")
        
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Initialize database with proper schema
        database_path = "data/silent_steno.db"
        logger.info(f"Initializing database at: {database_path}")
        
        database = initialize_database(database_path)
        
        # Verify database is working
        logger.info("Testing database connection...")
        with database.get_session() as session:
            # Test basic query
            result = session.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            logger.info(f"Database tables created: {[row[0] for row in result]}")
        
        # Check database health
        if database.health_check():
            logger.info("Database health check: PASSED")
        else:
            logger.error("Database health check: FAILED")
            return False
        
        # Print database status
        status = database.get_status()
        logger.info(f"Database status: {status}")
        
        logger.info("Database initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)