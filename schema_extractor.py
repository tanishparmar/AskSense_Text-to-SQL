"""
Schema Extractor Module

Handles CSV to SQLite conversion and database schema extraction.
"""

import sqlite3
import pandas as pd
import tempfile
import os
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SchemaExtractor:
    """Handles database schema extraction and CSV to SQLite conversion."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def csvs_to_sqlite(self, uploaded_files: List) -> str:
        """
        Convert uploaded CSV files to a SQLite database.
        
        Args:
            uploaded_files: List of uploaded CSV files
            
        Returns:
            str: Path to the created SQLite database
        """
        try:
            # Create a temporary SQLite database
            db_path = os.path.join(self.temp_dir, "user_data.sqlite")
            
            # Connect to SQLite database
            conn = sqlite3.connect(db_path)
            
            for uploaded_file in uploaded_files:
                # Read CSV file
                df = pd.read_csv(uploaded_file)
                
                # Clean column names (remove spaces, special characters)
                df.columns = [col.replace(' ', '_').replace('-', '_').replace('.', '_') 
                             for col in df.columns]
                
                # Write to SQLite
                table_name = uploaded_file.name.replace('.csv', '').replace(' ', '_')
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                
                logger.info(f"Successfully imported {uploaded_file.name} as table '{table_name}'")
            
            conn.close()
            return db_path
            
        except Exception as e:
            logger.error(f"Error converting CSV files to SQLite: {str(e)}")
            raise
    
    def extract_schema(self, sqlite_path: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Extract schema information from SQLite database.
        
        Args:
            sqlite_path: Path to SQLite database
            
        Returns:
            Dict mapping table names to list of (column_name, column_type) tuples
        """
        try:
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema = {}
            for table in tables:
                table_name = table[0]
                
                # Get column information
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                # Extract column name and type
                column_info = []
                for col in columns:
                    col_name = col[1]
                    col_type = col[2]
                    column_info.append((col_name, col_type))
                
                schema[table_name] = column_info
            
            conn.close()
            return schema
            
        except Exception as e:
            logger.error(f"Error extracting schema: {str(e)}")
            raise
    
    def create_detailed_schema_info(self, schema: Dict[str, List[Tuple[str, str]]]) -> str:
        """
        Create detailed schema information for the model.
        
        Args:
            schema: Schema dictionary from extract_schema()
            
        Returns:
            str: Formatted schema information
        """
        schema_parts = []
        
        for table_name, columns in schema.items():
            # Create detailed table description
            column_info = []
            for col_name, col_type in columns:
                column_info.append(f"{col_name} ({col_type})")
            
            table_info = f"Table '{table_name}' with columns: {', '.join(column_info)}"
            schema_parts.append(table_info)
        
        return "Database Schema:\n" + "\n".join(schema_parts)
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


# Convenience functions for backward compatibility
def csvs_to_sqlite(uploaded_files: List) -> str:
    """Convert uploaded CSV files to SQLite database."""
    extractor = SchemaExtractor()
    return extractor.csvs_to_sqlite(uploaded_files)


def extract_schema(sqlite_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """Extract schema from SQLite database."""
    extractor = SchemaExtractor()
    return extractor.extract_schema(sqlite_path) 