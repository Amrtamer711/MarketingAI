#!/usr/bin/env python3
"""
Startup script to ensure required directories and files exist
Run this before starting the main application
"""

import os
import shutil
from pathlib import Path
from config import EXCEL_FILE_PATH, HISTORY_DB_PATH, CREDENTIALS_PATH, DATA_DIR

def ensure_directories():
    """Create required directories if they don't exist"""
    directories = [
        "uploads",  # For temporary file uploads
        DATA_DIR,    # Create data directory based on environment
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Ensured directory exists: {directory}")

def ensure_history_database():
    """Create or update the history database with proper schema"""
    import sqlite3
    
    try:
        with sqlite3.connect(HISTORY_DB_PATH) as conn:
            # Create the table with all columns
            conn.execute("""
                CREATE TABLE IF NOT EXISTS completed_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_number INTEGER,
                    brand TEXT,
                    campaign_start_date TEXT,
                    campaign_end_date TEXT,
                    reference_number TEXT,
                    location TEXT,
                    sales_person TEXT,
                    submitted_by TEXT,
                    status TEXT,
                    filming_date TEXT,
                    videographer TEXT,
                    current_version TEXT,
                    version_history TEXT,
                    pending_timestamps TEXT,
                    submitted_timestamps TEXT,
                    returned_timestamps TEXT,
                    rejected_timestamps TEXT,
                    accepted_timestamps TEXT,
                    completed_at TEXT
                );
            """)
            
            # Create index on task_number for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_number 
                ON completed_tasks(task_number);
            """)
            
            # Create index on reference_number for duplicate checking
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reference_number 
                ON completed_tasks(reference_number);
            """)
            
            conn.commit()
            print(f"✅ History database initialized at {HISTORY_DB_PATH}")
            
    except Exception as e:
        print(f"❌ Error initializing history database: {e}")

def ensure_files():
    """Ensure required files exist"""
    # Check Excel file
    if not os.path.exists(EXCEL_FILE_PATH):
        print(f"⚠️  Excel file not found at {EXCEL_FILE_PATH}")
        # Create empty Excel file with headers
        import pandas as pd
        from excel_utils import initialize_excel
        import asyncio
        asyncio.run(initialize_excel())
        print(f"✅ Created new Excel file at {EXCEL_FILE_PATH}")
    else:
        print(f"✅ Excel file exists at {EXCEL_FILE_PATH}")
    
    # Check/Create SQLite history database
    ensure_history_database()
    
    # Check Dropbox credentials
    if not os.path.exists(CREDENTIALS_PATH):
        print(f"❌ CRITICAL: Dropbox credentials not found at {CREDENTIALS_PATH}")
        print("   Please upload your dropbox_creds.json file to the data directory")
    else:
        print(f"✅ Dropbox credentials found at {CREDENTIALS_PATH}")

def ensure_config_files():
    """Ensure required configuration files exist in data directory"""
    # Check if videographer_config.json exists in data
    config_path = f"{DATA_DIR}/videographer_config.json"
    
    if not os.path.exists(config_path):
        # Try to copy from local if exists
        local_config = "videographer_config.json"
        if os.path.exists(local_config):
            shutil.copy(local_config, config_path)
            print(f"✅ Copied {local_config} to {config_path}")
        else:
            # Create default config
            import json
            default_config = {
                "videographers": {},
                "sales_people": {},
                "location_mappings": {},
                "reviewer": {},
                "hod": {}
            }
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"✅ Created default config at {config_path}")
    else:
        print(f"✅ Config file exists at {config_path}")

def main():
    """Run all startup checks"""
    print("🚀 Running startup checks...")
    
    ensure_directories()
    ensure_files()
    ensure_config_files()
    
    print("\n✅ Startup checks complete!")

if __name__ == "__main__":
    main()