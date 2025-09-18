#!/usr/bin/env python3
"""
Migration 001: Remove UNIQUE constraint from Reference Number column

This migration removes the UNIQUE constraint from the Reference Number column
in the live_tasks table. The application already has duplicate detection and 
allows users to proceed with duplicates after confirmation, so the database
constraint is no longer needed and actually conflicts with the app logic.
"""

import sqlite3
import os
import sys
import shutil
from datetime import datetime

# Add parent directory to path to import logger
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import logger

# Use local history_logs.db in the migrate directory
HISTORY_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history_logs.db")

def backup_database():
    """Create a backup of the database before migration."""
    backup_path = f"{HISTORY_DB_PATH}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copy2(HISTORY_DB_PATH, backup_path)
        logger.info(f"Created backup at: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        raise

def check_current_schema(conn):
    """Check if UNIQUE constraint exists on Reference Number."""
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='live_tasks'")
    table_def = cursor.fetchone()
    if not table_def:
        raise Exception("live_tasks table not found!")
    
    sql = table_def[0]
    # Check if "Reference Number" TEXT UNIQUE exists
    has_unique = '"Reference Number" TEXT UNIQUE' in sql
    logger.info(f"Current schema has UNIQUE constraint: {has_unique}")
    return has_unique

def migrate():
    """Remove UNIQUE constraint from Reference Number column."""
    
    logger.info("Starting migration 001: Remove Reference Number UNIQUE constraint")
    
    # Create backup
    backup_path = backup_database()
    
    conn = None
    try:
        conn = sqlite3.connect(HISTORY_DB_PATH)
        conn.execute("PRAGMA foreign_keys=OFF")
        
        # Check current schema
        if not check_current_schema(conn):
            logger.info("UNIQUE constraint not found - migration already applied or not needed")
            return True
        
        # Begin transaction
        conn.execute("BEGIN EXCLUSIVE")
        
        # Get current max task_number for sequence preservation
        max_task = conn.execute("SELECT MAX(task_number) FROM live_tasks").fetchone()[0] or 0
        logger.info(f"Current max task_number: {max_task}")
        
        # Create new table without UNIQUE constraint
        logger.info("Creating new table without UNIQUE constraint...")
        conn.execute("""
            CREATE TABLE live_tasks_new (
                task_number INTEGER PRIMARY KEY AUTOINCREMENT,
                "Timestamp" TEXT,
                "Brand" TEXT,
                "Campaign Start Date" TEXT,
                "Campaign End Date" TEXT,
                "Reference Number" TEXT,
                "Location" TEXT,
                "Sales Person" TEXT,
                "Submitted By" TEXT,
                "Status" TEXT,
                "Filming Date" TEXT,
                "Videographer" TEXT,
                "Video Filename" TEXT,
                "Current Version" TEXT,
                "Version History" TEXT,
                "Pending Timestamps" TEXT,
                "Submitted Timestamps" TEXT,
                "Returned Timestamps" TEXT,
                "Rejected Timestamps" TEXT,
                "Accepted Timestamps" TEXT
            );
        """)
        
        # Copy all data from old table to new table
        logger.info("Copying all data to new table...")
        conn.execute("""
            INSERT INTO live_tasks_new SELECT * FROM live_tasks;
        """)
        
        # Get row count to verify
        old_count = conn.execute("SELECT COUNT(*) FROM live_tasks").fetchone()[0]
        new_count = conn.execute("SELECT COUNT(*) FROM live_tasks_new").fetchone()[0]
        logger.info(f"Copied {new_count} rows (original had {old_count})")
        
        if old_count != new_count:
            raise Exception(f"Row count mismatch! Old: {old_count}, New: {new_count}")
        
        # Drop old table
        logger.info("Dropping old table...")
        conn.execute("DROP TABLE live_tasks")
        
        # Rename new table to original name
        logger.info("Renaming new table...")
        conn.execute("ALTER TABLE live_tasks_new RENAME TO live_tasks")
        
        # Update sqlite_sequence to maintain task_number continuity
        if max_task > 0:
            logger.info(f"Updating sequence to maintain task_number continuity at {max_task}...")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='live_tasks'")
            conn.execute("INSERT INTO sqlite_sequence(name, seq) VALUES ('live_tasks', ?)", (max_task,))
        
        # Verify the migration
        if check_current_schema(conn):
            raise Exception("Migration verification failed - UNIQUE constraint still present!")
        
        # Commit transaction
        conn.commit()
        logger.info("Migration completed successfully!")
        
        # Final verification - count rows again
        final_count = conn.execute("SELECT COUNT(*) FROM live_tasks").fetchone()[0]
        logger.info(f"Final row count: {final_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if conn:
            conn.rollback()
        
        # Attempt to restore backup
        try:
            logger.info("Attempting to restore from backup...")
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, HISTORY_DB_PATH)
                logger.info("Database restored from backup")
            else:
                logger.error("Backup file not found!")
        except Exception as restore_error:
            logger.error(f"Failed to restore backup: {restore_error}")
            logger.error("CRITICAL: Database may be in inconsistent state!")
        
        raise e
        
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("\n=== Migration 001: Remove Reference Number UNIQUE Constraint ===\n")
    print("This migration will:")
    print("1. Create a backup of your database")
    print("2. Remove the UNIQUE constraint from Reference Number column")
    print("3. Preserve all existing data and task numbers")
    print("4. Allow duplicate reference numbers (with warning)")
    print("\nA backup will be created before any changes are made.")
    
    confirm = input("\nContinue with migration? (yes/no): ")
    
    if confirm.lower() == 'yes':
        try:
            migrate()
            print("\n✅ Migration completed successfully!")
            print("The UNIQUE constraint has been removed from Reference Number.")
            print("Duplicate reference numbers will now be allowed after user confirmation.")
        except Exception as e:
            print(f"\n❌ Migration failed: {e}")
            print("Check the logs for details. Your database has been restored from backup if possible.")
            sys.exit(1)
    else:
        print("\nMigration cancelled.")