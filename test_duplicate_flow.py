#!/usr/bin/env python3
"""Test script to verify duplicate reference number handling."""

import asyncio
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_utils import init_db, insert_task, check_duplicate_reference, save_task
from logger import logger

async def test_duplicate_flow():
    """Test that duplicates are detected but can still be inserted."""
    
    print("=== Testing Duplicate Reference Number Flow ===\n")
    
    # Initialize database
    init_db()
    print("✓ Database initialized\n")
    
    # Test reference number
    test_ref = "TEST-DUP-001"
    
    # Create first task with this reference
    print(f"1. Creating first task with reference: {test_ref}")
    first_task = {
        'brand': 'Test Brand 1',
        'start_date': '2025-01-01',
        'end_date': '2025-01-31',
        'reference_number': test_ref,
        'location': 'Dubai',
        'sales_person': 'Test Sales',
        'submitted_by': 'Test User'
    }
    
    result1 = await save_task(first_task)
    if result1['success']:
        print(f"✓ First task created successfully - Task #{result1['task_number']}\n")
    else:
        print(f"✗ Failed to create first task: {result1}\n")
        return
    
    # Check if duplicate is detected
    print(f"2. Checking if reference {test_ref} is detected as duplicate")
    dup_check = check_duplicate_reference(test_ref)
    
    if dup_check['is_duplicate']:
        print("✓ Duplicate correctly detected!")
        existing = dup_check['existing_entry']
        print(f"   - Existing task: #{existing['task_number']} - {existing['brand']}")
        print(f"   - Location: {existing['location']}")
        print(f"   - Campaign: {existing['start_date']} to {existing['end_date']}\n")
    else:
        print("✗ Duplicate NOT detected (this is a problem!)\n")
        return
    
    # Try to create second task with same reference
    print(f"3. Creating second task with same reference: {test_ref}")
    second_task = {
        'brand': 'Test Brand 2',
        'start_date': '2025-02-01', 
        'end_date': '2025-02-28',
        'reference_number': test_ref,
        'location': 'Abu Dhabi',
        'sales_person': 'Another Sales',
        'submitted_by': 'Test User 2'
    }
    
    # In real flow, user would confirm duplicate at this point
    print("   (Simulating user confirmation to proceed with duplicate...)")
    
    try:
        result2 = await save_task(second_task)
        if result2['success']:
            print(f"✓ Second task created successfully despite duplicate - Task #{result2['task_number']}")
            print("✓ UNIQUE constraint properly removed - duplicates allowed!\n")
        else:
            print(f"✗ Failed to create second task: {result2}\n")
    except Exception as e:
        print(f"✗ Error creating second task: {e}")
        print("   This suggests UNIQUE constraint is still active in this database\n")
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_duplicate_flow())