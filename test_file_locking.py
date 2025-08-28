#!/usr/bin/env python3
"""
Test cases for Excel file locking mechanism
Run these tests to verify locking works correctly
"""

import asyncio
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from excel_lock_utils import ExcelLockManager, safe_read_excel, safe_write_excel
from config import EXCEL_FILE_PATH

# Test 1: Basic Lock Acquisition and Release
async def test_1_basic_lock():
    """Test that basic lock acquisition and release works"""
    print("\n🧪 Test 1: Basic Lock Acquisition")
    lock_manager = ExcelLockManager()
    
    try:
        with lock_manager.acquire_lock("test"):
            print("✅ Lock acquired successfully")
            print(f"   Lock info: {lock_manager.get_lock_info()}")
            print(f"   Is locked: {lock_manager.is_locked()}")
        print("✅ Lock released successfully")
        print(f"   Is locked after release: {lock_manager.is_locked()}")
    except Exception as e:
        print(f"❌ Test failed: {e}")


# Test 2: Concurrent Read Operations
async def test_2_concurrent_reads():
    """Test that multiple reads can happen simultaneously"""
    print("\n🧪 Test 2: Concurrent Read Operations")
    
    async def read_task(task_id):
        start = time.time()
        try:
            df = await safe_read_excel()
            elapsed = time.time() - start
            print(f"✅ Read {task_id} completed in {elapsed:.2f}s - found {len(df)} rows")
            return True
        except Exception as e:
            print(f"❌ Read {task_id} failed: {e}")
            return False
    
    # Launch 3 concurrent reads
    tasks = [read_task(i) for i in range(1, 4)]
    results = await asyncio.gather(*tasks)
    
    if all(results):
        print("✅ All concurrent reads succeeded")
    else:
        print("❌ Some reads failed")


# Test 3: Write Lock Exclusivity
async def test_3_write_exclusivity():
    """Test that only one write can happen at a time"""
    print("\n🧪 Test 3: Write Lock Exclusivity")
    
    async def write_task(task_id, delay=0):
        await asyncio.sleep(delay)  # Stagger starts slightly
        start = time.time()
        try:
            df = await safe_read_excel()
            # Modify something
            df.loc[0, 'Brand'] = f"Test Brand {task_id}"
            success = await safe_write_excel(df)
            elapsed = time.time() - start
            if success:
                print(f"✅ Write {task_id} completed in {elapsed:.2f}s")
            else:
                print(f"⏳ Write {task_id} timed out after {elapsed:.2f}s")
            return success
        except Exception as e:
            elapsed = time.time() - start
            print(f"❌ Write {task_id} failed after {elapsed:.2f}s: {e}")
            return False
    
    # Launch 2 concurrent writes
    tasks = [
        write_task(1, 0),
        write_task(2, 0.1)  # Start slightly after
    ]
    results = await asyncio.gather(*tasks)
    
    print(f"Write results: {results}")
    print("Note: One write should complete quickly, the other should wait")


# Test 4: Lock Timeout Behavior
async def test_4_lock_timeout():
    """Test that locks timeout after the specified duration"""
    print("\n🧪 Test 4: Lock Timeout Behavior")
    
    # Create a lock manager with short timeout
    lock_manager = ExcelLockManager(lock_timeout=2)  # 2 second timeout
    
    async def hold_lock_forever():
        with lock_manager.acquire_lock("holder"):
            print("🔒 Holding lock indefinitely...")
            await asyncio.sleep(5)  # Hold for 5 seconds
    
    async def try_to_acquire():
        await asyncio.sleep(0.5)  # Let the first task grab the lock
        start = time.time()
        try:
            with lock_manager.acquire_lock("waiter"):
                print("✅ Acquired lock (unexpected!)")
        except TimeoutError:
            elapsed = time.time() - start
            print(f"✅ Lock timed out as expected after {elapsed:.2f}s")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    # Run both tasks
    await asyncio.gather(
        hold_lock_forever(),
        try_to_acquire(),
        return_exceptions=True
    )


# Test 5: Stress Test - Rapid Operations
async def test_5_stress_test():
    """Test system under rapid concurrent operations"""
    print("\n🧪 Test 5: Stress Test - 10 operations")
    
    operation_count = {"reads": 0, "writes": 0, "errors": 0}
    
    async def random_operation(op_id):
        import random
        op_type = random.choice(["read", "read", "write"])  # 2:1 read:write ratio
        
        try:
            if op_type == "read":
                df = await safe_read_excel()
                operation_count["reads"] += 1
                print(f"   R{op_id} ✓", end="", flush=True)
            else:
                df = await safe_read_excel()
                df.loc[0, 'Brand'] = f"Stress Test {op_id}"
                success = await safe_write_excel(df)
                if success:
                    operation_count["writes"] += 1
                    print(f"   W{op_id} ✓", end="", flush=True)
                else:
                    operation_count["errors"] += 1
                    print(f"   W{op_id} ✗", end="", flush=True)
        except Exception as e:
            operation_count["errors"] += 1
            print(f"   {op_id}❌", end="", flush=True)
    
    # Launch 10 concurrent operations
    start = time.time()
    tasks = [random_operation(i) for i in range(10)]
    await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    print(f"\n\n📊 Stress Test Results ({elapsed:.2f}s):")
    print(f"   Reads: {operation_count['reads']}")
    print(f"   Writes: {operation_count['writes']}")
    print(f"   Errors: {operation_count['errors']}")
    
    if operation_count["errors"] == 0:
        print("✅ All operations completed without errors")
    else:
        print("⚠️  Some operations failed")


# Main runner
async def run_all_tests():
    """Run all test cases"""
    print("🚀 Excel File Locking Test Suite")
    print("================================")
    
    # Check if Excel file exists
    import os
    if not os.path.exists(EXCEL_FILE_PATH):
        print(f"❌ Excel file not found at {EXCEL_FILE_PATH}")
        print("Creating a test Excel file...")
        df = pd.DataFrame({
            'Task #': [1, 2, 3],
            'Brand': ['Test1', 'Test2', 'Test3'],
            'Status': ['Active', 'Active', 'Active']
        })
        df.to_excel(EXCEL_FILE_PATH, index=False)
        print("✅ Test file created")
    
    await test_1_basic_lock()
    await test_2_concurrent_reads()
    await test_3_write_exclusivity()
    await test_4_lock_timeout()
    await test_5_stress_test()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())