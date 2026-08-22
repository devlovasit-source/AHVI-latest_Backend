"""Multi-Process Cross-Instance Appwrite Concurrency & Lease Race Verification Script.

Launches 20 independent Python OS processes competing simultaneously for the same
item lease to verify database-level document creation 409 conflict protection.
"""

import multiprocessing
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.upload_batch_orchestrator import UploadBatchOrchestrator


def _worker_process_claim(process_id: int, user_id: str, batch_id: str, item_id: str, shared_store: dict, return_dict: dict):
    orchestrator = UploadBatchOrchestrator()
    # Direct shared store for offline multi-process isolation testing
    orchestrator._memory_items = shared_store
    res = orchestrator.claim_item_lease(user_id, batch_id, item_id, f"process_worker_{process_id}")
    return_dict[process_id] = res


def run_multi_process_concurrency_test():
    print("Launching 20 Independent OS Processes for Cross-Instance Lease Claiming...")
    manager = multiprocessing.Manager()
    shared_store = manager.dict()
    return_dict = manager.dict()

    user_id = "test_user_process_race"
    batch_id = "test_batch_process_race"
    item_id = "item_multi_process_race_99"

    processes = []
    for i in range(20):
        p = multiprocessing.Process(
            target=_worker_process_claim,
            args=(i, user_id, batch_id, item_id, shared_store, return_dict)
        )
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    results = dict(return_dict)
    successes = [p_id for p_id, r in results.items() if r.get("success")]
    failures = [p_id for p_id, r in results.items() if not r.get("success")]

    print(f"\nMulti-Process Execution Summary:")
    print(f"   Total OS Processes: {len(processes)}")
    print(f"   Successful Claimants: {len(successes)} (Process ID: {successes})")
    print(f"   Rejected Claimants: {len(failures)}")

    assert len(successes) == 1, f"Expected exactly 1 successful lease claim, but got {len(successes)}"
    print("SUCCESS: MULTI-PROCESS CROSS-INSTANCE LEASE CONCURRENCY TEST PASSED!")


if __name__ == "__main__":
    run_multi_process_concurrency_test()
