"""
nanoStudio LLM Benchmarking CLI Entry Point (Backward-Compatibility Shim)
Core benchmark engine is located in tasks/llm/bench.py
"""
import os
import sys
import runpy

cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.insert(0, cur_dir)

target_script = os.path.join(cur_dir, "tasks", "llm", "bench.py")
runpy.run_path(target_script, run_name="__main__")
