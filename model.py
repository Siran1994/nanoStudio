"""
nanoStudio Model Entry Point (Backward-Compatibility Shim)
Core GPT model implementation is located in tasks/llm/model.py
"""
import sys
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from tasks.llm.model import *
