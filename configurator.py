"""
nanoStudio Configurator (Backward-Compatibility Shim)
Implementation is located in tasks/llm/configurator.py
"""
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
target_script = os.path.join(cur_dir, "tasks", "llm", "configurator.py")
exec(open(target_script, encoding='utf-8').read())
