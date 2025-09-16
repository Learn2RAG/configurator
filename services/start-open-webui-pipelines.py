#!/usr/bin/env python3
import subprocess
import os
import sys

learn2rag_path = os.environ.get("LEARN2RAG_PATH", ".")

python_exe = os.path.join(
    learn2rag_path,
    "services",
    "open-webui-pipelines",
    ".venv",
    "Scripts",
    "python.exe",
)
start_py = os.path.join(learn2rag_path, "services", "open-webui-pipelines", "start.py")

# Fallback to system Python if venv Python doesn't exist
if not os.path.isfile(python_exe):
    python_exe = sys.executable

# Only run if start.py exists
if os.path.isfile(start_py):
    cmd = [python_exe, start_py]
    print(f"Launching: {cmd}")
    subprocess.run(cmd, check=True)
else:
    print("Skipping start.py: File does not exist.")
