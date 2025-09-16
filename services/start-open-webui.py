#!/usr/bin/env python3
import platform
import sys
import traceback
import subprocess
import os

def global_exception_handler(exc_type, exc_value, exc_traceback):
    print("Exception caught in start-open-webui.py:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)

sys.excepthook = global_exception_handler

learn2rag_path = os.environ.get("LEARN2RAG_PATH", ".")

if platform.system() == "Windows":
    script = os.path.join(learn2rag_path, "services", "open-webui", "run.bat")
    if os.path.isfile(script):
        cmd = ["cmd", "/c", script]
    else:
        print(f"Skipping run.bat: File not found → {script}")
        sys.exit(0)
else:
    script = os.path.join(learn2rag_path, "services", "start-open-webui")
    if os.path.isfile(script):
        cmd = ["bash", "-eu", script]
    else:
        print(f"Skipping start-open-webui: File not found → {script}")
        sys.exit(0)

print(f"Launching: {cmd}")

try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"Failed with exit code {e.returncode}")
    sys.exit(e.returncode)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)
