#!/usr/bin/env python3
import platform
import subprocess
import sys
import os

learn2rag_path = os.environ.get("LEARN2RAG_PATH", ".")

if platform.system() == "Windows":
    script = os.path.join(learn2rag_path, "services", "start-import.bundle")
    cmd = ["python", script]
else:
    script = os.path.join(learn2rag_path, "services", "start-import")
    cmd = ["bash", "-eu", script]

print(f"Launching: {' '.join(cmd)}")
subprocess.run(cmd, check=True)
