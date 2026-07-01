import os
from pathlib import Path

for run in Path("runs").iterdir():
    has_capture_z = has_cv_z = False
    for subdir in run.iterdir():
        if subdir.parts[-1] == "cv_buffers.z":
            has_cv_z = True
        elif subdir.parts[-1] == "capture_buffers.z":
            has_capture_z = True
    if has_capture_z and has_cv_z:
        if (run / "cv_buffers").exists():
            print("rm -rf", (run / "cv_buffers"))
        if (run / "capture_buffers").exists():
            print("rm -rf", (run / "capture_buffers"))
