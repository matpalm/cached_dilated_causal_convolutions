import os
from pathlib import Path

ignore = set(["013"])
for run in Path("runs").iterdir():
    if run.parts[-1] in ignore:
        continue
    has_capture_z = has_cv_z = False
    for subdir in run.iterdir():
        if subdir.parts[-1] == "cv_buffers.z":
            has_cv_z = True
        elif subdir.parts[-1] == "capture_buffers.z":
            has_capture_z = True
    if has_capture_z and has_cv_z:
        print("rm -rf", (run / "cv_buffers"))
        print("rm -rf", (run / "capture_buffers"))
