#!/usr/bin/env python3
"""Small wrapper so CI/workflows can call a stable entrypoint

Historically the project has a script named "perdict 3_days.py" (with a space).
Workflows expect a clean filename `scripts/feature_pipeline.py`. This wrapper
invokes the real script by path so we don't need to rename files in the repo.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main():
    repo_scripts = Path(__file__).parent
    target = repo_scripts / "perdict 3_days.py"
    if not target.exists():
        print(f"Error: expected script not found: {target}")
        sys.exit(2)

    cmd = [sys.executable, str(target)] + sys.argv[1:]
    returncode = subprocess.call(cmd)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
