"""Code and environment identity recorded next to artifacts."""

import platform
import subprocess
import sys
from importlib.metadata import version

TRACKED_PACKAGES = ["churnlab", "scikit-learn", "pandas", "numpy", "scipy", "joblib"]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def code_identity() -> dict:
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # Tracked files only: generated evidence files do not change the code identity.
        "uncommitted_changes": _git("status", "--porcelain", "--untracked-files=no")
        not in ("", "unknown"),
    }


def environment_identity() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: version(name) for name in TRACKED_PACKAGES},
    }
