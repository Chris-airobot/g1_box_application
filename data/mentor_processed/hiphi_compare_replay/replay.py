#!/usr/bin/env python3
"""Repository-local entry point for the HiPHI/G1 comparison replay.

Run this file directly from any working directory.  The launcher adds the
repository root to ``sys.path`` and starts the local persistent-window replay;
the retargeting package supplies its data-loading and scene helpers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
LOCAL_OMNIRETARGETING_ROOT = TOOL_DIR.parent / "retargeting"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/hiphi_retargeting_pycache")
sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]


def _find_omniretargeting_root() -> Path:
    configured = os.environ.get("OMNIRETARGETING_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        LOCAL_OMNIRETARGETING_ROOT,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.resolve()
        if (candidate / "omniretargeting" / "hiphi_compare_replay.py").is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        "Cannot locate OmniRetargeting. Set OMNIRETARGETING_ROOT to its "
        f"repository root. Searched: {searched}"
    )


def main() -> None:
    repo_root = str(_find_omniretargeting_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Keep playlist/viewer ownership in this standalone tool.  The sibling
    # retargeting tree is used only for data and scene helpers, so P/N can swap
    # motions without leaving the single launch_passive context.
    from persistent_replay import main as replay_main

    replay_main()


if __name__ == "__main__":
    main()
