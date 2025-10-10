from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_LEGACY_SITE = _REPO_ROOT / "lib" / "python3.13" / "site-packages"

if _LEGACY_SITE.exists():
    legacy_resolved = _LEGACY_SITE.resolve()
    sys.path = [p for p in sys.path if Path(p).resolve() != legacy_resolved]

repo_str = str(_REPO_ROOT)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)
