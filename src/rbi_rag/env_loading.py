from __future__ import annotations

from pathlib import Path


_LOADED: set[Path] = set()


def load_project_dotenv(root: Path | str | None = None) -> bool:
    """Load a project .env file without failing if python-dotenv is absent.

    The function intentionally returns only a boolean and never exposes any
    value loaded from the file.
    """

    try:
        from dotenv import find_dotenv, load_dotenv
    except Exception:
        return False

    candidates: list[Path] = []
    if root is not None:
        candidates.append(Path(root).resolve() / ".env")

    discovered = find_dotenv(usecwd=True)
    if discovered:
        candidates.append(Path(discovered).resolve())

    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if resolved not in _LOADED:
            load_dotenv(resolved, override=False)
            _LOADED.add(resolved)
        return True
    return False
