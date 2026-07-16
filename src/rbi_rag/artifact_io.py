from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def rel_path(path: Path, root: Path = Path(".")) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("/", "\\")


def relative_posix(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or (sorted({key for row in rows for key in row}) if rows else ["empty"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(row.get(key), sort_keys=True, default=str)
                    if isinstance(row.get(key), (dict, list, tuple, Counter))
                    else row.get(key)
                )
                for key in fieldnames
            })


def make_checksum_manifest(root: Path, targets: list[Path]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    missing_targets: list[str] = []
    for target in targets:
        path = root / target
        if path.is_file():
            entries.append({"path": rel_path(path, root), "sha256": file_sha(path)})
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    entries.append({"path": rel_path(child, root), "sha256": file_sha(child)})
        else:
            missing_targets.append(str(target))
    return {
        "schema_version": 1,
        "created_at_utc": now_iso(),
        "targets": [str(target).replace("/", "\\") for target in targets],
        "entry_count": len(entries),
        "missing_targets": missing_targets,
        "entries": entries,
    }
