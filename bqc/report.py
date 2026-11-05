"""The one home for writing JSON reports under results/ (DESIGN.md section 10)."""
import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from bqc import SEED

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
LIBRARIES = ("numpy", "scikit-learn", "imbalanced-learn")


def write_report(
    results_dir: Path,
    command: str,
    parameters: Mapping[str, Any],
    body: Mapping[str, Any],
    now: datetime | None = None,
) -> Path:
    """Write `{"config": ..., **body}` as JSON and return the path.

    The config holds `command`, `parameters`, `seed`, `git_commit`, `libraries` and
    `config_hash` (SHA-256 of the other config fields). The file is named
    `<command>-<first 12 hex of config_hash>-<UTC timestamp>.json`; `results_dir` is
    created if missing. An existing file is never overwritten: FileExistsError is raised.
    `now` is for tests.
    """
    config: dict[str, Any] = {
        "command": command,
        "parameters": dict(parameters),
        "seed": SEED,
        "git_commit": _git_commit(),
        "libraries": _libraries(),
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config["config_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    name = f"{command}-{config['config_hash'][:12]}-{stamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / name
    with path.open("x", encoding="utf-8") as f:  # "x": never overwrite
        json.dump({"config": config, **body}, f, indent=1)
    return path


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _libraries() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions
