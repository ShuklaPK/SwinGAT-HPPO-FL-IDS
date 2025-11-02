import os
import time
import json
from pathlib import Path
from typing import Any, Dict
import yaml
from rich.console import Console

console = Console()


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
