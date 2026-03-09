from pathlib import Path
from typing import Any
import uuid

import yaml


def create_entry(root: str, category: str, data: Any) -> str:
    name = str(uuid.uuid4())
    category_root = Path(root) / 'data' / category
    category_root.mkdir(parents=True, exist_ok=True)
    with open(category_root / (name + '.yml'), 'w') as out:
        yaml.dump(data, out)
    return name


def get_all(root: str, category: str) -> dict[str, Any]:
    return {str(item.stem): yaml.safe_load(item.open()) for item in (Path(root) / 'data' / category).glob('*.yml')}


def get_entry(root: str, category: str, name: str) -> Any:
    for item in (Path(root) / 'data' / category).glob('*.yml'):
        if item.stem == name:
            return yaml.safe_load(item.open())
    raise FileNotFoundError


def get_entries(root: str, category: str, names: list[str]) -> dict[str, Any]:
    return {name: get_entry(root, category, name) for name in names}


def delete_entry(root: str, category: str, name: str) -> None:
    (Path(root) / 'data' / category / (name + '.yml')).unlink()
