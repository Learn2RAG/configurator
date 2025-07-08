from pathlib import Path
import uuid

import yaml


def create_entry(root, category, data):
    name = str(uuid.uuid4())
    category_root = Path(root) / 'data' / category
    category_root.mkdir(parents=True, exist_ok=True)
    with open(category_root / (name + '.yml'), 'w') as out:
        yaml.dump(data, out)
    return name


def get_entries(root, category):
    return {str(item.stem): yaml.safe_load(item.open()) for item in (Path(root) / 'data' / category).glob('*.yml')}


def delete_entry(root, category, name):
    (Path(root) / 'data' / category / (name + '.yml')).unlink()
