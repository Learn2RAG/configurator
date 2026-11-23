from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Self
import uuid

from pydantic import (
    BaseModel,
    StringConstraints,
    ValidationInfo,
    model_validator,
)
from pydantic.networks import AnyHttpUrl
import yaml

from ..pipeline import llm


class LanguageModelInput(BaseModel):
    api: llm.LLMApi
    label: Annotated[str, StringConstraints(strip_whitespace=True)] = ''
    model: Annotated[str, StringConstraints(strip_whitespace=True)]
    token: Annotated[str, StringConstraints(strip_whitespace=True)] | Literal[''] = ''
    url: AnyHttpUrl | Literal[''] = ''

    @model_validator(mode='after')
    def set_label(self) -> Self:
        if self.label == '':
            self.label = self.model
        return self

    @model_validator(mode='after')
    def set_local_llm_url(self, info: ValidationInfo) -> Self:
        if self.url == '':
            if self.api == llm.LLMApi.Ollama:
                self.url = (info.context or {})['local_llm_url']
            else:
                raise ValueError('URL is missing')
        return self


class Dumper(yaml.SafeDumper):
    pass

Dumper.add_multi_representer(
    Enum, lambda dumper, member: dumper.represent_data(member.value)
)
Dumper.add_multi_representer(
    AnyHttpUrl, lambda dumper, value: dumper.represent_data(str(value))
)


def create_entry(root: str, category: str, data: Any) -> str:
    name = str(uuid.uuid4())
    category_root = Path(root) / 'data' / category
    category_root.mkdir(parents=True, exist_ok=True)
    with open(category_root / (name + '.yml'), 'w') as out:
        yaml.dump(data, out, Dumper=Dumper)
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
