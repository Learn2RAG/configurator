from pathlib import Path
import shutil

from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po
from fastapi.testclient import TestClient
import pytest

from learn2rag.userui import ai_notice as ai_notice_module
from learn2rag.userui.ai_notice import ai_notice, select_language
from learn2rag.userui.app import build_app


@pytest.fixture(autouse=True)
def compiled_translations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    translations = tmp_path / 'translations'
    shutil.copytree(ai_notice_module.TRANSLATIONS_PATH, translations, ignore=shutil.ignore_patterns('*.mo'))
    for po_path in translations.glob('*/LC_MESSAGES/messages.po'):
        with po_path.open('rb') as po_file:
            catalog = read_po(po_file)
        with po_path.with_suffix('.mo').open('wb') as mo_file:
            write_mo(mo_file, catalog)
    monkeypatch.setattr(ai_notice_module, 'TRANSLATIONS_PATH', translations)
    ai_notice_module.get_translations.cache_clear()


@pytest.mark.parametrize(('header', 'expected'), [
    ('de-DE,de;q=0.9,en;q=0.8', 'de'),
    ('de-AT', 'de'),
    ('en-US,en;q=0.9,de;q=0.8', 'en'),
    ('en-GB', 'en'),
    ('fr-FR,en;q=0.5', 'en'),
    ('fr;q=0.9,en;q=0.5,de;q=0.8', 'de'),
    ('fr-FR', 'de'),
    ('', 'de'),
    (None, 'de'),
    ('not a language!!', 'de'),
])
def test_select_language(header: str | None, expected: str) -> None:
    assert select_language(header) == expected


def test_ai_notice_is_translated() -> None:
    assert ai_notice('en')['title'] == 'AI notice'
    assert ai_notice('de')['title'] == 'KI-Hinweis'
    assert ai_notice('de')['interaction'] == 'Sie interagieren mit einem KI-System, nicht mit einem Menschen.'
    assert ai_notice('de').keys() == ai_notice('en').keys()


@pytest.mark.parametrize('language', ['en', 'de'])
def test_index_footer_contains_ai_notice(language: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LEARN2RAG_PATH', str(tmp_path))
    client = TestClient(build_app())
    response = client.get('/', headers={'Accept-Language': language})
    assert response.status_code == 200
    assert f'<html lang="{language}"' in response.text
    for text in ai_notice(language).values():
        assert text in response.text
