import pathlib
import unittest

from ..loaders.html_loader import load_html_content

class ImporterLoadersTestCase(unittest.TestCase):
    def test_remote_url(self) -> None:
        docs = load_html_content('https://dice-research.org')
        assert len(docs) == 1
        doc, = docs
        assert 'source' in doc.metadata
        assert 'The DICE group at Paderborn University' in doc.page_content

    # def test_local_file(self):
    #     path = pathlib.Path(__file__).parent.resolve() / 'html'
    #     docs = load_html_content((path / 'local_file.html').as_uri())
    #     assert len(docs) == 1
    #     doc, = docs
    #     assert doc.page_content == 'Local file content'

    # def test_data_uri(self):
    #     docs = load_html_content('data:text/html;charset=utf-8,%3Cbody%3E%3Cp%3EData%20URI%20content%3C%2Fp%3E%3C%2Fbody%3E')
    #     assert len(docs) == 1
    #     doc, = docs
    #     assert doc.page_content == 'Data URI content'

    # TODO: actual tests
