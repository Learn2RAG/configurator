import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import requests
from qdrant_client.http.models import QueryResponse, ScoredPoint

from ..authorization import _get_doc_id, filter_authorized
from ..authorization_drupal import DrupalAuthorizationFilter

BASE_URL = "https://drupal.example.com"
PUBLIC_URL = f"{BASE_URL}/node/1"
RESTRICTED_URL = f"{BASE_URL}/node/2"


def _drupal_point(point_id: int, source: str) -> ScoredPoint:
    # The Drupal loader sets no document_id, so ingestion stores an empty string
    return ScoredPoint(id=point_id, score=1.0, version=1, payload={
        "content": f"chunk {point_id}",
        "source": source,
        "loader_id": "drupal_test",
        "document_id": "",
    })


def _response(status_code: int) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = ""
    return response


def _fake_session(status_by_url: Dict[str, int]) -> MagicMock:
    session = MagicMock()
    session.get.side_effect = lambda url, **kwargs: _response(status_by_url[url])
    return session


class GetDocIdTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_empty_document_id_falls_back_to_source(self) -> None:
        self.assertEqual(await _get_doc_id(_drupal_point(1, PUBLIC_URL)), PUBLIC_URL)
        self.assertEqual(await _get_doc_id(_drupal_point(2, RESTRICTED_URL)), RESTRICTED_URL)

    async def test_document_id_is_preferred(self) -> None:
        point = ScoredPoint(id=1, score=1.0, version=1, payload={"document_id": "sp-42", "source": "https://sharepoint/a.docx"})
        self.assertEqual(await _get_doc_id(point), "sp-42")

    async def test_point_id_as_last_resort(self) -> None:
        point = ScoredPoint(id=7, score=1.0, version=1, payload={"document_id": ""})
        self.assertEqual(await _get_doc_id(point), "7")


class DrupalAuthorizationFilterTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.auth_filter = DrupalAuthorizationFilter(loader_id="drupal_test", base_url=BASE_URL)

    async def _has_access(self, status_code: int, token: str | None = None) -> bool:
        with patch("learn2rag.pipeline.authorization_drupal._build_session") as build_session:
            build_session.return_value = _fake_session({PUBLIC_URL: status_code})
            return await self.auth_filter._user_has_access(token, {"source": PUBLIC_URL})

    async def test_only_200_grants_access(self) -> None:
        self.assertTrue(await self._has_access(200))
        for status_code in (301, 302, 401, 403, 404, 500):
            with self.subTest(status_code=status_code):
                self.assertFalse(await self._has_access(status_code))

    async def test_redirects_are_not_followed(self) -> None:
        with patch("learn2rag.pipeline.authorization_drupal._build_session") as build_session:
            session = _fake_session({PUBLIC_URL: 200})
            build_session.return_value = session
            await self.auth_filter._user_has_access(None, {"source": PUBLIC_URL})
        self.assertIs(session.get.call_args.kwargs["allow_redirects"], False)

    async def test_request_failure_denies_access(self) -> None:
        with patch("learn2rag.pipeline.authorization_drupal._build_session") as build_session:
            build_session.return_value.get.side_effect = requests.exceptions.ConnectionError()
            self.assertFalse(await self.auth_filter._user_has_access("token", {"source": PUBLIC_URL}))

    async def test_user_token_is_used_when_logged_in(self) -> None:
        with patch("learn2rag.pipeline.authorization_drupal._build_session") as build_session:
            build_session.return_value = _fake_session({PUBLIC_URL: 200})
            await self.auth_filter.filter_documents({"token": {"access_token": "user-token"}}, {PUBLIC_URL: {"source": PUBLIC_URL}})
            self.assertEqual(build_session.call_args.args, ("token", "", "", "user-token"))

            await self.auth_filter.filter_documents(None, {PUBLIC_URL: {"source": PUBLIC_URL}})
            self.assertEqual(build_session.call_args.args, ("none", "", "", ""))


class FilterAuthorizedTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_restricted_drupal_chunks_are_removed(self) -> None:
        """Regression: chunks with an empty document_id used to share one key, so one verdict applied to all."""
        points = [
            _drupal_point(1, PUBLIC_URL),
            _drupal_point(2, RESTRICTED_URL),
            _drupal_point(3, PUBLIC_URL),
            _drupal_point(4, RESTRICTED_URL),
        ]
        auth_filter = DrupalAuthorizationFilter(loader_id="drupal_test", base_url=BASE_URL)
        session = _fake_session({PUBLIC_URL: 200, RESTRICTED_URL: 403})

        with patch("learn2rag.pipeline.authorization._get_authorization_filter", return_value=auth_filter), \
                patch("learn2rag.pipeline.authorization_drupal._build_session", return_value=session):
            authorized = await filter_authorized({}, QueryResponse(points=points))

        self.assertEqual([point.id for point in authorized], [1, 3])
        # One access check per document, not per chunk
        checked_urls: list[Any] = sorted(call.args[0] for call in session.get.call_args_list)
        self.assertEqual(checked_urls, [PUBLIC_URL, RESTRICTED_URL])
