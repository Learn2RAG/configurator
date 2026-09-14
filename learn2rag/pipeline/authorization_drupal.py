import logging
from typing import Any, Mapping, Set

from .authorization_filter import AuthorizationFilter
from ..importer.loaders.drupal_loader import _build_session

logger = logging.getLogger(__name__)


class DrupalAuthorizationFilter(AuthorizationFilter):
    """Authorization Filter for resources in Drupal"""

    def __init__(
            self,
            loader_id: str,
            base_url: str,
    ):
        """
        Initialize the Drupal authorization filter.

        Args:
            loader_id: Unique identifier for this loader
            base_url: Base URL for the Drupal instance
        """
        self.loader_id = loader_id
        self.base_url = base_url

    async def _user_has_access(
            self,
            access_token: str | None,
            document: Mapping[str, Any],
        ) -> bool:
        """
        Check if a user has access to a specific file.

        Args:
            access_token: User's access token
            document: The document (metadata)

        Returns:
            True if the user has access, False otherwise
        """
        try:
            if access_token is not None:
                session = _build_session('token', '', '', access_token)
            else:
                # The user is not logged in, try without any credentials
                session = _build_session('none', '', '', '')
            access_url = document['source']
            response = session.get(access_url, timeout=30)
            if response.status_code >= 500:
                logger.error("Server error (%s) while checking for user's access; text: `%s`", response.status_code, response.text)
            logger.debug("User is allowed access: %s for the document: %s", response.ok, access_url)
            return response.ok
        except Exception as e:
            logger.error("Exception while checking user's access to a document", e)
            return False

    async def filter_documents(self, user_auth: Any, documents: Mapping[str, Any]) -> Set[str]:
        """
        Filter document IDs based on Drupal API.

        Args:
            user_auth: User's authorization data for this loader
            document_ids: List of document IDs (file paths) to filter

        Returns:
            List of authorized document IDs
        """
        access_token = user_auth['token']['access_token'] if user_auth is not None else None
        authorized_ids = []
        for doc_id, doc in documents.items():
            is_authorized = await self._user_has_access(access_token, doc)
            if is_authorized:
                authorized_ids.append(doc_id)
        return set(authorized_ids)
