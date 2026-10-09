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
        Check if a user has access to a specific file, with heavy debugging.
        """
        try:
            logger.debug(f"--- AUTH DEBUG: Starting check for document ---")
            logger.debug(f"AUTH DEBUG: Token value is '{access_token}', type: {type(access_token)}")

            content_type = document.get('content_type')
            logger.debug(f"AUTH DEBUG: Document keys available: {list(document.keys())}")
            logger.debug(f"AUTH DEBUG: Extracted content_type: '{content_type}'")

            if content_type == 'article' and not access_token:
                logger.debug("AUTH DEBUG: Denied by fast-metadata check (is article, no token).")
                return False

            # Setup session
            if access_token:
                logger.debug("AUTH DEBUG: Building session WITH token.")
                session = _build_session('token', '', '', access_token)
            else:
                logger.debug("AUTH DEBUG: Building session WITHOUT token (anonymous).")
                session = _build_session('none', '', '', '')

            access_url = document.get('source')
            logger.debug(f"AUTH DEBUG: Attempting network request to: {access_url}")

            # Network request (no redirects)
            response = session.get(access_url, timeout=10, allow_redirects=False)

            logger.debug(f"AUTH DEBUG: Response Status Code: {response.status_code}")
            logger.debug(f"AUTH DEBUG: Response Headers (Location/Redirect): {response.headers.get('Location', 'None')}")

            if response.status_code >= 500:
                logger.error("Server error (%s) while checking for user's access; text: `%s`", response.status_code,
                             response.text)

            is_authorized = (response.status_code == 200)
            logger.debug(
                f"AUTH DEBUG: Final verdict for {access_url} -> Authorized: {is_authorized}\n-------------------------")

            return is_authorized

        except Exception as e:
            logger.error(f"AUTH DEBUG: Exception while checking user's access to a document: {e}")
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
