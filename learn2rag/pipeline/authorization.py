from collections import defaultdict
from typing import Any, Dict, List, Mapping, Set
import logging

from qdrant_client.http.models import QueryResponse, ScoredPoint

from learn2rag.pipeline.authorization_filter import AuthorizationFilter
from .authorization_drupal import DrupalAuthorizationFilter
from learn2rag.pipeline.authorization_sharepoint import SharepointAuthorizationFilter
from learn2rag.pipeline.config import importer_config

logger = logging.getLogger(__name__)


class NoAuthorizationFilter(AuthorizationFilter):
    """Authorization filter that allows access to all documents."""

    async def filter_documents(self, user_auth: Any, documents: Mapping[str, Any]) -> Set[str]:
        """
        Return all document IDs without filtering.

        Args:
            user: the user identifier (ignored)
            document_ids: List of document IDs to filter

        Returns:
            All document IDs unchanged
        """
        return set(documents.keys())

def _create_authorization_filter(entry: Dict[str, str]) -> AuthorizationFilter:
    user_auth_type = entry.get('user_auth_type', 'none')
    logger.debug(f'{user_auth_type = }')
    if user_auth_type == 'none':
        return NoAuthorizationFilter()

    loader_type = entry.get("loader_type")

    if loader_type == 'DrupalLoader':
        return DrupalAuthorizationFilter(
            loader_id=entry['loader_id'],
            base_url=entry['base_url'],
        )

    if loader_type == "SharepointLoader":
        for elem in ('loader_id', 'client_id', 'client_secret', 'tenant_id', 'site_id', 'document_library_id'):
            if elem not in entry:
                raise RuntimeError(f'Key {elem} is required for SharepointLoader')

        return SharepointAuthorizationFilter( 
            loader_id=entry["loader_id"],
            client_id=entry["client_id"],
            client_secret=entry["client_secret"],
            tenant_id=entry["tenant_id"],
            site_id=entry["site_id"],
            document_library_id=entry["document_library_id"]
        )

    raise NotImplementedError()


_filters: Dict[str, AuthorizationFilter] = {}
_configuredLoaders: List[Dict[str, str]] = importer_config.get("loaders")


def _get_authorization_filter(loader_id: str) -> AuthorizationFilter:
    # Return an existing filter if already created
    if loader_id in _filters:
        return _filters[loader_id]

    loader_config = next(
        (loader for loader in _configuredLoaders if loader.get("loader_id") == loader_id),
        None,
    )

    # If loader not found, raise exception
    if loader_config is None:
        raise ValueError(f"Loader configuration not found for loader_id: {loader_id}")

    # Create and cache the filter
    _filters[loader_id] = _create_authorization_filter(loader_config)
    return _filters[loader_id]


async def _get_loader_id(point: ScoredPoint) -> str:
    if not point.payload:
        return 'unknown'
    return str(point.payload.get('loader_id', 'unknown'))


async def _get_doc_id(point: ScoredPoint) -> str:
    if not point.payload:
        return ''
    return str(point.payload.get("document_id", ""))


async def filter_authorized(user_auths: Mapping[str, Any], search_results: QueryResponse) -> List[ScoredPoint]:
    by_loader = defaultdict(list)
    for point in search_results.points:
        if point.payload:
            loader_id = await _get_loader_id(point)
            by_loader[loader_id].append(point)

    authorized_ids = {}
    for loader in by_loader:
        auth_filter = _get_authorization_filter(loader)
        authorized_ids[loader] = await auth_filter.filter_documents(user_auths.get(loader), {await _get_doc_id(point): point.payload for point in by_loader[loader]})
    authorized_points = [point for point in search_results.points if
                         authorized_ids[await _get_loader_id(point)].__contains__(await _get_doc_id(point))]
    logger.debug('Authorization filter accepted %s documents out of %s', len(authorized_points), len(search_results.points))
    return authorized_points
