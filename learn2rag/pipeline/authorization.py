from collections import defaultdict
from typing import List, Dict, Any
from typing import Set

from qdrant_client.http.models import QueryResponse, ScoredPoint

from learn2rag.pipeline.authorization_filter import AuthorizationFilter
from learn2rag.pipeline.authorization_sharepoint import SharepointAuthorizationFilter
from learn2rag.pipeline.config import importer_config


class NoAuthorizationFilter(AuthorizationFilter):
    """Authorization filter that allows access to all documents."""

    async def filter_documents(self, user: str, document_ids: Set[str]) -> Set[str]:
        """
        Return all document IDs without filtering.

        Args:
            user: the user identifier (ignored)
            document_ids: List of document IDs to filter

        Returns:
            All document IDs unchanged
        """
        return document_ids

def _create_authorization_filter(entry: Dict[str, str]) -> AuthorizationFilter:
    loader_type = entry.get("loader_type")
    loader_id = entry.get("loader_id")

    if loader_type == "SharepointLoader":
        return SharepointAuthorizationFilter(
            loader_id=loader_id,
            client_id=entry.get("client_id"),
            client_secret=entry.get("client_secret"),
            tenant_id=entry.get("tenant_id"),
            site_id=entry.get("site_id"),
            document_library_id=entry.get("document_library_id")
        )

    return NoAuthorizationFilter()


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


async def _get_loader_id(point: ScoredPoint) -> Any:
    return point.payload.get('loader_id', 'unknown')


async def _get_doc_id(point: ScoredPoint) -> Any:
    return point.payload.get("document_id", "")


async def filter_authorized(user: str, search_results: QueryResponse) -> List[ScoredPoint]:
    by_loader = defaultdict(list)
    [by_loader[await _get_loader_id(point)].append(await _get_doc_id(point)) for point in search_results.points]
    authorized_ids = {}
    for loader in by_loader:
        auth_filter = _get_authorization_filter(loader)
        authorized_ids[loader] = await auth_filter.filter_documents(user, set(by_loader[loader]))
    return [point for point in search_results.points if
            authorized_ids[await _get_loader_id(point)].__contains__(await _get_doc_id(point))]
