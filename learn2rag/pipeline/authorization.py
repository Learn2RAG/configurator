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
    if not point.payload:
        return 'unknown'
    return point.payload.get('loader_id', 'unknown')


async def _get_doc_id(point: ScoredPoint) -> Any:
    if not point.payload:
        return ''
    return point.payload.get("document_id", "")


async def filter_authorized(user: str, search_results: QueryResponse) -> List[ScoredPoint]:
    by_loader = defaultdict(list)
    for point in search_results.points:
        loader_id = await _get_loader_id(point)
        doc_id = await _get_doc_id(point)
        by_loader[loader_id].append(doc_id)

    authorized_ids = {}
    for loader in by_loader:
        auth_filter = _get_authorization_filter(loader)
        authorized_ids[loader] = await auth_filter.filter_documents(user, set(by_loader[loader]))
    return [point for point in search_results.points if
            authorized_ids[await _get_loader_id(point)].__contains__(await _get_doc_id(point))]
