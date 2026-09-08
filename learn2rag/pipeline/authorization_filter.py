from typing import Any, Mapping, Protocol, Set


class AuthorizationFilter(Protocol):
    """Interface for filtering documents based on authorization rules."""

    async def filter_documents(self, user_auth: Any, documents: Mapping[str, Any]) -> Set[str]:
        """
        Filter a list of documents based on authorization rules.

        Args:
            user: the user identifier
            documents: List of documents to filter

        Returns:
            List of authorized document IDs
        """
        ...
