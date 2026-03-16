from typing import Protocol, Set


class AuthorizationFilter(Protocol):
    """Interface for filtering documents based on authorization rules."""

    async def filter_documents(self, user: str, document_ids: Set[str]) -> Set[str]:
        """
        Filter a list of document IDs based on authorization rules.

        Args:
            user: the user identifier
            document_ids: List of document IDs to filter

        Returns:
            List of authorized document IDs
        """
        ...
