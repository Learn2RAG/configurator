import logging
from typing import Optional, Set, List

from O365 import Account
from O365.drive import Drive

from learn2rag.pipeline.authorization_filter import AuthorizationFilter

logger = logging.getLogger(__name__)


class SharepointAuthorizationFilter(AuthorizationFilter):
    """Authorization Filter for files uploaded to sharepoint"""

    def __init__(
            self,
            loader_id: str,
            client_id: str,
            client_secret: str,
            tenant_id: str,
            site_id: str,
            document_library_id: str
    ):
        """
        Initialize the SharePoint authorization filter.

        Args:
            loader_id: Unique identifier for this loader
            client_id: Azure AD application client ID
            client_secret: Azure AD application client secret
            tenant_id: Azure AD tenant ID
            site_id: SharePoint site ID
            document_library_id: SharePoint document library ID
        """
        self.loader_id = loader_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.site_id = site_id
        self.document_library_id = document_library_id

        credentials = (client_id, client_secret)
        self.account = Account(credentials, auth_flow_type='credentials', tenant_id=tenant_id)

        if not self.account.authenticate():
            raise ValueError("Failed to authenticate with SharePoint")

    def _get_drive(self) -> Optional[Drive]:
        """Get the SharePoint drive for the configured site."""
        try:
            sharepoint = self.account.sharepoint()
            site = sharepoint.get_site(self.site_id)
            return site.get_document_library(self.document_library_id)
        except Exception:
            return None

    def _get_groups(self, user_id: str) -> List[str]:
        groups = self.account.groups().get_user_groups(user_id)
        return [group.object_id for group in groups]

    @staticmethod
    def _user_has_access(user: str, document_id: str, drive: Drive, group_ids: List[str]) -> bool:
        """
        Check if a user has access to a specific file.

        Args:
            user: User identifier (email)
            document_id: The id of the drive item in sharepoint
            drive: SharePoint drive object

        Returns:
            True if the user has access, False otherwise
        """
        try:
            # Navigate to the file
            item = drive.get_item(document_id)
            if not item:
                return False

            # Get permissions for the file
            permissions = item.get_permissions()

            # Check if user has access through any permission
            for permission in permissions:
                if hasattr(permission, 'granted_to_identities'):
                    for identity in permission.granted_to_identities:
                        if hasattr(identity, 'user') and identity.user.email.lower() == user.lower():
                            return True
                elif hasattr(permission, 'granted_to') and hasattr(permission.granted_to, 'user'):
                    if permission.granted_to.user.email.lower() == user.lower():
                        return True

            return False
        except Exception:
            return False

    def filter_documents(self, user: str, document_ids: Set[str]) -> Set[str]:
        """
        Filter document IDs based on SharePoint permissions.

        Args:
            user: User identifier (email)
            document_ids: List of document IDs (file paths) to filter

        Returns:
            List of authorized document IDs
        """
        drive = self._get_drive()
        if not drive:
            logger.warning("loader %s could not authorize documents: drive does not exist", self.loader_id)
            return set([])

        group_ids = self._get_groups(user)
        authorized_ids = []
        for doc_id in document_ids:
            if self._user_has_access(user, doc_id, drive, group_ids):
                authorized_ids.append(doc_id)

        return set(authorized_ids)
