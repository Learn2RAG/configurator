import logging
from typing import Set, List

from azure.identity import ClientSecretCredential
from msgraph import GraphServiceClient
from msgraph.generated.drives.item.drive_item_request_builder import DriveItemRequestBuilder
from msgraph.generated.models.share_point_identity_set import SharePointIdentitySet

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

        scopes = ['https://graph.microsoft.com/.default']
        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret)
        self.graph_client = GraphServiceClient(credential, scopes) # type: ignore

    def _get_drive(self) -> DriveItemRequestBuilder:
        """Get the SharePoint drive for the configured site."""
        return self.graph_client.drives.by_drive_id(self.document_library_id)

    async def _get_groups(self, user_id: str) -> List[str]:
        groups = await self.graph_client.users.by_user_id(user_id).transitive_member_of.get()
        return [group.id for group in list(groups.value)]
    
    async def _get_owned_groups(self, user_id: str) -> List[str]:
        groups = await self.graph_client.users.by_user_id(user_id).owned_objects.graph_group.get()
        return [group.id for group in list(groups.value)]

    @staticmethod
    def _is_owner_permission(permission: SharePointIdentitySet) -> bool:
        
        site_user = permission.site_user
        if site_user is None:
            return False
        group = permission.group
        if group is None:
            return False
        return site_user.login_name == f"c:0o.c|federateddirectoryclaimprovider|{group.id}_o"

    @staticmethod
    async def _user_has_access(
            user_id: str,
            document_id: str,
            drive: DriveItemRequestBuilder,
            group_ids: List[str],
            owned_group_ids: List[str],
        ) -> bool:
        """
        Check if a user has access to a specific file.

        Args:
            user_id: User identifier (email)
            document_id: The id of the drive item in sharepoint
            drive: SharePoint drive object

        Returns:
            True if the user has access, False otherwise
        """
        try:
            # Navigate to the file
            item = drive.items.by_drive_item_id(document_id)
            if not item:
                return False

            # Get permissions for the file
            permissions = await item.permissions.get()

            # Check if the user has access through any permission
            for permission in list(permissions.value):
                v2 = permission.granted_to_v2
                if v2 is None:
                    continue
                if v2.user is not None and v2.user.id == user_id:
                    return True
                if v2.group is not None:
                    group = v2.group
                    if SharepointAuthorizationFilter._is_owner_permission(v2):
                        if owned_group_ids.__contains__(group.id):
                            return True
                    else:
                        if group_ids.__contains__(group.id):
                            return True
            return False
        except Exception:
            return False

    async def filter_documents(self, user: str, document_ids: Set[str]) -> Set[str]:
        """
        Filter document IDs based on SharePoint permissions.

        Args:
            user: User identifier (email)
            document_ids: List of document IDs (file paths) to filter

        Returns:
            List of authorized document IDs
        """
        drive = self._get_drive()
        member_group_ids = await self._get_groups(user)
        owned_group_ids = await self._get_owned_groups(user)
        authorized_ids = []
        for doc_id in document_ids:
            is_authorized = await self._user_has_access(user, doc_id, drive, member_group_ids, owned_group_ids)
            if is_authorized:
                authorized_ids.append(doc_id)

        return set(authorized_ids)
