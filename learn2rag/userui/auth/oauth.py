from typing import Any, Generator, Mapping
import logging

from authlib.integrations.starlette_client import OAuth
from fastapi import HTTPException, Request

from .utils import AuthImplRouter

logger = logging.getLogger(__name__)


def discover_applications(import_config: Mapping[str, Any]) -> Generator[tuple[str, str, dict[str, Any]], None, None]:
    for loader_config in import_config['loaders']:
        name = loader_config['loader_id']
        client_id = loader_config.get('oauth_client_id', '')
        client_secret = loader_config.get('oauth_client_secret', '')
        base_url = loader_config.get('base_url', '')
        if client_id != '' and client_secret != '' and base_url != '':
            yield name, base_url, {
                'client_id': client_id,
                'client_secret': client_secret,
                'authorize_url': base_url + '/oauth/authorize',
                'access_token_url': base_url + '/oauth/token',
                'client_kwargs': {'scope': 'authenticated'},
            }


class OAuthRouter(AuthImplRouter):
    def __init__(self) -> None:
        super().__init__()
        self.applications: dict[str, str] = {}

    def registered_applications(self) -> Mapping[str, str]:
        return self.applications


def build_router(import_config: Mapping[str, Any]) -> AuthImplRouter:
    router = OAuthRouter()

    oauth = OAuth()
    for name, label, kwargs in discover_applications(import_config):
        logger.info('Registering OAuth application: %s (%s)', name, label)
        oauth.register(name, **kwargs)
        router.applications[name] = label

    @router.get('/{name}/login')
    async def login(name: str, request: Request) -> Any:
        provider = getattr(oauth, name)
        redirect_uri = request.url_for('oauth_callback', name=name)
        return await provider.authorize_redirect(request, redirect_uri)

    @router.get('/{name}/callback')
    async def oauth_callback(name: str, request: Request) -> Any:
        provider = getattr(oauth, name)
        try:
            token = await provider.authorize_access_token(request)
        except Exception as e:
            logger.error('OAuth authentication failed', e)
            raise HTTPException(status_code=400, detail='Callback failed') from e
        logger.info('OAuth authentication succeeded')
        print(token)
        return {'status': 'success', 'token': token}

    return router
