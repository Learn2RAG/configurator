from typing import Any, Callable, Generator, Mapping
import logging
import secrets

from authlib.integrations.starlette_client import OAuth
from fastapi import HTTPException, Request, status
from fastapi.responses import RedirectResponse

from .utils import AuthImplRouter

logger = logging.getLogger(__name__)


def discover_applications(import_config: Mapping[str, Any]) -> Generator[tuple[str, str, dict[str, Any]], None, None]:
    for loader_config in import_config['loaders']:
        if loader_config.get('user_auth_type', 'none') == 'oauth':
            name = loader_config['loader_id']
            client_id = loader_config.get('oauth_client_id', '')
            client_secret = loader_config.get('oauth_client_secret', '')
            base_url = loader_config.get('base_url', '')
            if client_id != '' and client_secret != '' and base_url != '':
                logger.info('Discovered OAuth application: %s (%s), client_id=%s)', name, base_url, client_id)
                yield name, f'OAuth ({base_url})', {
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


def build_router(import_config: Mapping[str, Any], login_handler: Callable[[Request, str, str], None], logout_handler: Callable[[Request, str], None]) -> AuthImplRouter:
    router = OAuthRouter()

    oauth = OAuth()
    for name, label, kwargs in discover_applications(import_config):
        oauth.register(name, **kwargs)
        router.applications[name] = label

    @router.post('/{name}/login')
    async def login(name: str, request: Request) -> Any:
        provider = getattr(oauth, name)
        redirect_uri = request.url_for('oauth_callback')
        state = name + ':' + secrets.token_hex()
        return await provider.authorize_redirect(request, redirect_uri, state=state)

    @router.get('/callback')
    async def oauth_callback(request: Request) -> Any:
        name = request.query_params['state'].split(':')[0]
        provider = getattr(oauth, name)
        try:
            token = await provider.authorize_access_token(request)
        except Exception as e:
            logger.error('OAuth authentication failed', e)
            raise HTTPException(status_code=400, detail='Callback failed') from e
        logger.info('OAuth authentication succeeded')
        login_handler(request, name, token)
        return RedirectResponse('../..')

    @router.post('/{name}/logout')
    async def logout(name: str, request: Request) -> Any:
        logout_handler(request, name)
        return RedirectResponse(
            '../../..',
            status_code=status.HTTP_303_SEE_OTHER,
        )

    return router
