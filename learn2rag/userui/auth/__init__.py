from pathlib import Path
from typing import Any, Callable, Mapping
import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import oauth
from .utils import AuthImplRouter
from ..constants import SESSION_USER_AUTHS

logger = logging.getLogger(__name__)


def build_router(import_config: Mapping[str, Any], token_handler: Callable[[Request, str, str], None]) -> APIRouter:
    '''
    All supported user account integrations.
    New integrations can be added here.
    '''
    auth_routers: dict[str, AuthImplRouter] = {
        'oauth': oauth.build_router(import_config, token_handler),
    }

    router = APIRouter()
    for prefix, subrouter in auth_routers.items():
        router.include_router(subrouter, prefix='/' + prefix)

    templates = Jinja2Templates(directory=Path(__file__).parent / 'templates')

    @router.get('/')
    async def auth(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, 'auth.html', context={
            'auth_routers': auth_routers,
            'user_auths': request.session.get(SESSION_USER_AUTHS, {}),
        })

    return router
