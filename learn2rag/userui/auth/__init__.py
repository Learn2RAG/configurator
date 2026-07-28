from pathlib import Path
from typing import Any, Mapping
import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import oauth
from .utils import AuthImplRouter

logger = logging.getLogger(__name__)


def build_router(import_config: Mapping[str, Any]) -> APIRouter:
    '''
    All supported user account integrations.
    New integrations can be added here.
    '''
    auth_routers: dict[str, AuthImplRouter] = {
        'oauth': oauth.build_router(import_config),
    }

    router = APIRouter()
    for prefix, subrouter in auth_routers.items():
        router.include_router(subrouter, prefix='/' + prefix)

    templates = Jinja2Templates(directory=Path(__file__).parent / 'templates')

    @router.get('/')
    async def auth(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, 'auth.html', context={
            'auth_routers': auth_routers,
        })

    return router
