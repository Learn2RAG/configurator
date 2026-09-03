from typing import Any, Callable, Mapping
import logging

from fastapi import APIRouter, Request

from . import oauth
from .utils import AuthRouter

logger = logging.getLogger(__name__)


def build_router(import_config: Mapping[str, Any], login_handler: Callable[[Request, str, str], None], logout_handler: Callable[[Request, str], None]) -> AuthRouter:
    router = AuthRouter()

    '''
    All supported user account integrations.
    New integrations can be added here.
    '''
    router.auth_routers = {
        'oauth': oauth.build_router(import_config, login_handler, logout_handler),
    }
    for prefix, subrouter in router.auth_routers.items():
        router.include_router(subrouter, prefix='/' + prefix)

    return router
