from typing import (
    Any,
    Awaitable,
    Callable,
    MutableMapping,
)

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class PrefixRewriteMiddleware(BaseHTTPMiddleware):
    def __init__(
            self,
            app: Callable[[MutableMapping[str, Any], Callable[[], Awaitable[MutableMapping[str, Any]]], Callable[[MutableMapping[str, Any]], Awaitable[None]]], Awaitable[None]],
            *,
            source: str,
            target: str,
            paths: list[str],
    ):
        super().__init__(app)
        self.src = source
        self.dst = target
        self.paths = paths

    async def dispatch(self, req: Request, call_next: RequestResponseEndpoint) -> Response:
        path = req.url.path
        if path.startswith(self.src):
            if any(path.startswith(self.src + item) for item in self.paths):
                req.scope['path'] = path.replace(self.src, self.dst, 1)
        return await call_next(req)
