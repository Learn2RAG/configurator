from secrets import compare_digest
from typing import Annotated, Any, Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials


def require_basic_auth(username: str, password: str) -> Callable[[HTTPBasicCredentials], None]:
    security = HTTPBasic()

    def require_basic_auth_dependency(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> None:
        if not all([
                compare_digest(credentials.username.encode('utf-8'), username.encode('utf-8')),
                compare_digest(credentials.password.encode('utf-8'), password.encode('utf-8')),
        ]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                # detail='Unauthorized',
                headers={'WWW-Authenticate': 'Basic'},
            )
    return require_basic_auth_dependency
