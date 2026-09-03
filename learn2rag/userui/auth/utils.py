from abc import ABC, abstractmethod
from typing import Mapping

from fastapi import APIRouter


class AuthImplRouter(ABC, APIRouter):
    @abstractmethod
    def registered_applications(self) -> Mapping[str, str]:
        raise NotImplementedError()


class AuthRouter(ABC, APIRouter):
    auth_routers: dict[str, AuthImplRouter]
