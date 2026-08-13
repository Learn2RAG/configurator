import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import api
from .config import user_config, opt_config
from .qdrant import Qdrant
from .operators import BasicPipeline
from .operators.base import BaseOperator

pipeline: BaseOperator = BasicPipeline()



class TestResponse(BaseModel):
    message: str



app = FastAPI()


api_prefix = '/api'
app.include_router(api.build_router(), prefix=api_prefix)


@app.on_event("startup")
async def startup_event() -> None:
    Qdrant.ensure_collection(user_config["collection_name"], opt_config)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    message = str(exc)
    logging.error(f"validation_exception_handler: {message}")
    content = {'message': message}
    return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


@app.get("/test")
async def test() -> TestResponse:
    return TestResponse(message="Hello World")
