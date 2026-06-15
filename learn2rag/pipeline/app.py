import json
import logging
from operator import itemgetter
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    AsyncGenerator,
    List,
    Optional,
)

from fastapi import FastAPI, Body, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from qdrant_client.models import ScoredPoint

from . import ingestion
from .config import user_config, opt_config
from .qdrant import Qdrant
from .search import search_authorized
from .operators import BasicPipeline
from .operators.base import BaseOperator

pipeline: BaseOperator = BasicPipeline()


class QuestionInput(BaseModel):
    question: str
    user: str


class Message(BaseModel):
    role: str
    content: str


class ChatState(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    user: str = 'anonymous'  # FIXME use https://developers.openai.com/api/docs/guides/safety-best-practices#safety-identifiers

class TestResponse(BaseModel):
    message: str

async def simple_chatbot_response(input: QuestionInput) -> Any:
    return itemgetter('answer')(await pipeline(inputs={
        'question': input.question,
        'user': input.user,
    }))


example_query = "What approach did Arjun Singh's campaign use to respond to voters' concerns on social media platforms during the municipal elections in Delhi?"
example_messages = {
    "messages": [
        {
            "role": "user",
            "content": example_query
        }
    ],
    "user": "d56d14d0-79c7-4c49-9499-07634a2610c2"
}

app = FastAPI()


@app.on_event("startup")
async def startup_event() -> None:
    Qdrant.ensure_collection(user_config["collection_name"], opt_config)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    message = str(exc)
    logging.error(f"validation_exception_handler: {message}")
    content = {'message': message}
    return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


@app.post("/qanda")
async def qanda(
        input: QuestionInput = Body(
            ...,
            example={
                "question": example_query,
                "user": "d56d14d0-79c7-4c49-9499-07634a2610c2"
            }
        )
) -> ChatState:
    answer = await simple_chatbot_response(input)
    
    return ChatState(messages=[Message(content=answer, role="model")])


@app.get("/models")  # OpenAI API for Open WebUI
async def get_models() -> JSONResponse: return JSONResponse({
        'object': 'list',
        'data': [{"id": "Learn2RAG"}],
})


@app.post("/stream")
async def stream(
        inputs: ChatState = Body(
            ...,
            example=example_messages
        )
) -> StreamingResponse:
    return streaming_response(inputs)


@app.post("/chat/completions", response_model=None)  # OpenAI API for Open WebUI
async def chat_completions(
        inputs: ChatState = Body(
            ...,
            example=example_messages
        )
) -> JSONResponse | StreamingResponse:
    if inputs.stream:
        return streaming_response(inputs)
    else:
        return await simple_response(inputs)


async def run_pipeline(chat_state: ChatState) -> Any:
    if not chat_state.user:
        raise ValueError("User Missing")

    return await pipeline(inputs={
        'question': chat_state.messages[-1].content,
        'user': chat_state.user,
    })


async def event_stream(inputs: ChatState) -> AsyncGenerator[Any, Any]:
    try:
        answer = itemgetter('answer')(await run_pipeline(inputs))

        delta = {'content': answer}
        yield f"data: {json.dumps({'choices': [{'delta': delta, 'finish_reason': 'stop'}]})}\n\n"
    except Exception as e:
        logging.error('%s: %s', e.__class__, e)
        content = 'There is a problem with Learn2RAG configuration. Please contact your administrator.'  # FIXME
        delta = {'content': content}
        yield f"data: {json.dumps({'choices': [{'delta': delta, 'finish_reason': 'stop'}]})}\n\n"


async def simple_response(inputs: ChatState) -> JSONResponse:
    answer = itemgetter('answer')(await run_pipeline(inputs))

    return JSONResponse({
        'choices': [
            {
                'message': {
                    'content': answer,
                    'role': 'assistant',
                },
                'finish_reason': 'stop',
            },
        ],
    })


def streaming_response(inputs: ChatState) -> StreamingResponse:
    return StreamingResponse(
        event_stream(inputs),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.post("/search")
async def search(
        input: QuestionInput = Body(
            ...,
            example={
                "question": example_query,
                "user": "d56d14d0-79c7-4c49-9499-07634a2610c2"
            }
        )
) -> List[ScoredPoint]:
    return await search_authorized(user=input.user, question=input.question)


@app.get("/test")
async def test() -> TestResponse:
    return TestResponse(message="Hello World")
