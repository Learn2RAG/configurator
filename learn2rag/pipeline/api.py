'''
OpenAI-style API for clients such as Open WebUI, llama-ui and others
'''
from operator import itemgetter
from typing import (
    Any,
    AsyncGenerator,
    List,
    Optional,
)
import json
import logging

from fastapi import APIRouter, Body
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
)
from pydantic import BaseModel

from .operators import BasicPipeline
from .operators.base import BaseOperator

logger = logging.getLogger(__name__)

example_query = "What approach did Arjun Singh's campaign use to respond to voters' concerns on social media platforms during the municipal elections in Delhi?"
example_messages = {
    'messages': [
        {
            'role': 'user',
            'content': example_query
        }
    ],
    'user': 'd56d14d0-79c7-4c49-9499-07634a2610c2',
}


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


async def run_pipeline(pipeline: BaseOperator, chat_state: ChatState) -> Any:
    if not chat_state.user:
        raise ValueError('User Missing')

    return await pipeline(inputs={
        'question': chat_state.messages[-1].content,
        'user': chat_state.user,
    })


async def event_stream(pipeline: BaseOperator, inputs: ChatState) -> AsyncGenerator[Any, Any]:
    try:
        answer = itemgetter('answer')(await run_pipeline(pipeline, inputs))

        delta = {'content': answer}
        yield f"data: {json.dumps({'choices': [{'delta': delta, 'finish_reason': 'stop'}]})}\n\n"
    except Exception as e:
        logging.error('%s: %s', e.__class__, e)
        content = 'There is a problem with Learn2RAG configuration. Please contact your administrator.'  # FIXME
        delta = {'content': content}
        yield f"data: {json.dumps({'choices': [{'delta': delta, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


async def simple_response(pipeline: BaseOperator, inputs: ChatState) -> JSONResponse:
    answer = itemgetter('answer')(await run_pipeline(pipeline, inputs))

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


def streaming_response(pipeline: BaseOperator, inputs: ChatState) -> StreamingResponse:
    return StreamingResponse(
        event_stream(pipeline, inputs),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'}
    )


def build_router() -> APIRouter:
    router = APIRouter()

    pipeline: BaseOperator = BasicPipeline()

    @router.get('/v1/models')
    async def get_models() -> JSONResponse: return JSONResponse({
            'object': 'list',
            'data': [{'id': 'Learn2RAG'}],
    })

    @router.post('/v1/stream')
    async def stream(
            inputs: ChatState = Body(
                ...,
                example=example_messages
            )
    ) -> StreamingResponse:
        return streaming_response(pipeline, inputs)

    @router.post('/v1/chat/completions', response_model=None)
    async def chat_completions(
            inputs: ChatState = Body(
                ...,
                example=example_messages
            )
    ) -> JSONResponse | StreamingResponse:
        if inputs.stream:
            return streaming_response(pipeline, inputs)
        else:
            return await simple_response(pipeline, inputs)

    return router
