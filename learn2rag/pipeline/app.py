import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

from fastapi import FastAPI, Body, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import generate
from . import ingestion
from .config import user_config, opt_config
from .search import search_authorized


class QuestionInput(BaseModel):
    question: str
    user: str


class Message(BaseModel):
    role: str
    content: str


class ChatState(BaseModel):
    messages: List[Message]
    user: str | None = None  # FIXME


async def simple_chatbot_response(input: QuestionInput) -> str:
    results = await search_authorized(question=input.question, user=input.user)
    # sources = "\n".join(set(result.payload['path'] for result in results))
    answer = generate.generate(input.question, results, opt_config)
    # full_response = f"{answer}\n\n{sources}"
    return answer  # full_response


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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
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
):
    answer = await simple_chatbot_response(input)
    return {"messages": [{"content": answer}]}


@app.post("/stream")
async def stream(
        inputs: ChatState = Body(
            ...,
            example=example_messages
        )
):
    async def event_stream():
        question = inputs.messages[-1].content

        results = await search_authorized(user=inputs.user, question=question)
        # sources = "\n".join(set(result.payload['path'] for result in results))

        executor = ThreadPoolExecutor()
        loop = asyncio.get_event_loop()

        def sync_gen():
            for chunk in generate.generate_stream(question, results, opt_config):
                yield chunk

        chunks = await loop.run_in_executor(executor, lambda: list(sync_gen()))

        yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': None}]})}\n\n"

        for chunk in chunks:
            msg = {
                "choices": [
                    {
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(msg)}\n\n"
            # await asyncio.sleep(0.1) # delay for stream check

        # msg = {
        #     "choices": [
        #         {
        #             "delta": {"content": "\n\n" + sources},
        #             "finish_reason": None
        #         }
        #     ]
        # }
        # yield f"data: {json.dumps(msg)}\n\n"

        yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"

    return StreamingResponse(
        event_stream(),
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
):
    return await search_authorized(user=input.user, question=input.question)


@app.post("/ingest")
async def ingest():
    ingestion.index(user_config, opt_config)


@app.get("/test")
async def test():
    return {"message": "Hello World"}
