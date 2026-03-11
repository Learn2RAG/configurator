import itertools
import json
import os
import sys

from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from concurrent.futures import ThreadPoolExecutor
import asyncio

from .config import user_config, opt_config
from . import generate
from . import search as search_points
from . import ingestion


class QuestionInput(BaseModel):
    question: str


class Message(BaseModel):
    role: str
    content: str


class ChatState(BaseModel):
    messages: List[Message]


async def simple_chatbot_response(input: QuestionInput) -> str:
    question = input.question
    results = search_points.search(question, user_config, opt_config)
    # sources = "\n".join(set(result.payload['path'] for result in results))
    answer = generate.generate(question, results, opt_config)
    # full_response = f"{answer}\n\n{sources}"
    return answer #full_response


example_query = "What approach did Arjun Singh's campaign use to respond to voters' concerns on social media platforms during the municipal elections in Delhi?"
example_messages = {
    "messages": [
        {
            "role": "user",
            "content": example_query
        }
    ]
}


app = FastAPI()

@app.post("/qanda")
async def qanda(
    input: QuestionInput = Body(
        ...,
        example={
            "question": example_query
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

        results = search_points.search(question, user_config, opt_config)
        # sources = "\n".join(set(result.payload['path'] for result in results))

        executor = ThreadPoolExecutor()
        loop = asyncio.get_event_loop()

        def sync_gen():
            for chunk in generate.generate_stream(question, results, opt_config):
                yield chunk

        chunks = await loop.run_in_executor(executor, lambda: list(sync_gen()))

        yield f"data: {json.dumps({'choices':[{'delta':{}, 'finish_reason': None}]})}\n\n"

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

        yield f"data: {json.dumps({'choices':[{'delta':{}, 'finish_reason':'stop'}]})}\n\n"

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
            "question": example_query
        }
    )
):
    search_query = build_search_query(input.question)
    return search_points.search(search_query, user_config, opt_config)


@app.post("/ingest")
async def ingest():
    ingestion.index(user_config, opt_config)


@app.get("/test")
async def test():
    return {"message": "Hello World"}
