import json
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from concurrent.futures import ThreadPoolExecutor
import asyncio

import generate
import search


with open("user_config.json", "r") as file:
    user_config = json.load(file)

with open("opt_config.json", "r") as file:
    opt_config = json.load(file)


class QuestionInput(BaseModel):
    question: str


class Message(BaseModel):
    role: str
    content: str


class ChatState(BaseModel):
    messages: List[Message]


async def simple_chatbot_response(input: QuestionInput) -> str:
    question = input.question
    results = search.search(question, user_config, opt_config)
    sources = "\n".join(result.payload['path'] for result in results)
    answer = generate.generate(question, results, opt_config)
    full_response = f"{answer}\n\n{sources}"
    return full_response


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
        results = search.search(question, user_config, opt_config)
        sources = "\n".join(result.payload['path'] for result in results)

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
            await asyncio.sleep(0.1) # delay for stream check

        msg = {
            "choices": [
                {
                    "delta": {"content": "\n\n" + sources},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(msg)}\n\n"

        yield f"data: {json.dumps({'choices':[{'delta':{}, 'finish_reason':'stop'}]})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/test")
async def test():
    return {"message": "Hello World"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
