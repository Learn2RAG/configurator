"""
title: Langgraph stream integration
author: bartonzzx
author_url: https://github.com/bartonzzx
description: Integrate langgraph with open webui pipeline
required_open_webui_version: 0.4.3
requirements: none
version: 0.4.3
licence: MIT
"""

# https://github.com/open-webui/pipelines/tree/48ddbec455de76fc43224daf3438537cd8fcde87/examples/pipelines/integrations/langgraph_pipeline


import json
from typing import Annotated, Literal

from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from rag_pipeline import pipeline

import generate
import search


with open("user_config.json", "r") as file:
    user_config = json.load(file)

with open("opt_config.json", "r") as file:
    opt_config = json.load(file)


class QuestionInput(BaseModel):
    question: str


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


async def simple_chatbot_response(input: QuestionInput) -> str:
    question = input.question
    results = search.search(question, user_config, opt_config)
    sources = "\n".join(result.payload['path'] for result in results)
    answer = generate.generate(question, results, opt_config)
    full_response = f"{answer}\n\n{sources}"
    return full_response


def streaming_chatbot(state: ChatState):
    print('chatbot state', state)
    question = state['messages'][-1].content
    results = search.search(question, user_config, opt_config)
    sources = "\n".join(result.payload['path'] for result in results)
    answer = generate.generate(question, results, opt_config)
    content = f"{answer}\n\n{sources}"
    generate_custom_stream('normal', content)
    return {'messages': [ToolMessage(content=content)]}


'''
Define Langgraph
'''
def generate_custom_stream(type: Literal["think","normal"], content: str):
    content = "\n"+content+"\n"
    custom_stream_writer = get_stream_writer()
    return custom_stream_writer({type:content})


graph = pipeline()

graph_builder = StateGraph(ChatState)
graph_builder.add_node("chatbot", streaming_chatbot)
graph_builder.add_edge("chatbot", END)
graph_builder.add_edge(START, "chatbot")
chat_graph = graph_builder.compile()

app = FastAPI()

@app.post("/qanda")
async def qanda(
    input: QuestionInput = Body(
        ...,
        example={
            "question": "What approach did Arjun Singh's campaign use to respond to voters' concerns on social media platforms during the municipal elections in Delhi?"
        }
    )
):
    answer = await simple_chatbot_response(input)
    return {"messages": [{"content": answer}]}


@app.get("/test")
async def test():
    return {"message": "Hello World"}


@app.post("/stream")
async def stream(inputs: ChatState):
    async def event_stream():
        try:
            stream_start_msg = {
                'choices':
                    [
                        {
                            'delta': {},
                            'finish_reason': None
                        }
                    ]
                }

            # Stream start
            yield f"data: {json.dumps(stream_start_msg)}\n\n"

            # Processing langgraph stream response with <think> block support
            async for event in chat_graph.astream(input=inputs, stream_mode="custom"):
                print(event)
                normal_content = event.get("normal", None)

                normal_msg = {
                    'choices':
                    [
                        {
                            'delta':
                            {
                                'content': normal_content,
                            },
                            'finish_reason': None
                        }
                    ]
                }

                yield f"data: {json.dumps(normal_msg)}\n\n"

            # End of the stream
            stream_end_msg = {
                'choices': [
                    {
                        'delta': {},
                        'finish_reason': 'stop'
                    }
                ]
            }
            yield f"data: {json.dumps(stream_end_msg)}\n\n"

        except Exception as e:
            print('Error', e)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
