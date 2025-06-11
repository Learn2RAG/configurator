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

import os
import json
from typing import Annotated, Literal
from typing_extensions import TypedDict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from components import embeddings, llm, prompt, RedisVectorStore
from pipeline import pipeline, State


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


'''
Define Langgraph
'''
def generate_custom_stream(type: Literal["think","normal"], content: str):
    content = "\n"+content+"\n"
    custom_stream_writer = get_stream_writer()
    return custom_stream_writer({type:content})


vector_store = RedisVectorStore(embeddings)

graph = pipeline(
    vector_store=vector_store,
    llm=llm,
    prompt=prompt,
)


def chatbot(state: ChatState):
    print('chatbot state', state)
    response = graph.invoke({'question': state['messages'][-1].content})
    content = f"{response['answer'].answer}\n\n{response['answer'].sources}"
    generate_custom_stream('normal', content)
    return {'messages': [ToolMessage(content=content)]}


graph_builder = StateGraph(ChatState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge("chatbot", END)
graph_builder.add_edge(START, "chatbot")
chat_graph = graph_builder.compile()


'''
Define api processing
'''
app = FastAPI(
    title="Langgraph API",
    description="Langgraph API",
    )

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
