from collections.abc import Callable
from typing import Optional
from typing_extensions import TypedDict
import json
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import START, StateGraph
import search
import generate

with open("user_config.json", "r") as file:
    user_config = json.load(file)

with open("opt_config.json", "r") as file:
    opt_config = json.load(file)


class State(TypedDict):
    question: str
    context: Optional[str]
    answer: Optional[str]


class StateUpdate(TypedDict, total=False):
    question: Optional[str]
    context: Optional[str]
    answer: Optional[str]


def retrieve(user_config, opt_config) -> Callable[[State], StateUpdate]:
    return lambda state: {"context": search.context_content(state["question"], user_config, opt_config)}


def respond() -> Callable[[State], StateUpdate]:
    return lambda state: {
        "answer": generate.generate(state["question"], state.get("context", ""))
    }


def pipeline() -> CompiledStateGraph:
    graph_builder = StateGraph(State).add_sequence([
        ('retrieve', retrieve(user_config, opt_config)),
        ('generate', respond()),
    ])
    graph_builder.add_edge(START, 'retrieve')
    return graph_builder.compile()