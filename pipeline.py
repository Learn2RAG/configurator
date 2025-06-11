from collections.abc import Callable

from langchain.chains.openai_functions.qa_with_structure import AnswerWithSources
from langchain_core.documents import Document
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


class State(TypedDict):
    question: str
    context: List[Document]
    answer: AnswerWithSources


class StateUpdate(TypedDict, total=False):
    question: str
    context: List[Document]
    answer: AnswerWithSources


def retrieve(vector_store: VectorStore) -> Callable[[State], StateUpdate]:
    assert vector_store is not None, 'vector_store should be defined'
    return lambda state: {"context": vector_store.similarity_search(state["question"])}


def generate(llm: BaseLanguageModel[BaseMessage], prompt: ChatPromptTemplate) -> Callable[[State], StateUpdate]:
    structured_llm = llm.with_structured_output(AnswerWithSources)
    return lambda state: {'answer': structured_llm.invoke(prompt.invoke({
        'question': state['question'],
        'context': "\n\n".join(doc.page_content for doc in state["context"]),
    }))} # type: ignore[typeddict-item]  # FIXME


def pipeline(vector_store: VectorStore, llm: BaseLanguageModel[BaseMessage], prompt: ChatPromptTemplate) -> CompiledStateGraph:
    graph_builder = StateGraph(State).add_sequence([
        ('retrieve', retrieve(vector_store=vector_store)),
        ('generate', generate(llm=llm, prompt=prompt)),
    ])
    graph_builder.add_edge(START, 'retrieve')
    return graph_builder.compile()
