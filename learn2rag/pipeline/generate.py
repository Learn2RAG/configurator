"""Canonical application-level prompt construction for Learn2RAG generation."""

from collections.abc import Callable
from typing import Any, Generator

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import BaseMessage
from qdrant_client.http.models import ScoredPoint

from .llm import llm


context_template ="""
-----
Source: {source}
Content: 
{content}
"""


def build_prompt_messages(
    query: str,
    search_results: list[ScoredPoint],
    opt_config: dict[str, Any],
    *,
    source_transform: Callable[[Any], Any] | None = None,
) -> list[BaseMessage]:
    """Build the canonical LangChain messages used for RAG generation.

    Normal callers preserve the existing raw Source behavior. Public callers
    may transform source labels before invocation while retaining the same
    context content and retrieval order.
    """
    if hasattr(search_results, "points"):
        search_results = search_results.points

    context_parts = []
    for result in search_results:
        source = result.payload['source']  # type: ignore[index]
        if source_transform is not None:
            source = source_transform(source)
        context_parts.append(context_template.format(
            source=source,
            content=result.payload['content'],  # type: ignore[index]
        ))
    context = "\n\n".join(context_parts)
    system_message = SystemMessagePromptTemplate.from_template(opt_config["prompt"])
    user_message = HumanMessagePromptTemplate.from_template("{question}")
    prompt = ChatPromptTemplate.from_messages([system_message, user_message])
    return prompt.format_messages(context=context, question=query)


def invoke_prompt_messages(messages: list[BaseMessage]) -> Any:
    """Invoke the configured model with the already-inspected messages.

    Accepting built objects prevents prompt reconstruction from drifting away
    from the exact application-level content sent to the model boundary.
    """
    assert llm is not None
    answer = llm.invoke(messages)
    return answer.content


def generate(query: str, search_results: list[ScoredPoint], opt_config: dict[str, Any]) -> Any:
    messages = build_prompt_messages(query, search_results, opt_config)
    return invoke_prompt_messages(messages)


def generate_stream(query: str, search_results: list[ScoredPoint], opt_config: dict[str, Any]) -> Generator[str, None, None]:
    assert llm is not None
    messages = build_prompt_messages(query, search_results, opt_config)

    for chunk in llm.stream(messages):
        text_chunk = chunk.text()
        if text_chunk:
            yield text_chunk
