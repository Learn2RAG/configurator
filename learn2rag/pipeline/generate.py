from typing import Any, Generator
import logging
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from qdrant_client.http.models import ScoredPoint
from .llm import llm


profilingLogger = logging.getLogger('profiling')

context_template ="""
-----
Source: {source}
Content: 
{content}
"""

def generate(query: str, search_results: list[ScoredPoint], opt_config: dict[str, Any]) -> Any:
    assert llm is not None
    if hasattr(search_results, "points"):
        search_results = search_results.points
    context = "\n\n".join([context_template.format(source=result.payload['source'], content=result.payload['content']) for result in search_results]) # type: ignore[index]
    system_message = SystemMessagePromptTemplate.from_template(opt_config["prompt"])
    user_message = HumanMessagePromptTemplate.from_template("{question}")
    prompt = ChatPromptTemplate.from_messages([system_message, user_message])
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": query})
    return answer.content


def generate_stream(query: str, search_results: list[ScoredPoint], opt_config: dict[str, Any], request_id: str | None=None) -> Generator[str, None, None]:
    profilingLogger.info('start', extra={'activity': 'generate', 'request_id': request_id})
    assert llm is not None

    if hasattr(search_results, "points"):
        search_results = search_results.points
    context = "\n\n".join([context_template.format(source=result.payload['source'], content=result.payload['content']) for result in search_results]) # type: ignore[index]
    system_message = SystemMessagePromptTemplate.from_template(opt_config["prompt"])
    user_message = HumanMessagePromptTemplate.from_template("{question}")
    prompt = ChatPromptTemplate.from_messages([system_message, user_message])

    messages = prompt.format_messages(context=context, question=query)

    for chunk in llm.stream(messages):
        text_chunk = chunk.text()
        if text_chunk:
            yield text_chunk

    profilingLogger.info('end', extra={'activity': 'generate', 'request_id': request_id})
