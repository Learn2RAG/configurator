from typing import Any, Generator
import logging
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from qdrant_client.http.models import ScoredPoint
from .llm import llm
from .judges import judge_answer_relevance


context_template ="""
-----
Source: {source}
Content: 
{content}
"""

async def generate(query: str, search_results: list[ScoredPoint], opt_config: dict[str, Any]) -> Any:
    assert llm is not None
    if hasattr(search_results, "points"):
        search_results = search_results.points
    context = "\n\n".join([context_template.format(source=result.payload['source'], content=result.payload['content']) for result in search_results]) # type: ignore[index]
    system_message = SystemMessagePromptTemplate.from_template(opt_config["prompt"])
    user_message = HumanMessagePromptTemplate.from_template("{question}")
    prompt = ChatPromptTemplate.from_messages([system_message, user_message])
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": query}).content

    logging.info("\nFirst answer:")
    logging.info(answer)
    if opt_config["answer_relevance_judge"]:
        answer_relevance_score = await judge_answer_relevance(
            question=query,
            answer=answer,
        )
        logging.info("Answer relevance score: %s", answer_relevance_score)
        if answer_relevance_score < opt_config["judge_threshold"]:
            answer = chain.invoke({
                "context": context,
                "question": query,
            }).content
            logging.info("\nRetry answer:")
            logging.info(answer)
    return answer


def generate_stream(query: str, search_results: list[ScoredPoint], opt_config: dict[str, Any]) -> Generator[str, None, None]:
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
