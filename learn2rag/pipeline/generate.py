from typing import Any, Generator, Sequence
import logging
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from qdrant_client.http.models import ScoredPoint
from .chat import ASSISTANT_ROLES, Message
from .llm import llm


logger = logging.getLogger(__name__)

context_template ="""
-----
Source: {source}
Content:
{content}
"""

DEFAULT_HISTORY_LENGTH = 5
'''Number of previous answers used if `history_length` is not configured'''


def select_history(history: Sequence[Message], opt_config: dict[str, Any]) -> list[BaseMessage]:
    """
    Turns the conversation history into messages for the prompt.

    Only the last `history_length` answers are kept, together with the questions they
    belong to, because how much history a model can handle depends on the model.
    Roles other than user and assistant are dropped: a client must not be able to
    inject system instructions through the history.
    """
    history_length = opt_config.get("history_length", DEFAULT_HISTORY_LENGTH)
    if history_length <= 0:
        return []

    answers = 0
    start = len(history)
    for index in range(len(history) - 1, -1, -1):
        if history[index].role in ASSISTANT_ROLES:
            if answers == history_length:
                break
            answers += 1
        start = index

    messages: list[BaseMessage] = []
    for message in history[start:]:
        if message.role in ASSISTANT_ROLES:
            messages.append(AIMessage(content=message.content))
        elif message.role == 'user':
            messages.append(HumanMessage(content=message.content))
        else:
            logger.debug("Dropped a history message with role %r", message.role)

    logger.info(
        "history_selected messages=%d of=%d answers=%d history_length=%s",
        len(messages),
        len(history),
        answers,
        history_length,
    )
    return messages


def build_prompt(opt_config: dict[str, Any]) -> ChatPromptTemplate:
    system_message = SystemMessagePromptTemplate.from_template(opt_config["prompt"])
    user_message = HumanMessagePromptTemplate.from_template("{question}")
    # The history is inserted as messages instead of a template, so that braces
    # in earlier answers are not mistaken for template placeholders.
    return ChatPromptTemplate.from_messages([system_message, MessagesPlaceholder('history'), user_message])


def build_context(search_results: Sequence[ScoredPoint]) -> str:
    return "\n\n".join([
        context_template.format(source=result.payload['source'], content=result.payload['content'])
        for result in search_results if result.payload
    ])


def generate(query: str, search_results: Sequence[ScoredPoint], opt_config: dict[str, Any], history: Sequence[Message] = ()) -> Any:
    assert llm is not None
    if hasattr(search_results, "points"):
        search_results = search_results.points
    context = build_context(search_results)

    chain = build_prompt(opt_config) | llm
    answer = chain.invoke({
        "context": context,
        "question": query,
        "history": select_history(history, opt_config),
    })
    return answer.content


def generate_stream(query: str, search_results: list[ScoredPoint], opt_config: dict[str, Any], history: Sequence[Message] = ()) -> Generator[str, None, None]:
    assert llm is not None

    if hasattr(search_results, "points"):
        search_results = search_results.points
    context = build_context(search_results)

    messages = build_prompt(opt_config).format_messages(
        context=context,
        question=query,
        history=select_history(history, opt_config),
    )

    for chunk in llm.stream(messages):
        text_chunk = chunk.text()
        if text_chunk:
            yield text_chunk
