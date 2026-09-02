from langchain_core.messages import SystemMessage, HumanMessage
import ast
import logging
import os
import time
from .llm import llm


logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int for %s=%r. Falling back to %s.", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%r. Falling back to %s.", name, value, default)
        return default


def _invoke_llm(messages: list[SystemMessage | HumanMessage], *, purpose: str) -> str:
    if llm is None:
        return ''

    max_attempts = max(1, _env_int("L2R_OLLAMA_MAX_RETRIES", 2))
    retry_sleep_s = max(0.0, _env_float("L2R_OLLAMA_RETRY_BACKOFF_SECONDS", 2.0))

    for attempt in range(1, max_attempts + 1):
        try:
            t0 = time.time()
            logger.info("llm_invoke_start purpose=%s attempt=%d/%d", purpose, attempt, max_attempts)
            response = llm.invoke(messages, stream=False)
            duration_s = time.time() - t0
            logger.info(
                "llm_invoke_done purpose=%s attempt=%d/%d duration_s=%.2f",
                purpose,
                attempt,
                max_attempts,
                duration_s,
            )
            content = response.content
            return content.strip() if isinstance(content, str) else ''
        except Exception as exc:
            logger.warning(
                "llm_invoke_failed purpose=%s attempt=%d/%d error=%s",
                purpose,
                attempt,
                max_attempts,
                exc,
            )
            if attempt < max_attempts and retry_sleep_s > 0:
                time.sleep(retry_sleep_s * attempt)

    logger.error("llm_invoke_give_up purpose=%s attempts=%d", purpose, max_attempts)
    return ''


# future todo: add history handling / add state handling for loops in pipeline

try:
    with open("./learn2rag/pipeline/data/synonyms.txt", "r") as f:
        synonym_list = f.read()
except FileNotFoundError:
    synonym_list = ''

def rewrite_query(user_query: str) -> str:
    if llm is None:
        return ''
    system_message_rewrite_query = """
    You are a query rewriter for a RAG pipeline.
    
    Task:
    - Only rewrite the user's query if there are grammatical or syntactic errors in the query.
    - Rewrite the user's query so that it is optimal for document search.
    - Preserve the original meaning.
    - Do not provide an answer.
    - Return only the rewritten query text.
    """

    content = _invoke_llm([
        SystemMessage(content=system_message_rewrite_query),
        HumanMessage(content=user_query),
    ], purpose="rewrite_query")
    return content


def generate_subqueries(user_query: str, n: int=3) -> list[str]:
    if llm is None:
        return []
    system_message_generate_subqueries = f"""
    You are a query rewriter for a RAG pipeline.
    
    Task:
    - Generate exactly {n} meaningful search queries that cover different aspects of the user's query.
    - Each search query should be short, precise, and optimized for document search.
    - Consider the synonym list given below.
    - No explanations, no numbering, no JSON.
    - Return only a valid Python list of strings.
    
    Synonym list:
    {synonym_list}
    """

    content = _invoke_llm([
        SystemMessage(content=system_message_generate_subqueries),
        HumanMessage(content=user_query),
    ], purpose="generate_subqueries")
    if not content:
        return []

    try:
        result = ast.literal_eval(content.strip())
        if isinstance(result, list):
            return [str(x).strip() for x in result if str(x).strip()]
    except Exception as exc:
        logger.warning("generate_subqueries_parse_failed query=%r error=%s content=%r", user_query, exc, content[:500])

    return []


def generate_keywords(user_query: str, n: int=3) -> list[str]:
    if llm is None:
        return []
    system_message_generate_keywords = f"""
    You are a keyword extractor for a RAG pipeline.
    
    Task:
    - Extract the most relevant search terms from the user's query.
    - Include only obvious search terms; do not invent any.
    - Return a maximum of {n} keywords or keyword phrases.
    - If the search query is likely to contain a product name, do not split it into separate keywords.
    - Consider the synonym list given below.
    - No explanations; just a keyword list.
    - Return only a valid Python list of strings.
    
    Synonym list:
    {synonym_list}
    """

    content = _invoke_llm([
        SystemMessage(content=system_message_generate_keywords),
        HumanMessage(content=user_query),
    ], purpose="generate_keywords")
    if not content:
        return []

    try:
        result = ast.literal_eval(content.strip())
        if isinstance(result, list):
            return [str(x).strip() for x in result if str(x).strip()]
    except Exception as exc:
        logger.warning("generate_keywords_parse_failed query=%r error=%s content=%r", user_query, exc, content[:500])

    return []
