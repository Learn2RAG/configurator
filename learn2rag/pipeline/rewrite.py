from langchain_core.messages import SystemMessage, HumanMessage
import ast
from .llm import llm


# future todo: add history handling / add state handling for loops in pipeline

try:
    with open("./learn2rag/pipeline/data/synonyms.txt", "r") as f:
        synonym_list = f.read()
except FileNotFoundError:
    synonym_list = ''

def rewrite_query(user_query: str) -> str:
    system_message_rewrite_query = """
    You are a query rewriter for a RAG pipeline.
    
    Task:
    - Only rewrite the user's query if there are grammatical or syntactic errors in the query.
    - Rewrite the user's query so that it is optimal for document search.
    - Preserve the original meaning.
    - Do not provide an answer.
    - Return only the rewritten query text.
    """

    response = llm.invoke([
        SystemMessage(content=system_message_rewrite_query),
        HumanMessage(content=user_query),
    ])

    return response.content.strip()


def generate_subqueries(user_query: str, n: int=3) -> list[str]:
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

    response = llm.invoke([
        SystemMessage(content=system_message_generate_subqueries),
        HumanMessage(content=user_query),
    ])

    try:
        result = ast.literal_eval(response.content.strip())
        if isinstance(result, list):
            return [str(x).strip() for x in result if str(x).strip()]
    except Exception:
        pass

    return []


def generate_keywords(user_query: str, n: int=3) -> list[str]:
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

    response = llm.invoke([
        SystemMessage(content=system_message_generate_keywords),
        HumanMessage(content=user_query),
    ])

    try:
        result = ast.literal_eval(response.content.strip())
        if isinstance(result, list):
            return [str(x).strip() for x in result if str(x).strip()]
    except Exception:
        pass

    return []
