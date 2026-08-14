from qdrant_client.models import ScoredPoint
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import AnswerRelevancy, ContextRelevance

from .config import opt_config
from .llm import ragas_llm
#import logging

context_relevance_scorer: ContextRelevance | None
answer_relevance_scorer: AnswerRelevancy | None

if ragas_llm is not None:
    embeddings = embedding_factory("huggingface", model = opt_config["embedding_model"]) # type: ignore[no-untyped-call]
    context_relevance_scorer = ContextRelevance(llm = ragas_llm)
    answer_relevance_scorer = AnswerRelevancy(llm = ragas_llm, embeddings = embeddings)
else:
    context_relevance_scorer = None
    answer_relevance_scorer = None

async def judge_context_relevance(question: str, documents: list[ScoredPoint]) -> float:
    assert context_relevance_scorer is not None, "LLM for ragas must be configured"
    contexts = [point.payload["content"] for point in documents if point.payload]
    result = await context_relevance_scorer.ascore(user_input=question, retrieved_contexts=contexts)

    return float(result.value)


async def judge_answer_relevance(question: str, answer: str) -> float:
    assert answer_relevance_scorer is not None, "LLM for ragas must be configured"
    result = await answer_relevance_scorer.ascore(user_input=question, response=answer)

    return float(result.value)