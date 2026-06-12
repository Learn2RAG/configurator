import logging
import os
import httpx
from pydantic import SecretStr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%r. Falling back to %s.", name, value, default)
        return default


def _ollama_timeout() -> httpx.Timeout:
    timeout_s = max(1.0, _env_float("L2R_OLLAMA_TIMEOUT_SECONDS", 90.0))
    connect_s = min(10.0, timeout_s)
    write_s = min(30.0, timeout_s)
    pool_s = min(10.0, timeout_s)
    return httpx.Timeout(timeout=timeout_s, connect=connect_s, read=timeout_s, write=write_s, pool=pool_s)


class LLMClient():
    # ID is used as a key to store in user data, should not be changed
    ID: str
    # LABEL is a display label for user interface
    LABEL: str
    chat_model: BaseChatModel


llms = {}
def llm_client(cls: type[LLMClient]) -> type[LLMClient]:
    llms[cls.ID] = cls; return cls


# First @llm_client would be the default in UI when adding an external model
@llm_client
class OpenAIClient(LLMClient):
    ID = 'ChatOpenAI'
    LABEL = 'OpenAI'

    def __init__(self, *, url: str, token: SecretStr, model: str, proxy: str | None) -> None:
        self.chat_model = ChatOpenAI(
            model=model,
            temperature=0,
            base_url=url,
            api_key=token,
        )


@llm_client
class OllamaClient(LLMClient):
    ID = 'ChatOllama'
    LABEL = 'Ollama'

    def __init__(self, *, url: str, token: str | None, model: str, proxy: str | None) -> None:
        self.chat_model = ChatOllama(
            model=model,
            temperature=0,
            base_url=url,
            client_kwargs={
                'headers': {'Authorization': f'Bearer {token}'} if token else {},
                'proxy': proxy,
                'timeout': _ollama_timeout(),
            },
        )


def chat_model_from_env() -> BaseChatModel:
    default_llm = OpenAIClient
    llm_id = os.environ.get('LLM_API_TYPE', default_llm.ID)
    logger.debug('Using LLM: %s', llm_id)
    llm_kwargs = {
        'url': os.environ.get('LLM_API_URL'),
        'token': os.environ.get('LLM_API_TOKEN') or None,
        'model': os.environ.get('LLM_API_MODEL'),
        'proxy': os.environ.get('LLM_API_PROXY') or None,
    }
    logger.debug('Using LLM args: %s', llm_kwargs)
    return llms[llm_id](**llm_kwargs).chat_model


llm = chat_model_from_env() if 'LLM_API_TYPE' in os.environ else None
if not llm:
    logger.warning('LLM is not configured')
