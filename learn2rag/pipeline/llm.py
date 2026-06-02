import logging
import os
from pydantic import SecretStr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


logger = logging.getLogger(__name__)


class LLMClient():
    ID: str
    '''A key stored in user data, must not be changed'''

    LABEL: str | None
    '''
    A display label for the interface.
    If None, the option would be excluded from the interface.
    '''

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
