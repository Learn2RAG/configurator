import logging
import os
from pydantic import SecretStr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from typing import Any, ClassVar


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
'''A dict holding supported LLM client classes'''


def llm_client(cls: type[LLMClient]) -> type[LLMClient]:
    llms[cls.ID] = cls; return cls


# First @llm_client would be the default in UI when adding an external model
@llm_client
class OpenAIClient(LLMClient):
    '''A LLM client based on OpenAI API'''
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
    '''A LLM client based on Ollama API'''
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


class TestFakeChatModel(BaseChatModel):
    '''
    A mock BaseChatModel implementation.
    Responds with the full content of the system prompt.
    '''
    hint: ClassVar[str] = 'This is an internal model used for testing only.'

    @property
    def _llm_type(self) -> str: return 'test_fake_chat_model'

    def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any
    ) -> ChatResult:
        assert isinstance(messages[0], SystemMessage)
        content = f'{self.hint} {messages[0].content}'
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=content)),
            ],
        )


@llm_client
class FakeClient(LLMClient):
    '''A mock LLM client to use only in tests'''
    ID = 'ChatFake'
    LABEL = None

    def __init__(self, *, url: str, token: str | None, model: str, proxy: str | None) -> None:
        self.chat_model = TestFakeChatModel()


def chat_model_from_env() -> BaseChatModel:
    '''Returns an instance of LLM client based on the environment variables'''
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
