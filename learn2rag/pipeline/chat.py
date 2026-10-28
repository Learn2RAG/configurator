from pydantic import BaseModel


class Message(BaseModel):
    """
    A single message of a conversation as sent by an OpenAI API client.

    Clients send the whole conversation with every request, so a message is
    either a question of the user or an earlier answer of the pipeline.
    """
    role: str
    content: str


ASSISTANT_ROLES = ('assistant', 'model')
'''Roles used by clients for an answer of the pipeline'''
