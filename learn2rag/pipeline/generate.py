from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from .llm import llm

context_template ="""
-----
Source: {source}
Content: 
{content}
"""

def generate(query, search_results, opt_config) -> str:
    context = "\n\n".join([context_template.format(source=result.payload['path'], content=result.payload['content']) for result in search_results])
    system_message = SystemMessagePromptTemplate.from_template(opt_config["prompt"])
    user_message = HumanMessagePromptTemplate.from_template("{question}")
    prompt = ChatPromptTemplate.from_messages([system_message, user_message])
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": query})
    return answer.content


def generate_stream(query, search_results, opt_config):
    context = "\n\n".join([context_template.format(source=result.payload['path'], content=result.payload['content']) for result in search_results])
    system_message = SystemMessagePromptTemplate.from_template(opt_config["prompt"])
    user_message = HumanMessagePromptTemplate.from_template("{question}")
    prompt = ChatPromptTemplate.from_messages([system_message, user_message])

    messages = prompt.format_messages(context=context, question=query)

    for chunk in llm.stream(messages):
        text_chunk = chunk.text()
        if text_chunk:
            yield text_chunk
