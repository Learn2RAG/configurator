from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from llm import llm

def generate(query, search_results) -> str:
    template = """Kontext:
    {context}

    Beantworte die folgende Frage:
    {question}
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    context = "\n\n".join([doc.page_content for doc in search_results])

    return chain.run(context=context, question=query)