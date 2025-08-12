from fastapi import FastAPI, Body
from pydantic import BaseModel
import json
import search
import generate

app = FastAPI()

with open("user_config.json", "r") as file:
    user_config = json.load(file)

with open("opt_config.json", "r") as file:
    opt_config = json.load(file)


class QuestionInput(BaseModel):
    question: str


async def simple_chatbot_response(input: QuestionInput) -> str:
    question = input.question
    results = search.search(question, user_config, opt_config)
    sources = "\n".join(result.payload['path'] for result in results)
    answer = generate.generate(question, results)
    full_response = f"{answer}\n\n{sources}"
    return full_response


@app.post("/chat")
async def chat(
    input: QuestionInput = Body(
        ...,
        example={
            "question": "What approach did Arjun Singh's campaign use to respond to voters' concerns on social media platforms during the municipal elections in Delhi?"
        }
    )
):
    answer = await simple_chatbot_response(input)
    return {"messages": [{"content": answer}]}


@app.get("/test")
async def test():
    return {"message": "Hello World"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)