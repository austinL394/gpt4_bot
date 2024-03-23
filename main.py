import fastapi_poe as fp
import openai
import os

# Set your OpenAI API key
openai.api_key = str(os.getenv("OPEN_AI_KEY"))

class EchoBot(fp.PoeBot):
    async def get_response(self, request: fp.QueryRequest):
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages= request.query
        )
        yield fp.PartialResponse(text=completion.choices[0].message.content)

if __name__ == "__main__":
    fp.run(EchoBot(), allow_without_key=True)