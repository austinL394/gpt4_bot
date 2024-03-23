import fastapi_poe as fp
import openai
import os
import boto3
import json

openai.api_key = str(os.getenv("OPEN_AI_KEY"))
model_type = str(os.getenv("MODEL_TYPE"))


class GPTBot(fp.PoeBot):
    async def get_response(self, request: fp.QueryRequest):
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages= request.query
        )
        yield fp.PartialResponse(text=completion.choices[0].message.content)

if __name__ == "__main__":
        fp.run(GPTBot(), allow_without_key=True)