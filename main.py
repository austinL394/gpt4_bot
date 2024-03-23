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


class ClaudeBot(fp.PoeBot):
    async def get_response(self, request: fp.QueryRequest):
       
        bedrock_runtime = boto3.client(service_name='bedrock-runtime', 
            region_name='us-east-1', 
            aws_access_key_id=AWS_KEY, 
            aws_secret_access_key=AWS_SECRET_KEY
        )
        # modelId obtained from printing out modelIds in previous step
        modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'

        ### parameters for the LLM to control text-generation
        # temperature increases randomness as it increases
        temperature = 0.5
        # top_p increases more word choice as it increases
        top_p = 1
        # maximum number of tokens to generate in the output
        max_tokens_to_generate = 250


        system_prompt = "All your responses should be in Haiku form"

        messages = request.query

        body = json.dumps({
                    "messages": messages,
                    "system": system_prompt,
                    "max_tokens": max_tokens_to_generate,
                    "temperature": temperature,
                    "top_p": top_p,
                    "anthropic_version": "bedrock-2023-05-31"
        })

        response = bedrock_runtime.invoke_model(body=body, modelId=modelId, accept="application/json", contentType="application/json")

        response_body = json.loads(response.get('body').read())
        result = response_body.get('content', '')
        yield fp.PartialResponse(text=result)

if __name__ == "__main__":
    if model_type == "GPT":
        fp.run(GPTBot(), allow_without_key=True)
    else:
        fp.run(ClaudeBot(), allow_without_key=True)