import torch
import os
from openai import OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI
from .base_handler import RequestHandler

class LocalOpenAISDKHandler(RequestHandler):
    def __init__(self, config):
        self.config = config

    def get_client(self, model_name, is_async):
        """Return the appropriate client based on the model and mode (sync/async)."""
        sdk_mapping = {
            "LOCAL": [OpenAI, AsyncOpenAI],
            "INTERN": [OpenAI, AsyncOpenAI]
        }
        sdk_type = self.config['local_openai_sdk_llms'][model_name]
        sdk_config = self.config[sdk_type]

        # Choose the appropriate client class (sync or async)
        ClientClass = sdk_mapping[sdk_type][1 if is_async else 0]
        # Return the initialized client instance based on the SDK type
        return ClientClass(
                api_key=sdk_config[f'{sdk_type}_API_KEY'],
                base_url=sdk_config[f'{sdk_type}_BASE_URL']
            )

    def prepare_parameters(self, request, client):
        """Prepare common parameters for API requests."""
        if request.model_name == "internLM-72B":
            model_name = client.models.list().data[0].id
            messages = [{"role": "user", "content": request.prompt}]
            return {
                "model": f"{model_name}",
                "messages": messages,
                **request.kwargs,
            }
        messages = [{"role": "user", "content": request.prompt}]
        # print(request.kwargs)
        return {
            "model": f"{request.model_name}",
            "messages": messages,
            **request.kwargs,
        }

    def execute_request_sync(self, client, parameters):
        """Synchronous execution of the request."""
        if parameters.get("input"):  # Embedding request
            response = client.embeddings.create(**parameters)
            response_text = response.data[0].embedding
        else:  # Chat request
            response = client.chat.completions.create(**parameters)
            response_text = response.choices[0].message.content
        if response_text:
            return response_text
        else:
            raise ValueError("Empty response from API")

    async def execute_request_async(self, client, parameters):
        """Asynchronous execution of the request."""
        if parameters.get("input"):  # Embedding request
            response = await client.embeddings.create(**parameters)
            response_text = response.data[0].embedding
        else:  # Chat request
            response = await client.chat.completions.create(**parameters)
            response_text = response.choices[0].message.content

        if response_text:
            return response_text
        else:
            raise ValueError("Empty response from API")

    def handle_request(self, request):
        """Synchronous request handler."""
        client = self.get_client(request.model_name, is_async=False)
        parameters = self.prepare_parameters(request, client)
        return self.execute_request_sync(client, parameters)

    async def handle_request_async(self, request):
        """Asynchronous request handler."""
        client = self.get_client(request.model_name, is_async=True)
        parameters = self.prepare_parameters(request, client)
        return await self.execute_request_async(client, parameters)

    
class LocalRequestHandler(RequestHandler):
    def __init__(self, config, pipe):
        self.config = config
        self.pipe = pipe

    def handle_request(self, request):
        if request.model_name == "kolors":
            image = self.pipe(
                prompt=request.prompt,
                negative_prompt="",
                guidance_scale=5.0,
                num_inference_steps=50,
            ).images[0]
        elif request.model_name == "HunyuanDiT":
            image = self.pipe(request.prompt).images[0]
        elif request.model_name == "sd-3.5-large":
            image = self.pipe(
                request.prompt,
                num_inference_steps=28,
                guidance_scale=3.5,
            ).images[0]
        elif request.model_name == 'sd-3.5-large-turbo':
            image = self.pipe(
                request.prompt,
                num_inference_steps=4,
                guidance_scale=0.0,
            ).images[0]
        elif request.model_name == 'cogView-3-plus':
            image = self.pipe(
                prompt=request.prompt,
                guidance_scale=7.0,
                num_images_per_prompt=1,
                num_inference_steps=50,
                width=1024,
                height=1024,
            ).images[0]
        elif request.model_name == 'playground-v2.5':
            image = self.pipe(prompt=request.prompt, num_inference_steps=50, guidance_scale=3).images[0]
        
        save_folder = request.save_folder
        file_name = request.file_name
        if save_folder != '' and file_name != '':
            os.makedirs(save_folder, exist_ok=True)
            image.save(os.path.join(save_folder, file_name))
        return image