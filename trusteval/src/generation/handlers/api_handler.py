import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import cohere,replicate,anthropic
from zhipuai import ZhipuAI
from openai import OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI
import os
import time
import functools
import requests
import asyncio

from io import BytesIO
from PIL import Image

from .base_handler import RequestHandler


class OpenAISDKHandler(RequestHandler):
    def __init__(self, config):
        self.config = config

    def get_client(self, model_name, is_async):
        """Return the appropriate client based on the model and mode (sync/async)."""
        sdk_mapping = {
            "AZURE": [AzureOpenAI, AsyncAzureOpenAI],
            "OPENAI": [OpenAI, AsyncOpenAI],
            "DEEPINFRA": [OpenAI, AsyncOpenAI],
            "ZHIPU": [OpenAI, AsyncOpenAI],
            "DEEPSEEK": [OpenAI, AsyncOpenAI],
            "YI": [OpenAI, AsyncOpenAI],
            "QWEN": [OpenAI, AsyncOpenAI],
            "INTERN": [OpenAI, AsyncOpenAI]
        }
        sdk_type = self.config['openai_sdk_llms'][model_name]
        sdk_config = self.config[sdk_type]

        # Choose the appropriate client class (sync or async)
        ClientClass = sdk_mapping[sdk_type][1 if is_async else 0]
        # Return the initialized client instance based on the SDK type
        if sdk_type == "AZURE":
            return ClientClass(
                api_key=sdk_config['AZURE_API_KEY'],
                api_version=sdk_config['AZURE_API_VERSION'],
                azure_endpoint=sdk_config['AZURE_ENDPOINT'],
            )
        else:
            return ClientClass(
                api_key=sdk_config[f'{sdk_type}_API_KEY'],
                base_url=sdk_config[f'{sdk_type}_BASE_URL']
            )

    def prepare_parameters(self, request, client):
        """Prepare common parameters for API requests."""
        if request.model_name == "text-embedding-ada-002":
            prompt = request.prompt.replace('\n', ' ')
            return {"input": [prompt], "model": request.model_name}
        elif request.model_name == "internLM-72B":
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

    
class OtherSDKHandler(RequestHandler):
    def __init__(self, config):
        self.config = config

    def get_client(self, model_name, is_async):
        """Return the appropriate client based on the model and mode (sync/async)."""
        sdk_mapping = {
            "ANTHROPIC": [anthropic.Anthropic, anthropic.AsyncAnthropic],
            "GOOGLE": [genai.GenerativeModel, genai.GenerativeModel],  # Same class for sync/async
            "COHERE": [cohere.Client, cohere.Client],  # Same class for sync/async
        }
        sdk_type = self.config['other_sdk_llms'][model_name]
        
        # Select the appropriate client class (sync or async)
        ClientClass = sdk_mapping[sdk_type][1 if is_async else 0]
        
        # Initialize the client based on the SDK type
        if sdk_type == "ANTHROPIC":
            return ClientClass(api_key=self.config['ANTHROPIC']['ANTHROPIC_API_KEY'])
        elif sdk_type == "GOOGLE":
            genai.configure(api_key=self.config['GOOGLE']['GOOGLE_API_KEY'])
            ClientClass = ClientClass(model_name=model_name)
            return ClientClass
        elif sdk_type == "COHERE":
            return ClientClass(api_key=self.config['COHERE']['COHERE_API_KEY'])
        else:
            raise ValueError(f"Invalid SDK for {model_name}")

    def prepare_parameters(self, request):
        """Prepare the parameters for each SDK."""
        if self.config['other_sdk_llms'][request.model_name] == "ANTHROPIC":
            return {
                "model": f"{request.model_name}",
                "messages": [
                    {"role": "user", "content": request.prompt}
                ],
                "max_tokens": 2048,
                **request.kwargs,
            }
        elif self.config['other_sdk_llms'][request.model_name] == "GOOGLE":
            safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            config = genai.GenerationConfig(**request.kwargs)
            return {
                "contents": [request.prompt],
                "generation_config": config,
                "safety_settings":safety_settings,
            }
        elif self.config['other_sdk_llms'][request.model_name] == "COHERE":
            return {
                "message": request.prompt,
                "model": f"{request.model_name}",
                **request.kwargs,
            }
        else:
            raise ValueError(f"Invalid SDK for {request.model_name}")

    def execute_request_sync(self, client, parameters, sdk_type):
        """Synchronous execution of the request."""
        if sdk_type == "ANTHROPIC":
            response = client.messages.create(**parameters)
            return response.content[0].text
        elif sdk_type == "GOOGLE":
            # check **parameters
            response = client.generate_content(**parameters)
            return response.text
        elif sdk_type == "COHERE":
            response = client.chat(**parameters)
            return response.text
        else:
            raise ValueError(f"Invalid SDK type: {sdk_type}")

    async def execute_request_async(self, client, parameters, sdk_type):
        """Asynchronous execution of the request."""
        if sdk_type == "ANTHROPIC":
            response = await client.messages.create(**parameters)
            return response.content[0].text
        elif sdk_type == "GOOGLE":
            response = await client.generate_content_async(**parameters)
            try:
                return response.text
            except ValueError:
                if hasattr(response, "candidates") and response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                    if finish_reason.name in ['BLOCKLIST',"SAFETY",'PROHIBITED_CONTENT']:
                        return "I'm sorry!" 
                    else:
                        return None
                else:
                    return None
        elif sdk_type == "COHERE":
            response = await client.chat(**parameters)
            return response.text
        else:
            raise ValueError(f"Invalid SDK type: {sdk_type}")


    def handle_request(self, request):
        """Synchronous request handler."""
        sdk_type = self.config['other_sdk_llms'][request.model_name]
        client = self.get_client(request.model_name, is_async=False)
        parameters = self.prepare_parameters(request)
        return self.execute_request_sync(client, parameters, sdk_type)

    
    async def handle_request_async(self, request):
        """Asynchronous request handler."""
        sdk_type = self.config['other_sdk_llms'][request.model_name]
        client = self.get_client(request.model_name, is_async=True)
        parameters = self.prepare_parameters(request)
        return await self.execute_request_async(client, parameters, sdk_type)

    
class OpenAIVisionSDKHandler(RequestHandler):
    def __init__(self, config):
        self.config = config

    def get_client(self, model_name, is_async=False):
        """Return the appropriate client based on the model and mode (sync/async)."""
        client_mapping = {
            "OPENAI": [OpenAI, AsyncOpenAI],
            "AZURE": [AzureOpenAI, AsyncAzureOpenAI],
            "ZHIPU": [OpenAI, AsyncOpenAI],
            "DEEPINFRA": [OpenAI, AsyncOpenAI],
            "QWEN": [OpenAI, AsyncOpenAI],
            "INTERN": [OpenAI, None],
            'OPENROUTER': [OpenAI, AsyncOpenAI]
        }
        sdk_type = self.config['openai_sdk_vlms'][model_name]
        
        
        
        # Choose the correct client class (sync or async)
        ClientClass = client_mapping[sdk_type][1 if is_async else 0]
        
        # Initialize the client based on the SDK type
        if sdk_type == "AZURE":
            return ClientClass(
                api_key=self.config['AZURE']['AZURE_API_KEY'],
                api_version=self.config['AZURE']['AZURE_API_VERSION'],
                azure_endpoint=self.config['AZURE']['AZURE_ENDPOINT']
            )
        else:
            return ClientClass(
                api_key=self.config[f'{sdk_type}'][f'{sdk_type}_API_KEY'],
                base_url=self.config[f'{sdk_type}'][f'{sdk_type}_BASE_URL']
            )

    def prepare_parameters(self, request, image_messages, client):
        """Prepare the parameters for API requests."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                    *image_messages,  # Attach image messages
                ]
            }
        ]
        if request.model_name == "internLM-72B":
            model_name = client.models.list().data[0].id
            return {
                "model": f"{model_name}",
                "messages": messages,
                **request.kwargs,
            }
        # Prepare parameters for the API request
        return {
            "model": f"{request.model_name}",
            "messages": messages,
            **request.kwargs,
        }

    def execute_request_sync(self, client, parameters):
        """Synchronous execution of the request."""
        response = client.chat.completions.create(**parameters)
        response_text = response.choices[0].message.content
        if response_text:
            return response_text
        else:
            raise ValueError("Empty response from API")

    def handle_request(self, request):
        """Synchronous request handler."""
        image_urls = request.image_urls
        image_messages = self.generate_image_messages(self, image_urls)

        client = self.get_client(request.model_name, is_async=False)
        parameters = self.prepare_parameters(request, image_messages, client)

        return self.execute_request_sync(client, parameters)

    async def execute_request_async(self, client, parameters):
        """Asynchronous execution of the request."""
        response = await client.chat.completions.create(**parameters)
        response_text = response.choices[0].message.content
        if response_text:
            return response_text
        else:
            raise ValueError("Empty response from API")

    async def handle_request_async(self, request):
        """Asynchronous request handler."""
        image_urls = request.image_urls
        image_messages = self.generate_image_messages(self, image_urls)

        client = self.get_client(request.model_name, is_async=True)
        parameters = self.prepare_parameters(request, image_messages, client)

        return await self.execute_request_async(client, parameters)

class OtherVisionSDKHandler(RequestHandler):
    def __init__(self, config):
        self.config = config

    def prepare_anthropic_messages(self, request, image_urls):
        """Prepare messages for Anthropic API with text and images."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                ]
            }
        ]
        for image_url in image_urls:
            image = Image.open(image_url)
            media_type = f"image/{self.check_image_format(image_url)}"
            base64_image = self.encode_image_to_base64(image)
            messages[0]["content"].append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image,
                    }
                }
            )
        return messages

    def prepare_google_parameters(self, request, image_urls):
        """Prepare parameters for Google API."""
        parameters = [request.prompt]
        for image_url in image_urls:
            image = Image.open(image_url)
            parameters.append(image)
        return parameters

    def prepare_replicate_input(self, request, image_urls):
        """Prepare input for Replicate API."""
        if isinstance(image_urls, list):
            image_urls = image_urls[0]
        return {
            'text': request.prompt,
            'image': open(image_urls, 'rb'),
            "decoding_strategy": "top-p-sampling",
            "temperature": 0.0
        }

    def get_client(self, sdk_type, is_async):
        """Helper to get the correct client for sync or async requests."""
        if sdk_type == "ANTHROPIC":
            return anthropic.AsyncAnthropic(api_key=self.config['ANTHROPIC']['ANTHROPIC_API_KEY']) if is_async else anthropic.Anthropic(api_key=self.config['ANTHROPIC']['ANTHROPIC_API_KEY'])
        elif sdk_type == "REPLICATE":
            return None  # Replicate uses `replicate.run` or `replicate.async_run` directly

    def execute_request_sync(self, client, parameters, sdk_type):
        """Synchronous execution of the request."""
        if sdk_type == "ANTHROPIC":
            response = client.messages.create(**parameters)
            response_text = response.content[0].text
        elif sdk_type == "GOOGLE":
            response = client.generate_content(parameters)
            response_text = response.text
        elif sdk_type == "REPLICATE":
            response_text = replicate.run(
                "zsxkib/idefics3:b06f5f6b6249b27d0b00d1b794240e5641190d1582ad68c40ef53778459bb593",
                input=parameters
            )
        else:
            raise ValueError(f"Invalid SDK type: {sdk_type}")

        if response_text:
            return response_text
        else:
            raise ValueError("Empty response from API")

    async def execute_request_async(self, client, parameters, sdk_type):
        """Asynchronous execution of the request."""
        if sdk_type == "ANTHROPIC":
            response = await client.messages.create(**parameters)
            response_text = response.content[0].text
        elif sdk_type == "GOOGLE":
            response = await client.generate_content_async(contents=parameters)
            response_text = response.text
        elif sdk_type == "REPLICATE":
            response_text = await replicate.async_run(
                "zsxkib/idefics3:b06f5f6b6249b27d0b00d1b794240e5641190d1582ad68c40ef53778459bb593",
                input=parameters
            )
        else:
            raise ValueError(f"Invalid SDK type: {sdk_type}")

        if response_text:
            return response_text
        else:
            raise ValueError("Empty response from API")

    
    def handle_request(self, request):
        """Synchronous request handler."""
        image_urls = request.image_urls
        sdk_type = self.config['other_sdk_vlms'][f'{request.model_name}']

        if sdk_type == "ANTHROPIC":
            messages = self.prepare_anthropic_messages(request, image_urls)
            parameters = {
                "model": f"{request.model_name}",
                "messages": messages,
                "max_tokens": 2048,
                **request.kwargs,
            }
            client = self.get_client('ANTHROPIC', is_async=False)

        elif sdk_type == "GOOGLE":
            parameters = self.prepare_google_parameters(request, image_urls)
            genai.configure(api_key=self.config['GOOGLE']['GOOGLE_API_KEY'])
            config = genai.GenerationConfig(**request.kwargs)
            model = genai.GenerativeModel(model_name=request.model_name, generation_config=config)
            client = model  # Handle async logic later for Google

        elif sdk_type == "REPLICATE":
            parameters = self.prepare_replicate_input(request, image_urls)
            os.environ["REPLICATE_API_TOKEN"] = self.config['REPLICATE']['REPLICATE_API_TOKEN']
            client = self.get_client('REPLICATE', is_async=False)

        return self.execute_request_sync(client, parameters, sdk_type)

    
    async def handle_request_async(self, request):
        """Asynchronous request handler."""
        image_urls = request.image_urls
        sdk_type = self.config['other_sdk_vlms'][f'{request.model_name}']

        if sdk_type == "ANTHROPIC":
            messages = self.prepare_anthropic_messages(request, image_urls)
            parameters = {
                "model": f"{request.model_name}",
                "messages": messages,
                "max_tokens": 2048,
                **request.kwargs,
            }
            client = self.get_client('ANTHROPIC', is_async=True)

        elif sdk_type == "GOOGLE":
            parameters = self.prepare_google_parameters(request, image_urls)
            genai.configure(api_key=self.config['GOOGLE']['GOOGLE_API_KEY'])
            config = genai.GenerationConfig(**request.kwargs)
            model = genai.GenerativeModel(model_name=request.model_name, generation_config=config)
            client = model  # Handle async logic later for Google

        elif sdk_type == "REPLICATE":
            parameters = self.prepare_replicate_input(request, image_urls)
            os.environ["REPLICATE_API_TOKEN"] = self.config['REPLICATE']['REPLICATE_API_TOKEN']
            client = self.get_client('REPLICATE', is_async=True)

        return await self.execute_request_async(client, parameters, sdk_type)
        
class T2IHandler(RequestHandler):
    def __init__(self, config):
        self.config = config
    
    def handle_request(self, request):
        if self.config['t2i_models'][f'{request.model_name}'] == "OPENAI":
            client = OpenAI(
                api_key=self.config['OPENAI']['OPENAI_API_KEY'],
                base_url=self.config['OPENAI']['OPENAI_BASE_URL']
            )
            result = client.images.generate(
                prompt=request.prompt,
                model=request.model_name,
            )
            save_folder = request.save_folder
            file_name = request.file_name
            if request.save_folder == '' and request.file_name == '':
                response = requests.get(result.data[0].url)
                image = Image.open(BytesIO(response.content))
                return image
            return self.download_image(result.data[0].url, f"{save_folder}/{file_name}")

        
        elif self.config['t2i_models'][f'{request.model_name}'] == "REPLICATE":
            os.environ["REPLICATE_API_TOKEN"] = self.config['REPLICATE']['REPLICATE_API_TOKEN']
            if request.model_name == 'flux-1.1-pro':
                output = replicate.run(
                    "black-forest-labs/flux-1.1-pro",
                    input={
                        "prompt": request.prompt,
                        "aspect_ratio": "1:1",
                        "output_format": "webp",
                        "output_quality": 80,
                        "safety_tolerance": 5,
                        "prompt_upsampling": True
                    }
                )
                if request.save_folder == '' and request.file_name == '':
                    response = requests.get(output)
                    image = Image.open(BytesIO(response.content))
                    return image
                return self.download_image(output, f"{request.save_folder}/{request.file_name}")
            if request.model_name == 'flux_schnell':
                output = replicate.run(
                    "black-forest-labs/flux-schnell",
                    input={
                        "prompt": request.prompt,
                        "go_fast": True,
                        "megapixels": "1",
                        "num_outputs": 1,
                        "aspect_ratio": "1:1",
                        "output_format": "png",
                        "output_quality": 80,
                        "num_inference_steps": 4
                    }
                )
                if request.save_folder == '' and request.file_name == '':
                    response = requests.get(output)
                    image = Image.open(BytesIO(response.content))
                    return image
                return self.download_image(output, f"{request.save_folder}/{request.file_name}")
            if request.model_name == 'playgroundai/playground-v2.5-1024px-aesthetic':
                output = replicate.run(
                    "playgroundai/playground-v2.5-1024px-aesthetic:a45f82a1382bed5c7aeb861dac7c7d191b0fdf74d8d57c4a0e6ed7d4d0bf7d24",
                    input={
                        "width": 1024,
                        "height": 1024,
                        "prompt": request.prompt,
                        "scheduler": "DPMSolver++",
                        "num_outputs": 1,
                        "guidance_scale": 3,
                        "apply_watermark": True,
                        "negative_prompt": "ugly, deformed, noisy, blurry, distorted",
                        "prompt_strength": 0.8,
                        "num_inference_steps": 25
                    }
                )
                if request.save_folder == '' and request.file_name == '':
                    response = requests.get(output[0])
                    image = Image.open(BytesIO(response.content))
                    return image
                return self.download_image(output[0], f"{request.save_folder}/{request.file_name}")
        elif self.config['t2i_models'][f'{request.model_name}'] == "ZHIPU":
            client = ZhipuAI(
                api_key=self.config['ZHIPU']['ZHIPU_API_KEY'],
            )
            response = client.images.generations(
                model=request.model_name,
                prompt=request.prompt,
            )
            url = response.data[0].url
            if request.save_folder == '' and request.file_name == '':
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                return image
            return self.download_image(url, f"{request.save_folder}/{request.file_name}")
