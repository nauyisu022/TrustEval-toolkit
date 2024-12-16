import asyncio
import torch
import os
import sys
from typing import List, Any, Callable, Dict
try:
    from diffusers import KolorsPipeline, HunyuanDiTPipeline, StableDiffusion3Pipeline, CogView3PlusPipeline, DiffusionPipeline
except ImportError:
    pass
    #print("\033[94mDiffusers module is not installed, skipping related imports.\033[0m")

from .factories import ModelRequestFactory, RequestHandlerFactory
from tqdm.asyncio import tqdm_asyncio
import asyncio
from .utils.tools import retry_on_failure,retry_on_failure_async

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

MODEL_NAME_MAPPINGS = {
    'gpt-4o': 'gpt-4o',
    'gpt-4o-mini': 'gpt-4o-mini',
    "gpt-3.5-turbo":"gpt-3.5-turbo",
    'text-embedding-ada-002': 'text-embedding-ada-002',
    'glm-4': 'glm-4',
    'glm-4v': 'glm-4v',
    'glm-4-plus': 'glm-4-plus',
    'glm-4v-plus': 'glm-4v-plus',
    'internLM-72B': 'internLM-72B',
    'claude-3.5-sonnet': 'claude-3-5-sonnet-20240620',
    'claude-3-haiku': 'claude-3-haiku-20240307',
    'claude-3.5-haiku': 'later',
    'claude-3.5-opus': 'later',
    'claude-3-opus': 'claude-3-opus-20240229',
    'gemini-1.5-pro': 'gemini-1.5-pro',
    'gemini-1.5-flash': 'gemini-1.5-flash',
    'command-r-plus': 'command-r-plus-08-2024',
    'command-r': 'command-r-08-2024',
    'llama-3-8B': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama-3-70B': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'llama-3.1-70B': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'llama-3.1-8B': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama-3.2-90B-V': 'meta-llama/Llama-3.2-90B-Vision-Instruct',
    'llama-3.2-11B-V': 'meta-llama/Llama-3.2-11B-Vision-Instruct',
    'llama-2-13B': 'meta-llama/Llama-2-13b-chat-hf',
    'gemma-2-27B': 'google/gemma-2-27b-it',
    'qwen-2.5-72B': 'Qwen/Qwen2.5-72B-Instruct',
    'qwen-vl-max-0809': 'qwen-vl-max-0809',
    'qwen-2-vl-72B': 'qwen-vl-max-0809',
    'mistral-8x22B': 'mistralai/Mixtral-8x22B-Instruct-v0.1',
    'mistral-8x7B': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'mistral-7B': 'mistralai/Mistral-7B-Instruct-v0.3',
    'deepseek-chat': 'deepseek-chat',
    'yi-lightning': 'yi-lightning',
    'dalle3': 'dall-e-3',
    'flux-1.1-pro': 'flux-1.1-pro',
    'playground-v2.5': 'playground-v2.5',
    'cogview-3-plus': 'cogView-3-plus',
    'kolors': 'kolors',
    'sd-3.5-large': 'sd-3.5-large',
    'sd-3.5-large-turbo': 'sd-3.5-large-turbo',
    'HunyuanDiT': 'HunyuanDiT',
}

class ModelService:
    def __init__(self, request_type='llm', handler_type='api', model_name=None, config_path=os.path.join(PROJECT_ROOT, 'src/config/config.yaml'), **kwargs):
        self.request_type = request_type
        self.handler_type = handler_type
        self.model_name = MODEL_NAME_MAPPINGS.get(model_name, model_name)
        self.config_path = config_path
        self.request_factory = ModelRequestFactory()
        self.handler_factory = RequestHandlerFactory()
        self.kwargs = kwargs
        self.pipe = self._initialize_pipeline()

    def _initialize_pipeline(self):
        if self.request_type == 't2i' and self.handler_type == 'local':
            if self.model_name == "HunyuanDiT":
                return HunyuanDiTPipeline.from_pretrained(
                    "Tencent-Hunyuan/HunyuanDiT-Diffusers", torch_dtype=torch.float16
                )
            elif self.model_name == "kolors":
                return KolorsPipeline.from_pretrained(
                    "Kwai-Kolors/Kolors-diffusers", torch_dtype=torch.float16, variant="fp16"
                )
            elif self.model_name == 'sd-3.5-large':
                return StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
            elif self.model_name == 'sd-3.5-large-turbo':
                return StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.bfloat16)
            elif self.model_name == "cogView-3-plus":
                pipe = CogView3PlusPipeline.from_pretrained("THUDM/CogView3-Plus-3B", torch_dtype=torch.float16)
                # Enable it to reduce GPU memory usage
                pipe.enable_model_cpu_offload()
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
                return pipe
            elif self.model_name == "playground-v2.5":
                return DiffusionPipeline.from_pretrained(
                    "playgroundai/playground-v2.5-1024px-aesthetic",
                    torch_dtype=torch.float16,
                    variant="fp16",
                ).to("cuda")
        return None

    def _format_messages(self, conversation_history):
        formatted_messages = ""
        for message in conversation_history:
            role = message["role"]
            content = message["content"]
            formatted_messages += f"{role}: {content}\n\n"
        return formatted_messages

    def _process_single(self, prompt, **kwargs):
        if "system_prompt" in kwargs:
            system_prompt = kwargs.pop("system_prompt")
            conversation_history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            prompt = self._format_messages(conversation_history)
        request = self.request_factory.create_request(self.request_type, self.model_name, prompt, **self.kwargs, **kwargs)
        handler = self.handler_factory.create_handler(self.request_type, self.handler_type, self.model_name, self.config_path, self.pipe)
        return request.send_request(handler)

    async def _process_single_async(self, prompt, **kwargs):
        if "system_prompt" in kwargs:
            system_prompt = kwargs.pop("system_prompt")
            conversation_history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            prompt = self._format_messages(conversation_history)
        request = await self.request_factory.create_request_async(self.request_type, self.model_name, prompt, **self.kwargs, **kwargs)
        handler = await self.handler_factory.create_handler_async(self.request_type, self.handler_type, self.model_name, self.config_path)
        return await request.send_request_async(handler)

    def _process_multiturn(self, prompts):
        if self.request_type == "t2i":
            request = self.request_factory.create_request(self.request_type, self.model_name, prompts, **self.kwargs)
            handler = self.handler_factory.create_handler(self.request_type, self.handler_type, self.model_name, self.config_path, self.pipe)
            return request.send_request(handler)
        conversation_history = []
        responses = []
        for prompt in prompts:
            conversation_history.append({"role": "user", "content": prompt})
            messages = self._format_messages(conversation_history)
            request = self.request_factory.create_request(
                self.request_type, self.model_name, messages, **self.kwargs
            )
            handler = self.handler_factory.create_handler(
                self.request_type, self.handler_type, self.model_name, self.config_path
            )
            response = request.send_request(handler)
            conversation_history.append({"role": "assistant", "content": response})
            responses.append(response)
        return responses

    async def _process_multiturn_async(self, prompts):
        conversation_history = []
        responses = []
        for prompt in prompts:
            conversation_history.append({"role": "user", "content": prompt})
            messages = self._format_messages(conversation_history)
            request = await self.request_factory.create_request_async(
                self.request_type, self.model_name, messages, **self.kwargs
            )
            handler = await self.handler_factory.create_handler_async(
                self.request_type, self.handler_type, self.model_name, self.config_path
            )
            response = await request.send_request_async(handler)
            conversation_history.append({"role": "assistant", "content": response})
            responses.append(response)
        return responses
    
    @retry_on_failure(max_retries=3, delay=1, backoff=1.1)
    def process(self, prompt, **kwargs):
        if isinstance(prompt, str):
            return self._process_single(prompt, **kwargs)
        elif isinstance(prompt, list):
            return self._process_multiturn(prompt)
        else:
            raise ValueError("Prompt must be a string or a list of strings.")

    @retry_on_failure_async(max_retries=1, delay=1, backoff=1.1)
    async def process_async(self, prompt, **kwargs):
        if isinstance(prompt, str):
            return await self._process_single_async(prompt, **kwargs)
        elif isinstance(prompt, list):
            return await self._process_multiturn_async(prompt)
        else:
            raise ValueError("Prompt must be a string or a list of strings.")


async def apply_function_concurrently(
    func: Callable[..., Dict[str, Any]],
    elements: List[Dict[str, Any]],
    max_concurrency: int
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency)
    results = [None] * len(elements)

    async def bound_function(index: int, element: Dict[str, Any]):
        async with semaphore:
            result = await func(**element)
            results[index] = result

    tasks = [bound_function(index, element) for index, element in enumerate(elements)]
    await tqdm_asyncio.gather(*tasks)
    return results