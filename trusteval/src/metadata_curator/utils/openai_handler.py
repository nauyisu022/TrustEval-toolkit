import os
from openai import AsyncOpenAI, AsyncAzureOpenAI
import sys
import yaml
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, parent_dir)

config_file_path = os.path.join(parent_dir, "config", "config.yaml")
diversity_path = os.path.join(parent_dir, "src", "config.yaml")

with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

sys.path.append(parent_dir)
from generation.model_service import ModelService
sys.path.append(current_dir)

async def call_openai_api(model_name,prompt,temperature):
    service = ModelService(
        request_type="llm",
        handler_type="api",
        model_name=model_name,
        config_path=config_file_path,
        temperature=temperature,
        max_tokens=2048
    )

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = await service.process_async(prompt=prompt)
            if response is not None:
                return response
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_attempts} failed with error: {e}")
            if attempt == max_attempts - 1:
                return ""

    return ""

def get_openai_response(model="gpt-4o-mini", need_azure=False, temperature=0.5):
    async def get_response(prompt):
        if need_azure:
            return await call_openai_api(model=model, prompt=prompt, temperature=temperature)
        else:
            return await call_openai_api(model=model, prompt=prompt, temperature=temperature)
    
    return get_response

async def get_openai_text_response(model="gpt-4o-mini", prompt="", temperature=0.5, system_prompt=None):
    return await call_openai_api(model,prompt,temperature)

async def get_azure_openai_text_response(model="gpt-4o-mini", prompt="", temperature=0.5, system_prompt=None):
    return await call_openai_api(model,prompt,temperature)