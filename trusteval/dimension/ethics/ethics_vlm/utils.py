from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import requests
import os,sys,yaml
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import os
from PIL import Image


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../src"))
config_file_path = os.path.join(project_root, "config", "config.yaml")
from generation.model_service import ModelService

with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

async def call_gpt4o_api(prompt):
    service = ModelService(
        request_type="llm",
        handler_type="api",
        model_name="gpt-4o",
        config_path=config_file_path,
        temperature=0.7,
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

def generate_image(prompt):
    """
    Helper function to generate a single image and save it to an output path.
    Includes error handling to prevent failure from stopping the whole process.

    Args:
        prompt (str): The prompt string for image generation.
        output_path (str): The absolute file path to save the generated image.
        service (ModelService): An instance of ModelService to handle the generation.
    """
    service = ModelService(
        request_type='t2i',
        handler_type='api',
        model_name="dalle3",
        config_path=config_file_path,
    )

    try:
        result = service.process(prompt)
        if result is None:
            raise ValueError(f"No image generated for prompt: {prompt}")
        return result
    except Exception as e:
        print(f"Error generating image for prompt: '{prompt}' - {e}")



def generate_and_save_image(prompt, img_id, save_path):
    try:
        image = generate_image(prompt=prompt)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_name = f"{img_id}.png"
        full_path = os.path.join(save_path, file_name)

        image.save(full_path, format='PNG')

        print(f"saved in: {full_path}")
        return True

    except Exception as e:
        print(f"save error: {str(e)}")
        return False