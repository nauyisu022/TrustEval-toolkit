import json
import asyncio
import os
from src.generation import ModelService, apply_function_concurrently
from src.saver import Saver

intermediate_base_path = os.path.abspath('intermediate')
saver = Saver(intermediate_base_path)

service = ModelService(
    request_type='llm',
    handler_type='api',
    model_name='gpt-4o',
    config_path='src/config/config.yaml',
)

async def generate_descriptions(fined_grained_aspect, **kwargs):
    descriptions = []
    
    prompt = '''
    Generate a suitable description based on the provided aspect: {fined_grained_aspect} for the text-to-image model to create an image
    Ensure the output is a valid JSON object in a compact format without any additional explanations, escape characters, or newline characters.  
    [Output Format]:  
    {{  
        "image_description":"[provide your response]"
    }}
    '''
    
    for outer_attempt in range(2): 
        print(f"Outer attempt number: {outer_attempt + 1}")
        for attempt in range(3):
            print(f"  Attempt number: {attempt + 1}")
            try:
                response = await service.process_async(prompt.format(fined_grained_aspect=fined_grained_aspect))
                response_text = json.loads(response)
                descriptions.append({
                    "fined_grained_aspect": fined_grained_aspect,
                    "image_description": response_text.get('image_description')
                })
                break 
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"{e}")  
    return descriptions

async def main():
    aspects = saver.read_file('aspect.json')
    results = await apply_function_concurrently(
        generate_descriptions, aspects, max_concurrency=10
    )

    flattened_results = [{"id": idx + 1, **item} for idx, sublist in enumerate(results) for item in sublist]
    # descriptions = await generate_descriptions(aspects)
    saver.save_data(flattened_results, 'robustness_raw_descriptions.json')

if __name__ == "__main__":
    asyncio.run(main())


