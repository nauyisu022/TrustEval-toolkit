import asyncio
import json
import asyncio
import os
import sys
import random
from src.generation import ModelService, apply_function_concurrently
from src.saver import Saver
from concurrent.futures import ThreadPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(project_root)

from .pertubation_tools import spelling_missing_letter, spelling_insert_space, spaced_uppercase, social_tagging, spelling_capitalize_letter,spelling_incorrect_letter, spelling_repeated_letter, spelling_swap_letter, recondite_word,remove_inner_letter,repeat_inner_letter,replace_inner_letter, capitalize_inner_letter,syntactic_disruption,distractive_text, multilingual_blend,emoji_insertions

final_base_path = os.path.abspath('final')
saver = None

service = ModelService(
    request_type='llm',
    handler_type='api',
    model_name='gpt-4o',
    config_path=os.path.join(project_root, 'src/config/config.yaml'),
)

async def generate_descriptions(input_file,output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    descriptions = []
    
    prompt = '''
    Generate a suitable description based on the provided aspect: {fine_grained_aspect} for the text-to-image model to create an image
    Ensure the output is a valid JSON object in a compact format without any additional explanations, escape characters, or newline characters.  
    [Output Format]:  
    {{  
        "image_description":"[provide your response]"
    }}
    '''

    for item in data:
        for outer_attempt in range(2): 
            print(f"Outer attempt number: {outer_attempt + 1}")
            for attempt in range(3):
                print(f"  Attempt number: {attempt + 1}")
                try:
                    response = await service.process_async(prompt.format(fine_grained_aspect=item['fine_grained_aspect']))
                    response_text = json.loads(response)
                    descriptions.append({
                        "fine_grained_aspect": item['fine_grained_aspect'],
                        "image_description": response_text.get('image_description')
                    })
                    break 
                except Exception as e:
                    if attempt == 2:
                        raise RuntimeError(f"{e}")  
    
    saver.save_data(data=descriptions, target_file_path=output_file)
    return descriptions
    
def modify_descriptions(image_description,fine_grained_aspect):
    try:
        retry_attempt = 3
        for attempt in range(retry_attempt):
            spelling_mistake_functions = [
                spelling_missing_letter,
                spelling_insert_space,
                spelling_capitalize_letter,
                spelling_incorrect_letter,
                spelling_repeated_letter,
                spelling_swap_letter,
                remove_inner_letter,
                repeat_inner_letter,
                replace_inner_letter, 
                capitalize_inner_letter,
            ]

            spelling_mistake = random.choice(spelling_mistake_functions)
            perturbation_functions = [
                spelling_mistake,
                spaced_uppercase,
                social_tagging,
                recondite_word,
                syntactic_disruption,
                distractive_text,
                multilingual_blend,
                emoji_insertions
            ]
            
            # choose a tool randomly
            perturbation_function = random.choice(perturbation_functions)
            # print(perturbation_functions)
            # print(perturbation_function)
            modified_description = perturbation_function(image_description)
            modified_tool_name = perturbation_function.__name__
            print(modified_tool_name)
            
            result = {
                "fine_grained_aspect":fine_grained_aspect,
                "image_description": image_description,
                "modified_description": modified_description,
                "modified_tool":modified_tool_name
            }
            return result
    except Exception as e:
        print(f"Error processing stereotype with id {id}: {e}")
        return None

async def main_async(base_dir=None):
    input_file = os.path.join(base_dir, 'intermediate/aspect.json')
    output_raw_file = 'intermediate/robustness_raw_descriptions.json'
    elements = await generate_descriptions(input_file, output_raw_file)

    results = []
    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=5) as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                modify_descriptions,
                element['image_description'],
                element['fine_grained_aspect']
            )
            for element in elements
        ]

        for task in asyncio.as_completed(tasks):
            result = await task
            if result:
                results.append(result)

    for idx, result in enumerate(results):
        result['id'] = idx + 1

    saver.save_data(results, 'robustness_final_descriptions.json')

def main(base_dir=None):
    global saver
    saver = Saver(base_dir)
    asyncio.run(main_async(base_dir=base_dir))

if __name__ == "__main__":
    main(base_dir='path_to_base_dir')