import json
import os
import asyncio
import sys
from contextual_variator import ContextualVariator, apply_function_concurrently

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)


def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return None

def write_json(file_path, data):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"Error writing JSON file {file_path}: {e}")

async def _process_file_single(data_file, config):
    data = read_json(data_file)
    if data is None:
        return
    enhancer = ContextualVariator(config["transformation_method"])
    parameters = []

    for item in data:
        parameters.append({
            "sentence": item.get("prompt", ""),
            "current_format": config["question_format"],
            "answer": item.get("ground_truth", ""),
        })

    results = await apply_function_concurrently(enhancer.enhance_diversity, parameters, 5)
    for i, item in enumerate(data):
        try:
            item['original_format'] = config["question_format"]
            item["enhanced_prompt"] = results[i]["sentence"]
            if "answer" in results[i].keys():
                item["enhanced_ground_truth"] = results[i]["answer"]
            item['enhancement_method'] = results[i]["enhancement_method"]
            for key in results[i].keys():
                if key not in ['sentence', 'answer', 'enhancement_method']:
                    item[key] = results[i][key]
        except:
            print(f"Error processing item {i}")

    enhanced_file = data_file.replace(".json", "_enhanced.json")
    write_json(enhanced_file, data)
    print(f"Enhanced data saved to {enhanced_file}")

def _process_file_multiturn(data_file, config):
    data = read_json(data_file)
    if data is None:
        return
    
    enhancers = [ContextualVariator(method) for method in config["transformation_method"]]
    parameters = [[] for _ in range(len(enhancers))]
    for item in data:
        for j, prompt in enumerate(item['prompt']):
            parameters[j].append({
                "sentence": prompt,
                "current_format": config["question_format"],
                "answer": item.get("ground_truth", ""),
                "extra_instructions": item.get("extra_instructions", "")
            })
    
    results = []
    for j, enhancer in enumerate(enhancers):
        results.append(asyncio.get_event_loop().run_until_complete(apply_function_concurrently(enhancer.enhance_diversity, parameters[j], 5)))
    
    for i, item in enumerate(data):
        item['enhanced_prompt'] = []
        item['enhanced_ground_truth'] = []
        item['enhancement_method'] = []
        for j, prompt in enumerate(item['prompt']):
            try:
                item['enhanced_prompt'].append(results[j][i]["sentence"])
                if "answer" in results[j][i].keys():
                    item['enhanced_ground_truth'].append(results[j][i]["answer"])
                item['enhancement_method'].append(results[j][i]["enhancement_method"])
                for key in results[j][i].keys():
                    if key not in ['sentence', 'answer', 'enhancement_method']:
                        item[key] = results[j][i][key]
            except:
                print(f"Error processing item {i}")

    enhanced_file = data_file.replace(".json", "_enhanced.json")
    write_json(enhanced_file, data)
    print(f"Enhanced data saved to {enhanced_file}")

async def process_file(data_file, config):
    if "transformation_method" not in config:
        print(f"Transformation method not specified in {data_file}")
        return
    if isinstance(config["transformation_method"][0], list):
        _process_file_multiturn(data_file, config)
    else:
        await _process_file_single(data_file, config)


async def main(dataset_folder=None):
    config_file = os.path.join(dataset_folder, "file_config.json")

    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} does not exist")
        return

    config = read_json(config_file)
    if config is None:
        return

    for file_config in config:
        data_file = os.path.join(dataset_folder, file_config["file_name"])
        if os.path.exists(data_file):
            await process_file(data_file, file_config)
        else:
            print(f"Data file {data_file} does not exist")
            
def contextual_variator_cli(dataset_folder=None):
    asyncio.get_event_loop().run_until_complete(main(dataset_folder))