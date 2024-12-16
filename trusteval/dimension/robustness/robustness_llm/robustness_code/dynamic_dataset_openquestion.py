import csv
import random
import os
from datasets import load_dataset
import json
import asyncio
from tqdm.asyncio import tqdm_asyncio
from concurrent.futures import ThreadPoolExecutor
from .adv_generate import (
    spelling_missing_letter, spelling_capitalize_letter, spelling_incorrect_letter,
    spelling_insert_space, spelling_repeated_letter, spelling_swap_letter,
    emoji_insertions, social_tagging, spaced_uppercase, multilingual_blend,
    distractive_text, syntactic_disruption, recondite_word, EmojiSearch
)
from .add_template import generate_prompt

def cnn_dailymail(datapath, num):
    dataset = load_dataset(datapath, '3.0.0', split='train')
    sampled_indices = random.sample(range(len(dataset)), min(num, len(dataset)))
    sampled_data = [dataset[i] for i in sampled_indices]
    data = []
    for item in sampled_data:
        article = item['article']
        highlights = item['highlights']
        data.append({'dataset': 'cnn_dailymail', 'article': article, 'highlights': highlights})
    return data

def eli5(datapath, num):
    dataset = load_dataset(datapath, split='train')
    sampled_indices = random.sample(range(len(dataset)), min(num, len(dataset)))
    sampled_data = [dataset[i] for i in sampled_indices]
    data = []
    for item in sampled_data:
        question = item['question']
        answer = item['answer']
        data.append({'dataset': 'eli5', 'question': question, 'answer': answer})
    return data

adv_functions = [
    ('spelling_missing_letter', {'if_keybert': True}),
    ('spelling_incorrect_letter', {'if_keybert': True}),
    ('spelling_repeated_letter', {'if_keybert': True}),
    ('spelling_capitalize_letter', {'if_keybert': True}),
    ('spelling_insert_space', {'if_keybert': True}),
    ('spelling_swap_letter', {'if_keybert': True}),
    ('emoji_insertions', {'if_keybert': True}),
    ('social_tagging', {}),
    ('spaced_uppercase', {'if_keybert': True}),
    ('multilingual_blend', {'if_keybert': True}),
    ('distractive_text', {}),
    ('syntactic_disruption', {}),
    ('recondite_word', {})
]

def sync_wrapper(func, *args, **kwargs):
    def wrapped_func(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapped_func

async def apply_adv(data, adv_function, params, method_label):
    loop = asyncio.get_event_loop()

    for item in tqdm_asyncio(data, desc=f"Applying {adv_function.__name__}"):
        field_name = None
        if item['dataset'] == 'eli5':
            field_name = 'question'
        elif item['dataset'] == 'cnn_dailymail':
            field_name = 'article'

        if field_name:
            original_data = {f'original_{key}': value for key, value in item.items() if key not in ['label', 'dataset']}

            if asyncio.iscoroutinefunction(adv_function):
                item[field_name] = await adv_function(item[field_name], **params)
            else:
                wrapped_function = sync_wrapper(adv_function, **params)
                item[field_name] = await loop.run_in_executor(
                    None, wrapped_function, item[field_name])

            item.update(original_data)
            item['method'] = method_label
            item['enhanced_prompt'] = generate_prompt(item)

async def process_dataset(adv_function_name, params, all_data):
    adv_function = globals().get(adv_function_name)
    if not adv_function:
        print(f"Function {adv_function_name} is not defined.")
        return []

    data_adv = [dict(item) for item in all_data]
    method_label = f"{adv_function_name}-open"
    await apply_adv(data_adv, adv_function, params, method_label)

    return data_adv

emoji_search = EmojiSearch()

async def process_open_question():
    num = 400
    datasets_paths = {
        'eli5': "sentence-transformers/eli5",
        'cnn_dailymail': 'cnn_dailymail'
    }

    all_data = []

    for dataset_name, path in datasets_paths.items():
        if dataset_name == 'eli5':
            sampled_data = eli5(path, num)
        elif dataset_name == 'cnn_dailymail':
            sampled_data = cnn_dailymail(path, num)
        else:
            print(f"Unknown dataset: {dataset_name}")
            continue

        for data in sampled_data:
            data['original_prompt'] = generate_prompt(data)
            data['dataset_type'] = 'open_ended'

        all_data.extend(sampled_data)

    all_adv_data = []
    tasks = [process_dataset(adv_function_name, params, all_data) for adv_function_name, params in adv_functions]

    results = await tqdm_asyncio.gather(*tasks)
    for result in results:
        all_adv_data.extend(result)

    return all_adv_data