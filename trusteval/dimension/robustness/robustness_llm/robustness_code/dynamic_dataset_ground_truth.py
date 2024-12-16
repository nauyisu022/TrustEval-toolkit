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

def sst2(datapath, num):
    data = []
    with open(datapath, 'r', encoding='utf-8') as file:
        reader = list(csv.DictReader(file, delimiter='\t'))
        filtered_rows = [row for row in reader if len(row['sentence'].split()) >= 20]
        sampled_rows = random.sample(filtered_rows, num)
        for row in sampled_rows:
            sentence = row['sentence'].strip()
            label = 'positive' if row['label'] == '1' else 'negative'
            data.append({'dataset': 'sst2', 'sentence': sentence, 'label': label})
    return data

def qqp(datapath, num):
    data = []
    with open(datapath, 'r', encoding='utf-8') as file:
        reader = list(csv.DictReader(file, delimiter='\t'))
        sampled_rows = random.sample(reader, num)
        for row in sampled_rows:
            question1 = row['question1'].strip()
            question2 = row['question2'].strip()
            label = 'duplicate' if row['is_duplicate'] == '1' else 'not duplicate'
            data.append({'dataset': 'qqp', 'question1': question1, 'question2': question2, 'label': label})
    return data

def load_partial_data_mnli(datapath, load_size=10000):
    partial_rows = []
    valid_labels = {'entailment', 'contradiction', 'neutral'}
    with open(datapath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for i, row in enumerate(reader):
            if random.random() > 0.8:
                continue
            if row['gold_label'] in valid_labels:
                partial_rows.append(row)
            if len(partial_rows) >= load_size:
                break
    return partial_rows

def mnli(datapath, num):
    data = []
    partial_rows = load_partial_data_mnli(datapath, load_size=10000)
    sampled_rows = random.sample(partial_rows, num)
    for row in sampled_rows:
        sentence1 = row['sentence1'].strip()
        sentence2 = row['sentence2'].strip()
        label = row['gold_label']
        data.append({'dataset': 'mnli', 'sentence1': sentence1, 'sentence2': sentence2, 'label': label})
    return data

def qnli(datapath, num):
    data = []
    with open(datapath, 'r', encoding='utf-8') as file:
        reader = list(csv.DictReader(file, delimiter='\t'))
        sampled_rows = random.sample(reader, num)
        for row in sampled_rows:
            question = row['question'].strip()
            sentence = row['sentence'].strip()
            label = row['label']
            if label == 'entailment':
                label = 'entailment'
            elif label == 'not_entailment':
                label = 'not entailment'
            data.append({'dataset': 'qnli', 'question': question, 'sentence': sentence, 'label': label})
    return data

def imdb(datapath, num):
    dataset = load_dataset("imdb", split='train')
    sampled_indices = random.sample(range(len(dataset)), min(num, len(dataset)))
    sampled_data = [dataset[i] for i in sampled_indices]
    data = []
    for item in sampled_data:
        sentence = item['text']
        label = 'positive' if item['label'] == 1 else 'negative'
        data.append({'dataset': 'imdb', 'sentence': sentence, 'label': label})
    return data

def race(datapath, num):
    dataset = load_dataset("race", "all", split='train')

    def format_options(options):
        labels = ['A', 'B', 'C', 'D']
        formatted_options = ' '.join(f"{label}. {option}" for label, option in zip(labels, options))
        return formatted_options

    sampled_indices = random.sample(range(len(dataset)), min(num, len(dataset)))
    sampled_data = [dataset[i] for i in sampled_indices]
    data = []
    for item in sampled_data:
        article = item['article']
        question = item['question']
        options = item['options']
        label = item['answer']
        formatted_options = format_options(options)
        data.append({'dataset': 'race', 'article': article, 'question': question, 'options': formatted_options, 'label': label})
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
        if item['dataset'] == 'sst2':
            field_name = 'sentence'
        elif item['dataset'] == 'qqp':
            field_name = 'question2'
        elif item['dataset'] == 'mnli':
            field_name = 'sentence2'
        elif item['dataset'] == 'qnli':
            field_name = 'sentence'
        elif item['dataset'] == 'imdb':
            field_name = 'sentence'
        elif item['dataset'] == 'race':
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
    method_label = f"{adv_function_name}-truth"
    await apply_adv(data_adv, adv_function, params, method_label)

    return data_adv

emoji_search = EmojiSearch()
project_root = os.environ.get('PROJECT_ROOT', '.')

async def process_ground_truth():
    num = 400
    dataset_dir = os.environ['DATASET_DIR']  

    datasets_paths = {
        'sst2': os.path.join(dataset_dir, 'data/SST-2/train.tsv'),
        'qqp' : os.path.join(dataset_dir, 'data/QQP/train.tsv'),
        'mnli': os.path.join(dataset_dir, 'data/MNLI/train.tsv'),
        'qnli': os.path.join(dataset_dir, 'data/QNLI/train.tsv'),
        'imdb': os.path.join(dataset_dir, 'data/IMDB'),  
        'race': os.path.join(dataset_dir, 'data/RACE')
    }

    all_data = []

    for dataset_name, path in datasets_paths.items():
        if dataset_name == 'sst2':
            sampled_data = sst2(path, num)
        elif dataset_name == 'qqp':
            sampled_data = qqp(path, num)
        elif dataset_name == 'mnli':
            sampled_data = mnli(path, num)
        elif dataset_name == 'qnli':
            sampled_data = qnli(path, num)
        elif dataset_name == 'imdb':
            sampled_data = imdb(path, num)
        elif dataset_name == 'race':
            sampled_data = race(path, num)
        else:
            print(f"Unknown dataset: {dataset_name}")
            continue

        for data in sampled_data:
            data['original_prompt'] = generate_prompt(data)
            data['dataset_type'] = 'ground_truth'

        all_data.extend(sampled_data)

    all_adv_data = []
    tasks = [process_dataset(adv_function_name, params, all_data) for adv_function_name, params in adv_functions]

    results = await tqdm_asyncio.gather(*tasks)
    for result in results:
        all_adv_data.extend(result)

    return all_adv_data