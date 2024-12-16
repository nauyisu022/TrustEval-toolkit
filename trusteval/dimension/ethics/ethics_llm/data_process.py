import pandas as pd
import json
import os
import asyncio
from tqdm import tqdm

async def load_data(filepath, sep='\t'):
    return pd.read_csv(filepath, sep=sep)

async def filter_data(df):
    return df[df['rot-bad'] == 0]

async def extract_data_to_json(df):
    results = []
    dataset_num = 1

    for index, row in df.iterrows():
        entry = {
            "action": row['action'],
            "judgement": row['action-moral-judgment'],
            "dataset": "Social-Chem-101 Dataset",
            "category": row['rot-categorization'],
            "dataset_num": dataset_num
        }
        results.append(entry)
        dataset_num += 1

    return json.dumps(results, indent=4)

async def process_social_chem_101(base_dir):
    input_dir = os.path.join(base_dir, 'ori_dataset')
    output_dir = os.path.join(base_dir, 'data')

    filepath = os.path.join(input_dir, '1. social-chem-101/social-chem-101.v1.0.tsv')
    data = await load_data(filepath)
    filtered_data = await filter_data(data)
    json_data = await extract_data_to_json(filtered_data)

    output_path = os.path.join(output_dir, '1. social-chem-101.json')
    with open(output_path, 'w') as f:
        f.write(json_data)

async def load_and_process_csv(filepath, ambiguity_level):
    df = pd.read_csv(filepath)
    df['dataset'] = 'MoralChoice'
    df['ambiguity'] = ambiguity_level
    df = df[['context', 'ambiguity', 'action1', 'action2', 'generation_rule', 'dataset']]
    return df

async def convert_to_json(df):
    df['dataset_num'] = range(1, len(df) + 1)
    return json.dumps(df.to_dict(orient='records'), indent=4)

async def process_moral_choice(base_dir):
    input_dir = os.path.join(base_dir, 'ori_dataset')
    output_dir = os.path.join(base_dir, 'data')

    high_amb_df = await load_and_process_csv(os.path.join(input_dir, '2. moralchoice/scenarios/moralchoice_high_ambiguity.csv'), 'high')
    low_amb_df = await load_and_process_csv(os.path.join(input_dir, '2. moralchoice/scenarios/moralchoice_low_ambiguity.csv'), 'low')
    combined_df = pd.concat([high_amb_df, low_amb_df], ignore_index=True)
    json_data = await convert_to_json(combined_df)

    output_path = os.path.join(output_dir, '2. moralchoice.json')
    with open(output_path, 'w') as f:
        f.write(json_data)

async def process_commonsense(df, is_ambig=False):
    if is_ambig:
        return [{"scenario": row['input'], "ambiguous": "yes"} for _, row in df.iterrows()]
    else:
        return [{"scenario": row['input'], "judgement": row['label']} for _, row in df.iterrows()]

async def process_deontology(df):
    return [{"scenario": row['scenario'], "excuse": row['excuse'], "judgement": row['label']} for _, row in df.iterrows()]

async def process_justice(df):
    return [{"scenario": row['scenario'], "judgement": row['label']} for _, row in df.iterrows()]

async def process_virtue(df):
    merged_data = {}
    for _, row in df.iterrows():
        scenario, option = row['scenario'].split(' [SEP] ')
        if scenario not in merged_data:
            merged_data[scenario] = {"scenario": scenario, "options": [], "judgement": None}
        merged_data[scenario]["options"].append(option)
        if row['label'] == 1:
            merged_data[scenario]["judgement"] = option
    result = []
    for scenario, data in merged_data.items():
        data["options"] = ", ".join(data["options"])
        result.append(data)
    return result

async def process_utilitarianism(df):
    return [{"scenario1": row[0], "scenario2": row[1], "judgement": "1"} for _, row in df.iterrows()]

async def process_csv_file(filepath, category):
    if category == "commonsense":
        if "cm_ambig.csv" in filepath:
            df = pd.read_csv(filepath, header=None, names=['input'])
            return await process_commonsense(df, is_ambig=True)
        else:
            df = pd.read_csv(filepath)
            return await process_commonsense(df)
    elif category == "deontology":
        df = pd.read_csv(filepath)
        return await process_deontology(df)
    elif category == "justice":
        df = pd.read_csv(filepath)
        return await process_justice(df)
    elif category == "virtue":
        df = pd.read_csv(filepath)
        return await process_virtue(df)
    elif category == "utilitarianism":
        df = pd.read_csv(filepath, header=None)
        return await process_utilitarianism(df)
    return []

async def process_ethics(base_dir):
    input_dir = os.path.join(base_dir, 'ori_dataset')
    output_dir = os.path.join(base_dir, 'data')

    base_dir_ethics = os.path.join(input_dir, '3. ethics/')
    all_data = []
    dataset_num = 1

    for root, dirs, files in os.walk(base_dir_ethics):
        category = os.path.basename(root)
        for file in files:
            if file.endswith('.csv'):
                filepath = os.path.join(root, file)
                processed_data = await process_csv_file(filepath, category)

                for entry in processed_data:
                    entry["category"] = category
                    entry["dataset"] = "Ethics Dataset"
                    entry["dataset_num"] = dataset_num
                    dataset_num += 1

                all_data.extend(processed_data)

    output_path = os.path.join(output_dir, '3. ethics.json')
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=4)

async def process_normbank(base_dir):
    input_dir = os.path.join(base_dir, 'ori_dataset')
    output_dir = os.path.join(base_dir, 'data')

    csv_path = os.path.join(input_dir, '5. NormBank/NormBank.csv')
    df = pd.read_csv(csv_path)

    processed_data = [
        {
            "scenario": row['setting'],
            "action": row['behavior'],
            "constraint": row['constraints'],
            "judgement": row['label'],
            "dataset": "NormBank",
            "dataset_num": idx + 1
        }
        for idx, (_, row) in enumerate(df.iterrows())
    ]

    output_path = os.path.join(output_dir, '5. NormBank.json')
    with open(output_path, "w") as json_file:
        json.dump(processed_data, json_file, indent=4)

async def process_moral_stories(base_dir):
    input_dir = os.path.join(base_dir, 'ori_dataset')
    output_dir = os.path.join(base_dir, 'data')

    input_jsonl_path = os.path.join(input_dir, '6. Moral Stories/moral_stories_full.jsonl')
    output_json_path = os.path.join(output_dir, '6. MoralStories.json')

    processed_data = []

    with open(input_jsonl_path, 'r') as jsonl_file:
        for idx, line in enumerate(jsonl_file):
            record = json.loads(line)
            processed_record = {
                "scenario": record["situation"],
                "intention": record["intention"],
                "moral_action": record["moral_action"],
                "immoral_action": record["immoral_action"],
                "dataset": "Moral Stories",
                "dataset_num": idx + 1
            }
            processed_data.append(processed_record)

    with open(output_json_path, "w") as json_file:
        json.dump(processed_data, json_file, indent=4)

async def process_culturebank(base_dir):
    input_dir = os.path.join(base_dir, 'ori_dataset')
    output_dir = os.path.join(base_dir, 'data')

    csv_file_path1 = os.path.join(input_dir, '7. CultureBank/culturebank_reddit.csv')
    csv_file_path2 = os.path.join(input_dir, '7. CultureBank/culturebank_tiktok.csv')

    df1 = pd.read_csv(csv_file_path1)
    df2 = pd.read_csv(csv_file_path2)

    df = pd.concat([df1, df2], ignore_index=True)
    df_filtered = df[['eval_persona', 'eval_question', 'cultural group', 'eval_whole_desc', 'topic']]

    results = []
    for idx, row in df_filtered.iterrows():
        result = {
            'eval_persona': row['eval_persona'],
            'eval_question': row['eval_question'],
            'cultural group': row['cultural group'],
            'Judgment basis': row['eval_whole_desc'],
            'dataset': 'CultureBank',
            'category': row['topic'],
            'dataset_num': idx + 1
        }
        results.append(result)

    output_path = os.path.join(output_dir, '7. CultureBank.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

async def run_processing(base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(base_dir, 'data')
    os.makedirs(output_dir, exist_ok=True)

    task_functions = [
        lambda: process_social_chem_101(base_dir),
        lambda: process_moral_choice(base_dir),
        lambda: process_ethics(base_dir),
        lambda: process_normbank(base_dir),
        lambda: process_moral_stories(base_dir),
        lambda: process_culturebank(base_dir)
    ]

    for task in tqdm(task_functions, desc='Processing datasets'):
        await task()

def main(base_dir=None):
    asyncio.run(run_processing(base_dir))

if __name__ == '__main__':
    main() 