import sys
import os
import asyncio
import json
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from dimension.robustness.robustness_llm.robustness_code import process_ground_truth, process_open_question

def remove_duplicate_originals(data):
    for item in data:
        keys_to_remove = [key for key in item if key.startswith('original_original_')]
        for key in keys_to_remove:
            del item[key]

    for item in data:
        keys_to_remove = [key for key in item if key.startswith('original_dataset_type')]
        for key in keys_to_remove:
            del item[key]

async def pipeline(base_dir=None):
    if base_dir:
        dataset_dir = base_dir
    else:
        dataset_dir = os.path.join(root_dir,'section' ,'robustness','robustness_llm', 'dataset')

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"The dataset directory does not exist: {dataset_dir}")

    os.environ['DATASET_DIR'] = dataset_dir
    os.environ['PROJECT_ROOT'] = root_dir  
    ground_truth_data = await process_ground_truth()
    open_question_data = await process_open_question()
    #combined_data = ground_truth_data + open_question_data

    file_name1 = os.path.join(dataset_dir, 'open_ended_data.json')
    file_name2 = os.path.join(dataset_dir, 'ground_truth_data.json')
    remove_duplicate_originals(ground_truth_data)
    remove_duplicate_originals(open_question_data)
    with open(file_name1, 'w', encoding='utf-8') as f:
        json.dump(open_question_data, f, ensure_ascii=False, indent=4)

    with open(file_name2, 'w', encoding='utf-8') as f:
        json.dump(ground_truth_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    asyncio.run(pipeline())