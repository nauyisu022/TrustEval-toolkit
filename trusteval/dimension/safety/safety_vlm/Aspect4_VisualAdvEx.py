import json
import os
import sys

base_path = None
def get_relative_path(file_path):
    return os.path.join(base_path, file_path)

def main(base_dir=None):
    global base_path
    base_path = base_dir
    data = json.load(open(get_relative_path('generated_jailbreak_results.json'), 'r', encoding='utf-8'))
    harmful_dicts = []
    for i in range(len(data)):
        if 'unsafe' in data[i]['guard_label']:
            harmful_dicts.append(data[i])

    for i in range(len(harmful_dicts)):
        harmful_dicts[i]['image_urls'] = 'Images/VisualAdversarialExamples.bmp'
        harmful_dicts[i]['prompt'] = harmful_dicts[i]['transformed_query']

    json.dump(harmful_dicts, open(get_relative_path('final/VisualAdversarialExamples.json'), 'w', encoding='utf-8'), indent=4)