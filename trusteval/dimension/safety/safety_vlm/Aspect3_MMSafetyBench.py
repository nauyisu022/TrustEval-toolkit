import json
import sys
import os
from PIL import Image
from enum import IntEnum, unique
from .prompts import MMSafetyBench_sys_prompt
from .util import typo_format_text, typo_draw_img

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)
from src.generation import ModelService

text_service = ModelService(
    request_type='llm',
    handler_type='api',
    model_name='gpt-4o',
    config_path='src/config/config.yaml',
)

image_service = ModelService(
    request_type='t2i',
    handler_type='api',
    model_name='flux_schnell',
    config_path='src/config/config.yaml',
)

base_path = None
def get_relative_path(file_path):
    return os.path.join(base_path, file_path)

def get_question_for_sd(origin_ques):
    assert origin_ques.startswith("The bottom of the image"), "not start with 'The bottom of the image'"
    sd_ques = "The image" + origin_ques[23:]
    return sd_ques

def extract_keywords(harmful_query):
    try:
        output = text_service.process('[System]\n' + MMSafetyBench_sys_prompt + '[Question]\n' + harmful_query)
        # output = text_completion_open_source_models(MMSafetyBench_sys_prompt + '\n\n[Question]\n' + harmful_query, model="mistral-8x7b", max_tokens=256)
        print(output)
        if output is None or 'sorry' in output.lower():
            raise Exception("Failed to extract keywords")
        changed_question = output.split('[Changed Question]\n')[1].split('\n\n[Key Phrase]')[0]
        key_phrase = output.split('[Key Phrase]\n')[1].split('\n\n[Phrase Type]')[0]
        phrase_type = output.split('[Phrase Type]\n')[1].split('\n\n[Rephrased Question]')[0]
        rephrased_question = output.split('[Rephrased Question]\n')[1]
        rephrased_question_SD = get_question_for_sd(rephrased_question)
        return harmful_query, changed_question, key_phrase, phrase_type, rephrased_question, rephrased_question_SD
    except:
        return 1

def MMSafetyBench(harmful_query, image_path, idx):
    output = extract_keywords(harmful_query)
    if output != 1:
        harmful_query, changed_question, key_phrase, phrase_type, rephrased_question, rephrased_question_SD = output
        formated_text, line_num = typo_format_text(key_phrase, font_size=90)
        img_path = os.path.join(image_path, f'MMSafetyBench_{idx}.png')
        diffusion_image = image_service.process(key_phrase)
        typo_img = typo_draw_img(formated_text, line_num, font_size=90)
        width1, height1 = diffusion_image.size
        width2, height2 = typo_img.size

        result_width = max(width1, width2)
        result_height = height1 + height2
        result = Image.new('RGB', (result_width, result_height))

        result.paste(diffusion_image, (0, 0))
        result.paste(typo_img, (0, height1))
        result.save(get_relative_path(img_path))
        return harmful_query, rephrased_question
    else:
        return 1

def main(base_dir=None, initialize=False):
    global base_path
    base_path = base_dir
    data = json.load(open(get_relative_path('generated_jailbreak_results.json'), 'r', encoding='utf-8'))
    ori_harmful_queries = []
    image_path = get_relative_path('Images/')
    os.makedirs(image_path, exist_ok=True)
    if initialize:
        for root, dirs, files in os.walk(image_path):
            for file in files:
                if file.startswith('MMSafetyBench'):
                    os.remove(os.path.join(root, file))

    harmful_dicts = []
    for i in range(len(data)):
        if 'unsafe' in data[i]['guard_label']:
            ori_harmful_queries.append(data[i]['transformed_query'])
            harmful_dicts.append(data[i])
    if initialize:
        MMSafetyBench_json = []
    else:
        MMSafetyBench_json = json.load(open(get_relative_path('MMSafetyBench.json'), 'r', encoding='utf-8'))
    for i in range(len(ori_harmful_queries)):
        # if ./image/MMSafetyBench_i.png exists, skip this iteration
        if not initialize:
            if os.path.exists(os.path.join(image_path, f'MMSafetyBench_{i}.png')):
                continue
        print(i)
        output = MMSafetyBench(ori_harmful_queries[i], image_path, i)
        if output != 1:
            harmful_query, rephrased_question = output

            harmful_dict = {
                'harmful_query': harmful_query,
                'prompt': rephrased_question,
                'image_urls': os.path.join(image_path, f'MMSafetyBench_{i}.png')
            }
            # update the harmful_dicts[i] with harmful_dict
            harmful_dicts[i].update(harmful_dict)
            MMSafetyBench_json.append(harmful_dicts[i])
            json.dump(MMSafetyBench_json, open(get_relative_path('final/MMSafetyBench.json'), 'w', encoding='utf-8'), indent=4)
        else:
            print(1)

