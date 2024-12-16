import os
import json
import sys
from .prompts import *
from .util import *
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)
from src.generation import ModelService

base_path = None
def get_relative_path(file_path):
    return os.path.join(base_path, file_path)

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
def description_prompt_split(output):
    description = output.split('||')[1]
    prompt = output.split('[')[1].split(']')[0]
    return description, prompt

def Visual_Roleplay(harmful_query, path_dir, idx):
    output = text_service.process(Visual_Roleplay_Character_Description_Gen_Prompt + harmful_query)
    print(output)
    char_des, t2i_prompt = description_prompt_split(output)
    diffusion_image = image_service.process(t2i_prompt)
    formated_text, line_num = typo_format_text(char_des, font_size=60)
    typo_description = typo_draw_img(formated_text, line_num, font_size=60)
    formated_text, line_num = typo_format_text(harmful_query, font_size=60)
    typo_query = typo_draw_img(formated_text, line_num, font_size=60)
    width1, height1 = typo_description.size
    width2, height2 = diffusion_image.size
    width3, height3 = typo_query.size
    img_path = os.path.join(path_dir, f'VisualRoleplay_{idx}.png')
    result_width = max(width1, width2, width3)
    result_height = height1 + height2 + height3
    result = Image.new('RGB', (result_width, result_height))

    result.paste(typo_description, (0, 0))
    result.paste(diffusion_image, (0, height1))
    result.paste(typo_query, (0, height1 + height2))
    result.save(get_relative_path(img_path))
    return img_path

def main(base_dir=None, initialize=False):
    global base_path
    base_path = base_dir
    data = json.load(open(get_relative_path('generated_jailbreak_results.json'), 'r', encoding='utf-8'))
    ori_harmful_queries = []
    image_path_dir = get_relative_path('Images/')
    os.makedirs(image_path_dir, exist_ok=True)
    if initialize:
        for root, dirs, files in os.walk(image_path_dir):
            for file in files:
                if file.startswith('VisualRoleplay'):
                    os.remove(os.path.join(root, file))

    harmful_dicts = []
    for i in range(len(data)):
        if 'unsafe' in data[i]['guard_label']:
            ori_harmful_queries.append(data[i]['transformed_query'])
            harmful_dicts.append(data[i])
    if initialize:
        VisualRoleplay_json = []
    else:
        VisualRoleplay_json = json.load(open(get_relative_path('VisualRoleplay.json'), 'r', encoding='utf-8'))
    for i in range(len(ori_harmful_queries)):
        if not initialize:
            if os.path.exists(get_relative_path(os.path.join(image_path_dir, f'VisualRoleplay_{i}.png'))):
                continue
        print(i)
        try:
            img_path = Visual_Roleplay(ori_harmful_queries[i], image_path_dir, i)
            harmful_dict = {
                'harmful_query': ori_harmful_queries[i],
                'prompt': Visual_Roleplay_text_input,
                'image_urls': img_path
            }
            # update the harmful_dicts[i] with harmful_dict
            harmful_dicts[i].update(harmful_dict)
            VisualRoleplay_json.append(harmful_dicts[i])
            json.dump(VisualRoleplay_json, open(get_relative_path('final/VisualRoleplay.json'), 'w', encoding='utf-8'), indent=4)
        except Exception as e:
            print(f"Error processing query {i}: {e}")
            pass