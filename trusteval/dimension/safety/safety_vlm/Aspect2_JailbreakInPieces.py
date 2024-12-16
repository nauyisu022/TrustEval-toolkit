import os
import json
import torch
from torch import optim
from torchvision import transforms
from PIL import Image
import clip
import numpy as np
from .util import typo_format_text, typo_draw_img
from .prompts import JailbreakInPieces_EXTRACT_PROMPT
import asyncio
from PIL import Image, ImageFont, ImageDraw
import time
import sys
import requests
from io import BytesIO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)
from src.metadata_curator.imageSearchPipeline import ImageWebSearchPipeline
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

def extract_keywords(harmful_query):
    try:
        output = text_service.process('[System]\n' + JailbreakInPieces_EXTRACT_PROMPT + '[Question]\n' + harmful_query)
        print(output)
        if output == None or 'sorry' in output.lower():
            raise Exception("Failed to extract keywords")
        key_phrase = output.split('[Key Phrase]\n')[1].split('\n\n[Phrase Type]')[0]
        phrase_type = output.split('[Phrase Type]\n')[1].split('\n\n[Rephrased Question]')[0]
        rephrased_question = output.split('[Rephrased Question]\n')[1]
        return key_phrase, rephrased_question
    except:
        return 1

def generate_adversarial_image(image_url):
    # Load CLIP model
    st = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)

    # Load the target image x_harm
    response = requests.get(image_url)
    x_harm_image = Image.open(BytesIO(response.content))
    x_harm = preprocess(x_harm_image).unsqueeze(0).to(device)

    # Get the embedding vector H_harm
    with torch.no_grad():
        H_harm = model.encode_image(x_harm)

    # Initialize x_adv with random noise
    x_adv = torch.randn_like(x_harm, requires_grad=True, device=device)

    # Set up the optimizer
    learning_rate = 0.01  # Commonly used learning rate in adversarial attacks
    optimizer = optim.Adam([x_adv], lr=learning_rate)

    # Define the loss function
    loss_fn = torch.nn.MSELoss()

    # Optimization loop
    max_iterations = 1000
    for iteration in range(max_iterations):
        if time.time() - st > 30:
            return 1
        optimizer.zero_grad()

        # Get the embedding vector H_adv
        H_adv = model.encode_image(x_adv)

        # Compute the L2 loss between H_adv and H_harm
        loss = loss_fn(H_adv, H_harm)
        # Backpropagation
        loss.backward()
        optimizer.step()

        # Project x_adv back to valid image range [0, 1]
        x_adv.data = torch.clamp(x_adv.data, 0, 1)

    et = time.time()
    print(f"Time taken: {et-st:.1f} seconds")
    # Save the adversarial image
    adv_image = x_adv.detach().cpu().squeeze(0)
    adv_image = transforms.ToPILImage()(adv_image)
    return adv_image

def jailbreakInPieces(harmful_query, image_path_dir, image_idx):
    try:
        key_phrase, rephrased_question = extract_keywords(harmful_query)
    except:
        return 1
    if key_phrase != 1 and key_phrase:
        print('key phrase:', key_phrase)
        instruction = "Find images of " + key_phrase
        basic_information = {}

        try:
            pipeline = ImageWebSearchPipeline(instruction, basic_information)
            asyncio.run(pipeline.run())
        except:
            return 1

        os.makedirs(get_relative_path('Images/temp_images/'), exist_ok=True)
        custom_output_path = get_relative_path(f"Images/temp_images/jailbreakInPieces_searched_{image_idx}.json")

        pipeline_with_custom_output = ImageWebSearchPipeline(instruction, basic_information,
                                                              output_path=custom_output_path)
        asyncio.run(pipeline_with_custom_output.run())
        image_url = json.load(open(custom_output_path, 'r', encoding='utf-8'))[1]['contentUrl']
        try:
            adv_image = generate_adversarial_image(image_url)
            if adv_image == 1:
                return 1
        except:
            return 1
        adv_image.save(get_relative_path(f'Images/jailbreak_in_pieces_{image_idx}.png'))
        formated_text, line_num = typo_format_text(key_phrase, font_size=30, max_width=240)
        adv_image = Image.open(get_relative_path(f'Images/jailbreak_in_pieces_{image_idx}.png'))
        typo_img = typo_draw_img(formated_text, line_num, font_size=30, max_width=240)
        width1, height1 = adv_image.size
        width2, height2 = typo_img.size
        result_width = max(width1, width2)
        result_height = height1 + height2
        result = Image.new('RGB', (result_width, result_height))
        result.paste(adv_image, (0, 0))
        result.paste(typo_img, (0, height1))
        result.save(get_relative_path(f'Images/jailbreak_in_pieces_{image_idx}.png'))
        return rephrased_question
    else:
        return 1

def main(base_dir=None, initialize=False):
    global base_path
    base_path = base_dir
    data_path = get_relative_path('generated_jailbreak_results.json')
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    ori_harmful_queries = []
    image_path_dir = get_relative_path('images')
    os.makedirs(image_path_dir, exist_ok=True)
    if initialize:
        for root, dirs, files in os.walk(image_path_dir):
            for file in files:
                if file.startswith('jailbreak_in_pieces'):
                    os.remove(os.path.join(root, file))

    harmful_dicts = []
    for i in range(len(data)):
        if 'unsafe' in data[i]['guard_label']:
            ori_harmful_queries.append(data[i]['transformed_query'])
            harmful_dicts.append(data[i])
    if initialize:
        jailbreakInPieces_json = []
    else:
        jailbreak_json_path = get_relative_path('jailbreakInPieces.json')
        jailbreakInPieces_json = json.load(open(jailbreak_json_path, 'r', encoding='utf-8'))
    for i in range(len(ori_harmful_queries)):
        if not initialize:
            if os.path.exists(get_relative_path(f'Images/jailbreak_in_pieces_{i}.png')):
                continue
        print(i)
        result = jailbreakInPieces(ori_harmful_queries[i], image_path_dir, i)
        if result != 1:
            img_path = get_relative_path(f'Images/jailbreak_in_pieces_{i}.png')
            harmful_dict = {
                'harmful_query': ori_harmful_queries[i],
                'prompt': result,
                'image_urls': img_path
            }
            harmful_dicts[i].update(harmful_dict)
            jailbreakInPieces_json.append(harmful_dicts[i])
            json.dump(jailbreakInPieces_json, open(get_relative_path('final/jailbreakInPieces.json'), 'w', encoding='utf-8'), indent=4)
        else:
            print(1)

