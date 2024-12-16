import csv
import random
import itertools
import os
from datasets import load_dataset
import json
from PIL import Image
from text_adv_generate import spelling_missing_letter, spelling_capitalize_letter, spelling_incorrect_letter, spelling_insert_space, spelling_repeated_letter, spelling_swap_letter, emoji_insertions, spaced_uppercase, syntactic_disruption, recondite_word
from image_adv_gen.perturb_func import imagenet_C_corrupt, rotate_left, rotate_right, rotate_180, flip_left_right, origin_image
from collections import Counter
import asyncio

caption_prompt = "Generate a concise description of the following image, highlighting key elements, emotions, and notable details."
vqa_prompt_format = "Given the image and question, provide an answer.\n # Quesion:"

text_adv_type = ["spelling_missing_letter", "spelling_capitalize_letter", "spelling_incorrect_letter", "spelling_insert_space", "spelling_repeated_letter", "spelling_swap_letter", "emoji_insertions", "spaced_uppercase", "syntactic_disruption", "recondite_word"]

image_adv_type = ["imagenet_C_corrupt_0", "imagenet_C_corrupt_1", "imagenet_C_corrupt_2", "imagenet_C_corrupt_3", "imagenet_C_corrupt_4", "imagenet_C_corrupt_5", "imagenet_C_corrupt_6", "imagenet_C_corrupt_7", "imagenet_C_corrupt_8", "imagenet_C_corrupt_9", "imagenet_C_corrupt_10", "imagenet_C_corrupt_11", "imagenet_C_corrupt_12", "imagenet_C_corrupt_13", "imagenet_C_corrupt_14", "imagenet_C_corrupt_15", "imagenet_C_corrupt_16", "imagenet_C_corrupt_17", "imagenet_C_corrupt_18", "rotate_left", "rotate_right", "rotate_180", "flip_left_right"]

def load_annotations(annotations_path):
    with open(annotations_path) as f:
        return json.load(f)

def save_image(image_path, save_path):
    with Image.open(image_path) as img:
        img.save(save_path)

def process_image_adv(adv_type, ori_image_path_name, adv_image_path_name):
    if adv_type.startswith("imagenet_C_corrupt"):
        severity = int(adv_type.split("_")[-1])
        imagenet_C_corrupt(ori_image_path_name, adv_image_path_name, severity=1, corruption_number=severity)
    elif adv_type == "rotate_left":
        rotate_left(ori_image_path_name, adv_image_path_name)
    elif adv_type == "rotate_right":
        rotate_right(ori_image_path_name, adv_image_path_name)
    elif adv_type == "rotate_180":
        rotate_180(ori_image_path_name, adv_image_path_name)
    elif adv_type == "flip_left_right":
        flip_left_right(ori_image_path_name, adv_image_path_name)
    elif adv_type == "None":
        origin_image(ori_image_path_name, adv_image_path_name)

async def apply_text_adv(text_adv, question):
    if text_adv == "spelling_missing_letter":
        return spelling_missing_letter(question, if_keybert=False)
    elif text_adv == "spelling_capitalize_letter":
        return spelling_capitalize_letter(question, if_keybert=False)
    elif text_adv == "spelling_incorrect_letter":
        return spelling_incorrect_letter(question, if_keybert=False)
    elif text_adv == "spelling_insert_space":
        return spelling_insert_space(question, if_keybert=False)
    elif text_adv == "spelling_repeated_letter":
        return spelling_repeated_letter(question, if_keybert=False)
    elif text_adv == "spelling_swap_letter":
        return spelling_swap_letter(question, if_keybert=False)
    elif text_adv == "emoji_insertions":
        return await emoji_insertions(question, if_keybert=False)
    elif text_adv == "spaced_uppercase":
        return spaced_uppercase(question, if_keybert=False)
    elif text_adv == "syntactic_disruption":
        return await syntactic_disruption(question)
    elif text_adv == "recondite_word":
        return await recondite_word(question)
    return question

async def mscoco(datapath, num=50, ori_image_dir="mscoco-ori", adv_image_dir="mscoco-adv", output_root="../output"):
    annotations_path = os.path.join(datapath, 'annotations', 'captions_train2014.json')
    images_dir = os.path.join(datapath, 'train2014')
    annotations = load_annotations(annotations_path)
    dataset = annotations['annotations']
    images = {img['id']: img['file_name'] for img in annotations['images']}

    ori_image_dir = os.path.join(output_root, ori_image_dir)
    adv_image_dir = os.path.join(output_root, adv_image_dir)
    os.makedirs(ori_image_dir, exist_ok=True)
    os.makedirs(adv_image_dir, exist_ok=True)

    adv_type = "image"
    text_adv = "None"

    sampled_data = random.sample(dataset, num)
    data = []

    counter = 0
    for entry in sampled_data:
        ori_image_id = entry['image_id']
        ori_image_name = images[ori_image_id]
        ori = f"ori_image_{counter}.jpg"
        adv = f"adv_image_{counter}.jpg"
        ori_image_path_name = os.path.join(ori_image_dir, ori)  
        adv_image_path_name = os.path.join(adv_image_dir, adv)

        original_image_path = os.path.join(images_dir, ori_image_name)
        save_image(original_image_path, ori_image_path_name)

        image_adv = random.choice(image_adv_type)
        process_image_adv(image_adv, ori_image_path_name, adv_image_path_name)

        data.append({
            'ori_image_path': os.path.relpath(ori_image_path_name, output_root),
            'adv_image_path': os.path.relpath(adv_image_path_name, output_root),
            'ori_prompt': caption_prompt,
            'ori_caption': entry['caption'],
            'adv_prompt': caption_prompt,
            'adv_caption': entry['caption'],
            'adv_type': adv_type,
            'text_adv': text_adv,
            'image_adv': image_adv,
            'dataset_type': "caption"
        })
        
        counter += 1

    return data

async def vqa2(datapath, num=1500, ori_image_dir="vqav2-ori", adv_image_dir="vqav2-adv", output_root="../output"):
    with open(os.path.join(datapath, 'v2_mscoco_train2014_annotations.json')) as f:
        annotations = json.load(f)

    with open(os.path.join(datapath, 'v2_OpenEnded_mscoco_train2014_questions.json')) as f:
        questions = json.load(f)['questions']

    question_map = {q['question_id']: (q['image_id'], q['question']) for q in questions}

    images_dir = os.path.join(datapath, 'train2014')
    dataset = annotations['annotations']

    ori_image_dir = os.path.join(output_root, ori_image_dir)
    adv_image_dir = os.path.join(output_root, adv_image_dir)
    os.makedirs(ori_image_dir, exist_ok=True)
    os.makedirs(adv_image_dir, exist_ok=True)

    sampled_data = random.sample(dataset, num)
    data = []

    count = 0
    for entry in sampled_data:
        a = 0
        b = 0
        question_id = entry['question_id']
        answers = entry['answers']

        yes_answers = [ans['answer'] for ans in answers if ans['answer_confidence'] == "yes"]
        maybe_answers = [ans['answer'] for ans in answers if ans['answer_confidence'] == "maybe"]
        no_answers = [ans['answer'] for ans in answers if ans['answer_confidence'] == "no"]

        final_answer = None

        if yes_answers:
            yes_answer_counts = Counter(yes_answers)
            final_answer = yes_answer_counts.most_common(1)[0][0]
        elif maybe_answers:
            maybe_answer_counts = Counter(maybe_answers)
            final_answer = maybe_answer_counts.most_common(1)[0][0]
        elif no_answers:
            no_answer_counts = Counter(no_answers)
            final_answer = no_answer_counts.most_common(1)[0][0]

        if question_id not in question_map:
            continue

        ori_image_id, ori_question = question_map[question_id]
        ori_image_name = f"COCO_train2014_{ori_image_id:012d}.jpg"
        ori = f"ori_image_{count}.jpg"
        adv = f"adv_image_{count}.jpg"
        ori_image_path_name = os.path.join(ori_image_dir, ori)  
        adv_image_path_name = os.path.join(adv_image_dir, adv)

        original_image_path = os.path.join(images_dir, ori_image_name)
        save_image(original_image_path, ori_image_path_name)

        random.seed()
        adv_type = random.choice(["image", "text", "image_text"])
        image_adv = "None"
        text_adv = "None"
        adv_question = ori_question

        if adv_type == "image":
            image_adv = random.choice(image_adv_type)
            process_image_adv(image_adv, ori_image_path_name, adv_image_path_name)
        elif adv_type == "text":
            process_image_adv(image_adv, ori_image_path_name, adv_image_path_name)
            text_adv = random.choice(text_adv_type)
            adv_question = await apply_text_adv(text_adv, ori_question)
        elif adv_type == "image_text":
            random.seed()
            a = random.randint(0, 9)
            b = random.randint(0, 22)
            image_adv = image_adv_type[b]
            text_adv = text_adv_type[a]
            process_image_adv(image_adv, ori_image_path_name, adv_image_path_name)
            adv_question = await apply_text_adv(text_adv, ori_question)
            if adv_question == ori_question:
                adv_type = "image"
                text_adv = "None"

        ori_prompt = vqa_prompt_format + ori_question + "\n# Answer: "
        adv_prompt = vqa_prompt_format + adv_question + "\n# Answer: "

        data.append({
            'ori_image_path': os.path.relpath(ori_image_path_name, output_root),
            'adv_image_path': os.path.relpath(adv_image_path_name, output_root),
            'ori_prompt': ori_prompt,
            'ori_question': ori_question,
            'adv_prompt': adv_prompt,
            'adv_question': adv_question,
            'adv_type': adv_type,
            'text_adv': text_adv,
            'image_adv': image_adv,
            'groud-truth': final_answer,
            'dataset_type': "vqa"
        })
        count += 1

    return data

async def generate_data(datapath, mscoco_num=5, vqa_num=5, output_root="../output"):
    data1 = await mscoco(datapath, num=mscoco_num, output_root=output_root)
    data2 = await vqa2(datapath, num=vqa_num, output_root=output_root)

    return data1, data2

async def main():
    datapath = "../MSCOCO"
    output_root = "../output"

    combined_data = await generate_data(datapath, mscoco_num=5, vqa_num=5, output_root=output_root)

    output_file = os.path.join(output_root, "combined_data.json")
    os.makedirs(output_root, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)

    print(f"Data saved to {output_file}")

    return combined_data

if __name__ == "__main__":
    asyncio.run(main())
