from pertubation_tools import spelling_missing_letter, spelling_insert_space, spaced_uppercase, social_tagging, spelling_capitalize_letter,spelling_incorrect_letter, spelling_repeated_letter, spelling_swap_letter, recondite_word,remove_inner_letter,repeat_inner_letter,replace_inner_letter, capitalize_inner_letter,syntactic_disruption,distractive_text, multilingual_blend,emoji_insertions
from src.saver import Saver
import random
import os
import tqdm
import json

final_base_path = os.path.abspath('final')
saver = Saver(final_base_path)

def modify_descriptions(sentence):
    spelling_mistake_functions = [
        spelling_missing_letter,
        spelling_insert_space,
        spelling_capitalize_letter,
        spelling_incorrect_letter,
        spelling_repeated_letter,
        spelling_swap_letter,
        remove_inner_letter,
        repeat_inner_letter,
        replace_inner_letter, 
        capitalize_inner_letter,
    ]

    spelling_mistake = random.choice(spelling_mistake_functions)
    perturbation_functions = [
        spelling_mistake,
        spaced_uppercase,
        social_tagging,
        recondite_word,
        syntactic_disruption,
        distractive_text,
        multilingual_blend,
        emoji_insertions
    ]
    
    # choose a tool randomly
    perturbation_function = random.choice(perturbation_functions)
    # print(perturbation_functions)
    # print(perturbation_function)
    modified_description = perturbation_function(sentence)
    modified_tool_name = perturbation_function.__name__
    print(modified_tool_name)
    
    result = {
        "image_description": sentence,
        "modified_description": modified_description,
        "modified_tool":modified_tool_name
    }
    return result

def main():
    with open('intermediate/robustness_raw_descriptions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    modified_data = []

    for item in tqdm.tqdm(data, desc="Processing descriptions"):
        sentence = item['image_description']
        modified_description = modify_descriptions(sentence)
        modified_description['id'] = len(modified_data) + 1
        modified_data.append(modified_description)

    saver.save_data(modified_data, 'robustness_final_descriptions.json')

if __name__ == "__main__":
    main()