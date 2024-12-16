import csv
import json
from tqdm import tqdm
import numpy as np
from prettytable import PrettyTable
import os
import time
import openai

try:
    with open("./apikey.txt", "r") as f:
        api_key = f.read()
except:
    api_key = ''

input_file_name = "hallusion_data_with_res.json"
output_file_name = "hallusion_data_with_res_evaluated.json"


def evaluate_by_chatgpt(data, save_json_path, output_entry="res", correctness_entry="correctness", gpt_model="gpt-4o", load_json=True):
    if load_json and os.path.exists(save_json_path):
        with open(save_json_path, 'r') as f:
            output = json.load(f)
    else:
        output = []

    for sample in tqdm(data[len(output):]):
        prompt = 'Imagine you are an intelligent teacher. Thoroughly read the question, reference answer and the prediction answer to ensure a clear understanding of the information provided. Assess the correctness of the predictions. '
        prompt += 'If the prediction answer does not conflict with the reference answer, please generate “correct”. If the prediction answer conflict with the reference answer, please generate “incorrect”. The output should only be “correct” or “incorrect”. \n\n Question:'
        prompt += sample['prompt']
        prompt += '\nReference answer: '
        prompt += sample['ground_truth']
        prompt += '\nPrediction answer:'
        prompt += sample[output_entry]
        prompt += '\nOutput:'

        # https://github.com/openai/openai-python/issues/322#issuecomment-1767841683
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=gpt_model, 
                    messages=[{"role": "user", "content": prompt}], 
                    api_key=api_key, 
                    request_timeout=5)
                break
            except:
                print("Timeout, retrying...")
                time.sleep(5)  # Wait for 5 seconds before retrying

        output_text = response['choices'][0]['message']['content']


        if 'incorrect' in output_text.lower(): 
            gpt_correctness = False

        elif 'correct' in output_text.lower():
            gpt_correctness = True
        else:
            gpt_correctness = False

        sample[correctness_entry] = gpt_correctness

        output.append(sample)

        with open(save_json_path, 'w') as f:
            json.dump(output, f)

    return output


def get_percentage(data, model_correctness_entry): # per question

    correct = 0
    total = len(data)

    for r in data:
        if r[model_correctness_entry]:
            correct += 1

    return correct, total


if __name__ == "__main__":

    with open(input_file_name, 'r') as file:
        data = json.load(file)
        print(input_file_name)

    data = evaluate_by_chatgpt(data, output_file_name, output_entry="res", correctness_entry="correctness", gpt_model="gpt-4o", load_json=True)

    # data = [i for i in data if i["subcategory"] == "illusion"]

    correct, total = get_percentage(data, model_correctness_entry="correctness")
    
    print("Accuracy:" + str(round(100 * correct/total, 4)))

#     human_check_correctness = [i["correctness"] for i in all_data]

#     print("##### Human Evaluate #####")
# ;
#     # question level
#     table1 = [["per question", "Total"], 
#               ["Overall", round(100 * all_data["correct"]/all_data["total"], 4)]]
#     tab1 = PrettyTable(table1[0])
#     tab1.add_rows(table1[1:])
#     print(tab1)
#     q_acc_human = round(100 * all_data["correct"]/all_data["total"], 4)
#     print(q_acc_human)
