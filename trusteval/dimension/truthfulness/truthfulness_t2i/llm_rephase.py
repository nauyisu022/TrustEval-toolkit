import json
from openai import AzureOpenAI
import time

_prompt="""Generate natural language captions or prompts for a list of keywords or phrases based on the following guidelines:
    
Input: A list of key words or phrases.

Output: Ten coherent, flowing sentence or paragraph that incorporates all the input key words/phrases in different ways.

Rules:
- Do not add any new content or concepts not present in the original input.
- Do not remove or omit any of the provided key words/phrases.
- Ensure the output is grammatically correct and reads naturally.
- Maintain the original meaning and intent of the key words. Do not use words that are too uncommon or obscure.
- Use appropriate conjunctions, prepositions, and sentence structures to connect the key words seamlessly. 
- Format your answer as a JSON object with five keys "1", "2", "3", "4", "5" and the value as the output modified sentence or paragraph. Do not output anything else.
- From "1" to "5", the sentences should be increasingly detailed and creative. But still strictly adhere to the key words in the original input.

Your task is to transform the given key words into five different fluent, comprehensive sentence or paragraph that similar to captions for photos, while strictly adhering to those rules.

Here is the input:
[Start of Input]
{input}
[End of Input]
"""

def call_llm(input):
    client = AzureOpenAI(
        api_key='',  # replace with your actual API key
        api_version='2023-12-01-preview',
        azure_endpoint='' # replace with your actual endpoint
    )
    success = False
    attempt = 0
    while not success and attempt < 3:
        try:
            chat_completion = client.chat.completions.create(
                model='', # replace with your actual model name
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Please follow user's instructions strictly and output in JSON format."},
                    {"role": "user", "content": _prompt.format(input=input)}],
                temperature=0.7,
                max_tokens=512,
                top_p=0.95,
            )
            
            success = True
            output = chat_completion.choices[0].message.content.strip()
            print(json.loads(output))
            dict = json.loads(output)
            if dict == None:
                raise Exception("LLM returned None")
            return dict
        except Exception as e:
            print(e)
            attempt += 1
            time.sleep(3)
            
    return None


import argparse

def main():
    parser = argparse.ArgumentParser(description='Rephrase prompts using LLM')
    parser.add_argument('input_file', type=str, help='Path to the input JSON file')
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        input_data = json.load(f)

    for key, value in input_data.items():
        output = call_llm(value["prompt"])
        input_data[key]["llm_rephrased_prompt"] = output

    output_file = args.input_file.replace('.json', '_rephrased.json')
    with open(output_file, 'w') as f:
        json.dump(input_data, f)

if __name__ == "__main__":
    main()