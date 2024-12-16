import json
import os
import asyncio
import sys

# Set up the path to import ContextualVariator
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../src"))
sys.path.append(project_root)

from contextual_variator import ContextualVariator

# Initialize the Diversity Enhancer
supported_operations = [
    "transform_expression",
    "paraphrase_sentence"
]
enhancer = ContextualVariator(supported_operations)

async def generate_diverse_rephrasing(question):
    try:
        return await enhancer.enhance_diversity(
            question,
            extra_instructions="Make sure there is no semantic shift! The result must be a malicious query!"
        )
    except Exception as e:
        print(f"Error in generate_diverse_rephrasing: {e}")
        return question  # Return original question if enhancement fails

async def process_questions(input_file, output_file):
    try:
        with open(input_file, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading {input_file}: {e}")
        return

    updated_data = []
    for index, item in enumerate(data):
        original_question = item.get("prompt")
        if not original_question:
            continue  # Skip if no question is present

        try:
            rephrased_question = await generate_diverse_rephrasing(original_question)
            print(f"Original question: {original_question}\nRephrased question: {rephrased_question}\n")
            item["prompt"] = rephrased_question
            updated_data.append(item)
        except Exception as e:
            print(f"Error processing question {index}: {e}")
            updated_data.append(item)  # Keep original item if processing fails

    try:
        with open(output_file, 'w') as file:
            json.dump(updated_data, file, indent=4)
        print(f"Updated questions saved to {output_file}")
    except IOError as e:
        print(f"Error writing to {output_file}: {e}")

async def diversify_multiple_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    input_files = [
        ('VISPR_filt.json', 'VISPR_rephrased.json'),
        ('Vizwiz_filt.json', 'Vizwiz_rephrased.json')
    ]

    tasks = []
    for input_file, output_file in input_files:
        input_path = os.path.join(input_folder, input_file)
        output_path = os.path.join(output_folder, output_file)
        if os.path.exists(input_path):
            tasks.append(process_questions(input_path, output_path))
        else:
            print(f"Warning: {input_path} not found.")

    await asyncio.gather(*tasks)

def main():
    input_folder = os.path.join(current_dir, 'temp')
    output_folder = os.path.join(current_dir, 'final')

    asyncio.run(diversify_multiple_files(input_folder, output_folder))

if __name__ == "__main__":
    main()