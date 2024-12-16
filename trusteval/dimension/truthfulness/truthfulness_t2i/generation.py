# a multi-processing script to generate a large number of prompts

import json
import argparse
from gma.text2vision.prompt_generator import *
from gma.text2vision.metadata import Text2ImageMetaData
from tqdm import tqdm
import multiprocessing as mp
import random   
from llm_rephase import call_llm
# Set up metadata and generator
metadata = Text2ImageMetaData(path_to_metadata="./metadata")

# Function to generate a single prompt
def generate(generator, complexity=8, number_of_global_attributes=1):
    task_plans = generator.sample_task_plans(number_of_global_attributes=number_of_global_attributes, complexity=complexity, sample_numbers=1)
    sg = task_plans[0]
    prompt = generator.generate(sg)
    prompt['image_description'] = prompt['prompt']
    del prompt['prompt']
    prompt['modified_description'] = call_llm(prompt['prompt'])
    
    return prompt

parser = argparse.ArgumentParser(description='Generate prompts with specified parameters')
parser.add_argument('--max_complexity', type=int, default=12, help='Maximum complexity (3 to specified value, inclusive)')
parser.add_argument('--max_attributes', type=int, default=11, help='Maximum number of attributes (0 to specified value, inclusive)')
parser.add_argument('--total_prompts', type=int, default=300, help='Total number of prompts to generate')
args = parser.parse_args()

# Define ranges for complexity and number of attributes
complexities = range(3, args.max_complexity + 1)
attributes = range(0, args.max_attributes + 1)

# Check if max_attributes exceeds the limit
if args.max_attributes > 12:
    raise ValueError("Maximum number of attributes cannot exceed 12")

# Total number of prompts
total_prompts = args.total_prompts

# Number of files to split into, more files means more parallelism
num_files = 10

####### end of parameters ######

print(f"Generating {total_prompts} prompts with {num_files} files")
print(f"Complexities: {complexities}")
print(f"Attributes: {attributes}")


prompts_per_file = total_prompts
# Number of prompts per complexity
prompts_per_complexity = prompts_per_file // len(complexities)
# Number of prompts per attribute within each complexity
prompts_per_attribute = prompts_per_complexity // len(attributes)
if num_files > 1:
    seed_list = [random.randint(0, 100) for _ in range(num_files)]
    print(seed_list)
else:
    seed_list = [42]

# Function to generate a batch of prompts and save to a file
def generate_batch(batch_idx):
    generator = Text2ImagePromptGenerator(metadata = metadata, seed=seed_list[batch_idx])
    file_name = f"./prompts_batch_{batch_idx}_{total_prompts}.jsonl"
    with open(file_name, "w") as f:
        idx = 0
        for complexity in tqdm(complexities, desc=f"Batch {batch_idx} - Complexity"):
            for number_of_global_attributes in attributes:
                for _ in range(prompts_per_attribute):
                    prompt = generate(generator, complexity, number_of_global_attributes)
                    prompt['id'] = idx
                    json.dump(prompt, f)
                    f.write('\n')
                    idx += 1
    print(f"Saved {idx} prompts to {file_name}")

# Create a list of batch indices to process
batch_indices = range(num_files)

# Use multiprocessing to generate each batch in parallel
if __name__ == "__main__":
    with mp.Pool(processes=num_files) as pool:
        pool.map(generate_batch, batch_indices)
