    
import os,sys
import csv
import json
import re
from tenacity import retry, wait_random_exponential, stop_after_attempt
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(project_root)
print(project_root)
from src.metadata_curator.pipeline import TextWebSearchPipeline
from src.generation import ModelService

service = ModelService(
    request_type='llm',
    handler_type='api',
    model_name='gpt-4o',
    config_path=os.path.join(project_root, 'src/config/config.yaml'),
    temperature=0.6
)

def get_api_res(string):
    service = ModelService(
        request_type='llm',
        handler_type='api',
        model_name='gpt-4o',
        config_path=os.path.join(project_root, 'src/config/config.yaml'),
        temperature=0,
        top_p=1,
    )
    response =  service.process(string)
    return response


async def run_pipeline(instruction, basic_information, output_path):
    output_format = {
        "Example": [
            "Specific example sentence 1 mentioned on the webpage with scenarios and defense for the disparagement or discrimination",
            "Specific example sentence x mentioned on the webpage with scenarios and defense for the disparagement or discrimination (and so on)"
        ]
    }

    extractor = TextWebSearchPipeline(
        instruction=instruction,
        basic_information=basic_information,
        need_azure=True,
        output_format=output_format,
        keyword_model="gpt-4o",
        response_model="gpt-4o",
        include_url=True,
        include_summary=True,
        include_original_html=False,
        include_access_time=True
    )

    await extractor.run(output_file=output_path)
    

class FairnessSaver:
    def __init__(self, base_directory=None):
        """
        Initialize the FairnessSaver with an optional base directory.
        All files will be saved in this directory.
        """
        self.base_directory = base_directory or os.getcwd()

    def _get_full_path(self, filename):
        """Helper method to get the full path of the file."""
        return filename #os.path.join(self.base_directory, filename)

    def create_output_directory(self, directory_name):
        """
        Create the output directory if it doesn't exist.
        """
        full_path = self._get_full_path(directory_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f"Created directory: {full_path}")
        else:
            print(f"Directory already exists: {full_path}")

    def save_to_csv(self, data:dict, filename:str):
        full_path = self._get_full_path(filename)
        file_exists = os.path.isfile(full_path)
        with open(full_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Response'])
            writer.writerow([data])

    def save_to_json(self, data:dict, filename:str):
        full_path = self._get_full_path(filename)
        with open(full_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def load_json_data(self, json_path:str):
        full_path = self._get_full_path(json_path)
        with open(full_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def process_csv(self, filename):
        directory, base_filename = os.path.split(filename)
        temp_filename = os.path.join(directory, 'temp_' + base_filename)
        with open(filename, mode='r', encoding='utf-8') as infile, open(temp_filename, mode='w', newline='', encoding='utf-8') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            header = next(reader)
            writer.writerow(header)
            for row in reader:
                processed_row = [re.sub(r'^\d+\.\s*', '', cell) for cell in row]
                writer.writerow(processed_row)
        os.replace(temp_filename, filename)

    def merge_json_files(self, folder_path, output_file):
        merged_data = []
        current_id = 1
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                category_name = os.path.splitext(filename)[0]
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    for obj in data:
                        obj['id'] = current_id
                        obj['category'] = category_name
                        merged_data.append(obj)
                        current_id += 1
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(merged_data, outfile, ensure_ascii=False, indent=4)

    def load_jsonl_data(self, jsonl_file):
        """
        Load intermediate from a JSONL file where each line is a valid JSON object.
        :param jsonl_file: Path to the JSONL file
        :return: List of JSON objects
        """
        full_path = self._get_full_path(jsonl_file)
        data = []
        with open(full_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line.strip()))  
        return data
    