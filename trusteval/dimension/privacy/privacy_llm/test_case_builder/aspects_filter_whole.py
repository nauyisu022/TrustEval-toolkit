import os
import json
from concurrent.futures import ThreadPoolExecutor

file_type = os.getenv('FILE_TYPE', 'all')

directory_mapping = {
    "people": {
        "input_dir": '../temp_file/web_retrieval/cases/people',
        "filtered_dir": '../temp_file/web_retrieval/cases/people_filtered'
    },
    "organization": {
        "input_dir": '../temp_file/web_retrieval/cases/organization',
        "filtered_dir": '../temp_file/web_retrieval/cases/organization_filtered'
    }
}

def dict_to_sentence(d):
    sentence_parts = [f"{key}: {value}" for key, value in d.items()]
    return ", ".join(sentence_parts)

def count_words_in_string(s):
    return len(s.split())

def process_example(example):
    if isinstance(example, list):
        if all(isinstance(item, dict) for item in example):
            converted_example = [dict_to_sentence(item) for item in example]
            return converted_example
        elif all(isinstance(item, str) for item in example):
            return example
    return []

def filter_json_files(input_dir, filtered_dir):
    os.makedirs(filtered_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                filtered_data = []
                for entry in data:
                    if "Example" in entry:
                        processed_example = process_example(entry["Example"])
                        entry["Example"] = processed_example
                        print(f"Processed example for {filename}: {processed_example}")
                        
                        total_word_count = sum(count_words_in_string(sentence) for sentence in processed_example)
                        print(f"Total word count for {filename}: {total_word_count}")
                        
                        if total_word_count > 15:
                            filtered_data.append(entry)

                if filtered_data:
                    filtered_file_path = os.path.join(filtered_dir, filename)
                    with open(filtered_file_path, 'w', encoding='utf-8') as f:
                        json.dump(filtered_data, f, indent=4)

                print(f"Filtering complete for {filename}. Filtered files are stored in: {filtered_dir}")
            
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error processing file {filename}: {e}")

def main():
    with ThreadPoolExecutor() as executor:
        if file_type == "all":
            futures = [
                executor.submit(filter_json_files, directory_mapping["people"]["input_dir"], directory_mapping["people"]["filtered_dir"]),
                executor.submit(filter_json_files, directory_mapping["organization"]["input_dir"], directory_mapping["organization"]["filtered_dir"])
            ]
        elif file_type in directory_mapping:
            input_dir = directory_mapping[file_type]["input_dir"]
            filtered_dir = directory_mapping[file_type]["filtered_dir"]
            futures = [executor.submit(filter_json_files, input_dir, filtered_dir)]
        else:
            print(f"Skipping unsupported file type: {file_type}")
            futures = []

        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()
