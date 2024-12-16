import os
import json
from concurrent.futures import ThreadPoolExecutor

file_type = os.getenv('FILE_TYPE', 'all')

directories = {
    "law": "../../temp_file/web_retrieval/privacy/law",
    "organization": "../../temp_file/web_retrieval/privacy/organization",
    "people": "../../temp_file/web_retrieval/privacy/people"
}

output_files = {
    "law": "../../temp_file/law_merged.json",
    "organization": "../../temp_file/organization_merged.json",
    "people": "../../temp_file/people_merged.json"
}

def merge_json_files(directory, output_file):
    merged_data = []

    if not os.path.isdir(directory):
        print(f"Directory does not exist: {directory}")
        return

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        merged_data.extend(data)
                    elif isinstance(data, dict):
                        merged_data.append(data)
                    else:
                        print(f"Unsupported data type in file: {file_path}")
            except json.JSONDecodeError as e:
                print(f"Error reading JSON file: {file_path}\nError message: {e}")
            except Exception as e:
                print(f"Error processing file: {file_path}\nError message: {e}")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=4, ensure_ascii=False)
        print(f"All JSON files successfully merged into {output_file}")
    except Exception as e:
        print(f"Error writing to output file: {output_file}\nError message: {e}")

def main():
    with ThreadPoolExecutor() as executor:
        futures = []
        if file_type == 'all':
            for key in directories.keys():
                directory = directories[key]
                output_file = output_files[key]
                futures.append(executor.submit(merge_json_files, directory, output_file))
        else:
            if file_type in directories:
                directory = directories[file_type]
                output_file = output_files[file_type]
                futures.append(executor.submit(merge_json_files, directory, output_file))
            else:
                print(f"Unsupported file type: {file_type}")
        
        for future in futures:
            try:
                future.result()  
            except Exception as e:
                print(f"An error occurred during file merging: {e}")

if __name__ == "__main__":
    main()
