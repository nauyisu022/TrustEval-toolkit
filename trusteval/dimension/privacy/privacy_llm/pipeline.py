import subprocess
import os
import shutil
import sys
import json

def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    for item in data:
        try:
            if isinstance(item.get('transformed_query'), str):
                new_item = {
                    'transformed_query': item['transformed_query'],
                    'enhancement_method': item.get('enhancement_method', ''),
                    'context_query': item.get('context_query', '')
                }
            elif isinstance(item.get('transformed_query'), dict):
                new_item = {
                    'transformed_query': item['transformed_query'].get('sentence', ''),
                    'enhancement_method': item['transformed_query'].get('enhancement_method', ''),
                    'context_query': item.get('context_query', '')
                }
            else:
                print(f"Unexpected format for item: {item}")
                continue

            new_item['prompt'] = f"{new_item['context_query']} {new_item['transformed_query']}"
            processed_data.append(new_item)

        except Exception as e:
            print(f"Error processing item: {item}")
            print(f"Error message: {str(e)}")
            continue

    output_path = file_path.replace('.json', '.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    print(f"Processed file saved to: {output_path}")

def create_directories():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    final_dir = os.path.join(current_dir, "final")
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
        print(f"Created directory: {final_dir}")

    temp_file_dir = os.path.join(current_dir, "temp_file")
    if not os.path.exists(temp_file_dir):
        os.makedirs(temp_file_dir)
        print(f"Created directory: {temp_file_dir}")

    return current_dir

current_dir = create_directories()

def run_script(script_path, timeout):
    original_dir = os.getcwd()
    process = None

    try:
        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)

        if script_dir:
            os.chdir(script_dir)

        print(f"\n=== Starting {script_name} in {script_dir} ===")

        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
            universal_newlines=True  
        )

        stdout, stderr = process.communicate(timeout=timeout)

        if process.returncode == 0:
            print(f"=== Successfully completed {script_name} ===")
            print("Output:", stdout)
        else:
            print(f"=== Error executing {script_name} ===")
            print("Error output:", stderr)

        return process.returncode

    except subprocess.TimeoutExpired:
        print(f"Error: {script_name} exceeded the timeout limit of {timeout} seconds.")
        if process:
            process.kill()
            stdout, stderr = process.communicate()
            print("Partial output before timeout:", stdout)
        return -1

    except Exception as e:
        print(f"Unexpected error running {script_name}: {e}")
        if process:
            process.kill()
        return -1

    finally:
        os.chdir(original_dir)
        print(f"Returned to original directory: {original_dir}")

def copy_and_clean_folders(source_dir, target_dir):
    folders = ['temp_file', 'final']

    for folder in folders:
        source_folder = os.path.join(source_dir, folder)
        target_folder = os.path.join(target_dir, folder)

        try:
            if os.path.exists(source_folder):
                if os.path.exists(target_folder):
                    shutil.rmtree(target_folder)

                shutil.copytree(source_folder, target_folder)
                print(f"Copied {folder} to {target_folder}")

                for item in os.listdir(source_folder):
                    item_path = os.path.join(source_folder, item)
                    try:
                        if os.path.isfile(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        print(f"Error cleaning {item_path}: {e}")
                print(f"Cleaned contents of {source_folder}")
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")

def pipeline(base_dir=None):
    timeout_limit = 5000
    os.environ['FILE_TYPE'] = 'all'
    os.environ['TIMEOUT_LIMIT'] = str(timeout_limit)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts = [
        ('run_whole.py', 'test_case_builder'),
        ('aspects_filter_whole.py', 'test_case_builder'),
        ('test_case_builder_whole.py', 'test_case_builder'),
        ('Merge_json_whole.py', 'metadata_curator/Preprocess'),
        ('add_context_LLM_whole.py', 'metadata_curator/Preprocess')
    ]

    for script_name, script_subdir in scripts:
        script_path = os.path.join(current_dir, script_subdir, script_name)
        print(f"\nExecuting {script_name}...")

        if not os.path.exists(script_path):
            print(f"Warning: Script not found at {script_path}")
            continue

        return_code = run_script(script_path, timeout_limit)

        if return_code != 0:
            print(f"Warning: {script_name} failed with return code {return_code}")
            continue

    print("\nAll scripts execution attempted.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    final_dir = os.path.join(current_dir, 'final')
    target_files = [
        'Rephrased_malicious_law.json',
        'Rephrased_malicious_organization.json',
        'Rephrased_malicious_people.json'
    ]

    for file_name in target_files:
        file_path = os.path.join(final_dir, file_name)
        if os.path.exists(file_path):
            print(f"Processing {file_name}...")
            process_json_file(file_path)
        else:
            print(f"File not found: {file_path}")

    if base_dir is not None:
        try:
            copy_and_clean_folders(current_dir, base_dir)
            print("Folders copied and cleaned.")
        except Exception as e:
            print(f"Error during folder operations: {e}")
    else:
        print("No folder copy requested (base_dir is None)")

if __name__ == "__main__":
    pipeline()