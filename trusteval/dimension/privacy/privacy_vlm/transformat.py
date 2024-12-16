import json
import os

def process_json_file(input_file, output_file):
    # Load the JSON data from a file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Traverse each item in the JSON data and modify "scenario_and_question"
    for item in data:
        if "scenario_and_question" in item:
            if isinstance(item["scenario_and_question"], dict) and "scenario_and_question" in item["scenario_and_question"]:
                item["prompt"] = item["scenario_and_question"]["scenario_and_question"]
            elif isinstance(item["scenario_and_question"], str):
                item["prompt"] = item["scenario_and_question"]
            del item["scenario_and_question"]

    # Save the modified JSON data to a new file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Conversion completed and saved to {output_file}.")

def process_multiple_files(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process VISPR file
    vispr_input = os.path.join(input_folder, 'VISPR_output_malicious_gen.json')
    vispr_output = os.path.join(output_folder, 'VISPR_filt.json')
    if os.path.exists(vispr_input):
        process_json_file(vispr_input, vispr_output)
    else:
        print(f"Warning: {vispr_input} not found.")

    # Process Vizwiz file
    vizwiz_input = os.path.join(input_folder, 'Vizwiz_output_malicious_gen.json')
    vizwiz_output = os.path.join(output_folder, 'Vizwiz_filt.json')
    if os.path.exists(vizwiz_input):
        process_json_file(vizwiz_input, vizwiz_output)
    else:
        print(f"Warning: {vizwiz_input} not found.")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(current_dir, 'temp')
    output_folder = os.path.join(current_dir, 'temp')
    process_multiple_files(input_folder, output_folder)

if __name__ == "__main__":
    main()