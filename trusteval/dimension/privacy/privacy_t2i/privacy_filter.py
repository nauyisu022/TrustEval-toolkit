import os
import json

for entity in ['organization','people']:
    
    directory = f'intermediate/web_retrieval/{entity}'

    # Directory to save filtered JSON files
    filtered_directory = f'intermediate/web_retrieval/{entity}_filtered'
    os.makedirs(filtered_directory, exist_ok=True)

    def dict_to_sentence(d):
        sentence_parts = []
        for key, value in d.items():
            sentence_parts.append(f"{key}: {value}")
        return ", ".join(sentence_parts)

    # Function to count words in a string
    def count_words_in_string(s):
        return len(s.split())

    # Function to convert Example lists and filter them
    def process_example(example):
        if isinstance(example, list):
            if all(isinstance(item, dict) for item in example):
                # Convert each dictionary to a sentence
                converted_example = [dict_to_sentence(item) for item in example]
                return converted_example
            elif all(isinstance(item, str) for item in example):
                # Already a list of strings
                return example
        return []

    # Function to filter JSON files
    def filter_json_files(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                
                filtered_data = []
                for entry in data:
                    if "Example" in entry:
                        # Convert Example to list of sentences if needed
                        processed_example = process_example(entry["Example"])
                        # Update entry with processed example
                        entry["Example"] = processed_example
                        print(processed_example)
                        # Count total words in processed example
                        total_word_count = sum(count_words_in_string(sentence) for sentence in processed_example)
                        print(total_word_count)
                        #print(a)
                        if total_word_count > 15:
                            filtered_data.append(entry)
                
                if filtered_data:
                    filtered_file_path = os.path.join(filtered_directory, filename)
                    with open(filtered_file_path, 'w', encoding='utf-8') as f:
                        json.dump(filtered_data, f, indent=4)

    # Run the filter function
    filter_json_files(directory)

    print("Filtering complete. Filtered files are stored in:", filtered_directory)