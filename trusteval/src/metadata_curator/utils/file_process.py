import json
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)