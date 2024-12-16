import json

def increment_id_in_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        if 'id' in item:
            item['id'] += 1

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

file_path = 'section/safety/safety_t2i/final/safety_final_descriptions.json'
increment_id_in_json(file_path)