import json

def increment_id_in_json(file_path):
    # 读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 更新每个对象的 id 字段
    for item in data:
        if 'id' in item:
            item['id'] += 1

    # 将更新后的数据写回到源文件
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# 使用示例
file_path = 'section/safety/safety_t2i/final/safety_final_descriptions.json'
increment_id_in_json(file_path)