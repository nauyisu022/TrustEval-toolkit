import os
import json
from concurrent.futures import ThreadPoolExecutor

# 从环境变量中获取文件类型，如果没有设置则默认为 'all'
file_type = os.getenv('FILE_TYPE', 'all')

# 定义不同类型的数据目录和过滤后的输出目录
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
    """
    将字典转换为字符串句子
    """
    sentence_parts = [f"{key}: {value}" for key, value in d.items()]
    return ", ".join(sentence_parts)

def count_words_in_string(s):
    """
    计算字符串中的单词数量
    """
    return len(s.split())

def process_example(example):
    """
    处理 Example 列表，将字典转换为句子，并返回字符串列表
    """
    if isinstance(example, list):
        if all(isinstance(item, dict) for item in example):
            # 将每个字典转换为句子
            converted_example = [dict_to_sentence(item) for item in example]
            return converted_example
        elif all(isinstance(item, str) for item in example):
            # 已经是字符串列表
            return example
    return []

def filter_json_files(input_dir, filtered_dir):
    """
    过滤 JSON 文件，将满足条件的条目保存到过滤后的目录
    """
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
                        # 转换并处理 Example
                        processed_example = process_example(entry["Example"])
                        entry["Example"] = processed_example
                        print(f"Processed example for {filename}: {processed_example}")
                        
                        # 计算总单词数
                        total_word_count = sum(count_words_in_string(sentence) for sentence in processed_example)
                        print(f"Total word count for {filename}: {total_word_count}")
                        
                        # 如果单词数大于 15，添加到过滤后的数据
                        if total_word_count > 15:
                            filtered_data.append(entry)

                # 将过滤后的数据保存到新文件
                if filtered_data:
                    filtered_file_path = os.path.join(filtered_dir, filename)
                    with open(filtered_file_path, 'w', encoding='utf-8') as f:
                        json.dump(filtered_data, f, indent=4)

                print(f"Filtering complete for {filename}. Filtered files are stored in: {filtered_dir}")
            
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error processing file {filename}: {e}")

def main():
    # 创建线程池执行器
    with ThreadPoolExecutor() as executor:
        # 如果 file_type 是 "all"，并行处理 people 和 organization
        if file_type == "all":
            futures = [
                executor.submit(filter_json_files, directory_mapping["people"]["input_dir"], directory_mapping["people"]["filtered_dir"]),
                executor.submit(filter_json_files, directory_mapping["organization"]["input_dir"], directory_mapping["organization"]["filtered_dir"])
            ]
        # 如果指定为某个类型，则处理该类型
        elif file_type in directory_mapping:
            input_dir = directory_mapping[file_type]["input_dir"]
            filtered_dir = directory_mapping[file_type]["filtered_dir"]
            futures = [executor.submit(filter_json_files, input_dir, filtered_dir)]
        # 跳过 law 或其他不支持的类型
        else:
            print(f"Skipping unsupported file type: {file_type}")
            futures = []

        # 等待所有任务完成
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()
