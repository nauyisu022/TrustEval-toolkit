import os
import json
from concurrent.futures import ThreadPoolExecutor

# 从环境变量中获取文件类型，如果没有设置则默认为 'all'
file_type = os.getenv('FILE_TYPE', 'all')

# 文件夹路径和输出文件的映射
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
    """
    合并指定目录中的 JSON 文件并输出到指定文件
    """
    merged_data = []

    # 检查目录是否存在
    if not os.path.isdir(directory):
        print(f"Directory does not exist: {directory}")
        return

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                # 打开并读取每个 JSON 文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 如果 JSON 文件包含列表，则扩展 merged_data 列表
                    if isinstance(data, list):
                        merged_data.extend(data)
                    # 如果 JSON 文件包含对象，则将其添加到列表中
                    elif isinstance(data, dict):
                        merged_data.append(data)
                    else:
                        print(f"Unsupported data type in file: {file_path}")
            except json.JSONDecodeError as e:
                print(f"Error reading JSON file: {file_path}\nError message: {e}")
            except Exception as e:
                print(f"Error processing file: {file_path}\nError message: {e}")

    # 写入合并后的数据到新的 JSON 文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=4, ensure_ascii=False)
        print(f"All JSON files successfully merged into {output_file}")
    except Exception as e:
        print(f"Error writing to output file: {output_file}\nError message: {e}")

def main():
    # 使用线程池并发执行文件合并
    with ThreadPoolExecutor() as executor:
        futures = []
        # 如果 file_type 是 "all"，并行处理所有类型
        if file_type == 'all':
            for key in directories.keys():
                directory = directories[key]
                output_file = output_files[key]
                # 提交任务到线程池
                futures.append(executor.submit(merge_json_files, directory, output_file))
        else:
            # 并行处理指定类型的文件
            if file_type in directories:
                directory = directories[file_type]
                output_file = output_files[file_type]
                futures.append(executor.submit(merge_json_files, directory, output_file))
            else:
                print(f"Unsupported file type: {file_type}")
        
        # 等待所有任务完成
        for future in futures:
            try:
                future.result()  # 捕获合并任务中的异常
            except Exception as e:
                print(f"An error occurred during file merging: {e}")

if __name__ == "__main__":
    main()
