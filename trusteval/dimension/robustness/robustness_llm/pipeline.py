import sys
import os
import asyncio
import json
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 引入 Saver 类

# 动态添加 src 目录到 sys.path
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from dimension.robustness.robustness_llm.robustness_code import process_ground_truth, process_open_question

def remove_duplicate_originals(data):
    # 移除所有带有 `original_original_` 前缀的字段
    for item in data:
        keys_to_remove = [key for key in item if key.startswith('original_original_')]
        for key in keys_to_remove:
            del item[key]

    for item in data:
        keys_to_remove = [key for key in item if key.startswith('original_dataset_type')]
        for key in keys_to_remove:
            del item[key]

# 封装主函数
async def pipeline(base_dir=None):
    # 获取绝对路径
    if base_dir:
        dataset_dir = base_dir
    else:
        dataset_dir = os.path.join(root_dir,'section' ,'robustness','robustness_llm', 'dataset')

    # 确保路径存在
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"The dataset directory does not exist: {dataset_dir}")

    # 设置环境变量或修改process_ground_truth函数以使用绝对路径
    os.environ['DATASET_DIR'] = dataset_dir
    os.environ['PROJECT_ROOT'] = root_dir  # 新增这一行
    # 处理数据
    ground_truth_data = await process_ground_truth()
    open_question_data = await process_open_question()
    #combined_data = ground_truth_data + open_question_data

    file_name1 = os.path.join(dataset_dir, 'open_ended_data.json')
    file_name2 = os.path.join(dataset_dir, 'ground_truth_data.json')
    # 删除重复项
    remove_duplicate_originals(ground_truth_data)
    remove_duplicate_originals(open_question_data)
    # 保存数据
    with open(file_name1, 'w', encoding='utf-8') as f:
        json.dump(open_question_data, f, ensure_ascii=False, indent=4)

    with open(file_name2, 'w', encoding='utf-8') as f:
        json.dump(ground_truth_data, f, ensure_ascii=False, indent=4)

# 入口函数
if __name__ == '__main__':
    asyncio.run(pipeline())