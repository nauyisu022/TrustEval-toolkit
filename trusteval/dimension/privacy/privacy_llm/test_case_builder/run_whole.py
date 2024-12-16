import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../src"))
metadata_curator_root = os.path.abspath(os.path.join(project_root, "metadata_curator"))
privacy_llm_dir = os.path.dirname(current_dir)
#print(metadata_curator_root)
sys.path.append(metadata_curator_root)
from pipeline import TextWebSearchPipeline
sys.path.append(current_dir)
local_metadata_curator = os.path.join(privacy_llm_dir, "metadata_curator")


# 从环境变量中获取文件类型，如果没有设置则默认为 'all'
file_type = os.getenv('FILE_TYPE', 'all')

# 定义不同类型的数据文件和输出路径
data_mapping = {
    "people": {
        "aspect_file": "../metadata_curator/aspects_guidemap/people.json",
        "output_dir": "../temp_file/web_retrieval/cases/people",
        "instruction_template": "Please find examples about the privacy related or invasion actions aim at peoples' {}, do not return its mitigation methods. Note that its peoples' privacy cases."
    },
    "organization": {
        "aspect_file": "../metadata_curator/aspects_guidemap/organization.json",
        "output_dir": "../temp_file/web_retrieval/cases/organization",
        "instruction_template": "Please find examples about the privacy related or invasion actions aim at organizations' {}, do not return its mitigation methods. Note that its organizations' privacy cases."
    }
}

async def run_pipeline(e, instruction_template, output_dir):
    try:
        instruction = instruction_template.format(e.lower())
        print(f"Processing instruction: {instruction}")
        print(f"Output directory: {output_dir}")

        output_path = os.path.join(output_dir, f"{e}.json")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 初始化 TextWebSearchPipeline
        extractor = TextWebSearchPipeline(
            instruction=instruction,
            basic_information={},
            need_azure=True,
            output_format={
                "Example": [
                    "Specific example 1 mentioned on the webpage",
                    "Specific example x mentioned on the webpage (and so on)"
                ]
            },
            keyword_model="gpt-4o",
            response_model="gpt-4o",
            include_url=True,
            include_summary=True,
            include_original_html=False,
            include_access_time=True
        )

        # 运行管道并将输出保存到指定文件
        await extractor.run(output_file=output_path)
        print(f"Processed: {e}")
    except Exception as exc:
        import traceback
        print(f"Detailed error for {e}:")
        print(traceback.format_exc())
        raise

def process_data_type(data_type):
    """
    根据数据类型处理相关文件。
    """
    if data_type not in data_mapping:
        print(f"Data type '{data_type}' is not supported.")
        return

    # 获取数据文件和输出路径
    aspect_file = data_mapping[data_type]["aspect_file"]
    output_dir = data_mapping[data_type]["output_dir"]
    instruction_template = data_mapping[data_type]["instruction_template"]

    # 加载 aspect 文件
    with open(aspect_file, 'r') as f:
        data = json.load(f)

    # 创建异步任务列表
    tasks = [
        run_pipeline(el, instruction_template, output_dir)
        for v in data.values() for el in v
    ]

    # 创建一个新的事件循环，并设置为当前线程的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # 并发执行任务
    if tasks:
        loop.run_until_complete(asyncio.gather(*tasks))

    # 关闭事件循环
    loop.close()

def main():
    # 创建线程池执行器
    with ThreadPoolExecutor() as executor:
        # 如果 file_type 是 "all"，并发处理 people 和 organization
        if file_type == "all":
            futures = [
                executor.submit(process_data_type, "people"),
                executor.submit(process_data_type, "organization")
            ]
        # 如果指定为某个类型，则并行处理该类型
        elif file_type in data_mapping:
            futures = [executor.submit(process_data_type, file_type)]
        # 跳过 law 或其他不支持的类型
        else:
            print(f"Skipping unsupported file type: {file_type}")
            futures = []

        # 等待所有任务完成
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
