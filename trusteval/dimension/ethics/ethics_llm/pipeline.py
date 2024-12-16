import os
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将 code 文件夹添加到 Python 路径
sys.path.append(current_dir)
# 从各模块中导入所需函数
from data_process import run_processing
from test_case_builder import run_all_generations

async def pipeline(base_dir=None):
    # 如果base_dir为None，使用当前脚本所在目录
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    print("Running data processing...")
    await run_processing(base_dir)
    print("Data processing completed.\n")

    # 第二步：生成案例
    print("Running case generation...")
    await run_all_generations(base_dir)
    print("Case generation completed.\n")



def main(base_dir=None):
    asyncio.run(pipeline(base_dir))

if __name__ == "__main__":
    main()  # 使用默认路径
    # main("/path/to/your/directory")  # 使用指定路径