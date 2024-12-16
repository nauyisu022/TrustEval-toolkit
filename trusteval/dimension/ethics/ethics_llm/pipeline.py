import os
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from data_process import run_processing
from test_case_builder import run_all_generations

async def pipeline(base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    print("Running data processing...")
    await run_processing(base_dir)
    print("Data processing completed.\n")

    print("Running case generation...")
    await run_all_generations(base_dir)
    print("Case generation completed.\n")



def main(base_dir=None):
    asyncio.run(pipeline(base_dir))

if __name__ == "__main__":
    main()