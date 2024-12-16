import os
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from Ration_Vizwiz import process_images_vizwiz
from Ration_VISPR import process_images_vispr
from transformat import process_multiple_files
from Diversify import diversify_multiple_files

def pipeline(base_dir=None):
    if base_dir:
        current_dir=base_dir
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    vizwiz_input_folder = os.path.join(current_dir, 'Vizwiz_Priv')
    vispr_input_folder = os.path.join(current_dir, 'VISPR')
    output_folder = os.path.join(current_dir, 'output')
    print("process vizwiz datasets...")
    process_images_vizwiz(vizwiz_input_folder, output_folder)
    print("process vispr datasets...")
    process_images_vispr(vispr_input_folder, output_folder)
    print("process transformat...")
    process_multiple_files(output_folder, output_folder)
    print("process contextual variator...")
    asyncio.run(diversify_multiple_files(output_folder, output_folder))

if __name__ == "__main__":
    pipeline()