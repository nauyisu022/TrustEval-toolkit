import asyncio
import json
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(current_dir, 'robustness_code')
sys.path.append(code_dir)
from dynamic_dataset import generate_data
sys.path.append(current_dir)
async def pipeline(base_dir=None):
    #datapath = "MSCOCO"
    if base_dir:
        output_root=base_dir
        datapath=base_dir
    else:
        output_root = "output"
        datapath = "MSCOCO"

    ms_coco_data, vqa_data = await generate_data(datapath, mscoco_num=400, vqa_num=400, output_root=output_root)

    ms_coco_file = os.path.join(output_root, "ms_coco_data.json")
    os.makedirs(output_root, exist_ok=True)
    with open(ms_coco_file, 'w', encoding='utf-8') as f:
        json.dump(ms_coco_data, f, ensure_ascii=False, indent=4)
    print(f"MSCOCO data saved to {ms_coco_file}")

    vqa_file = os.path.join(output_root, "vqa_data.json")
    with open(vqa_file, 'w', encoding='utf-8') as f:
        json.dump(vqa_data, f, ensure_ascii=False, indent=4)
    print(f"VQA data saved to {vqa_file}")
    combined_data=ms_coco_data+vqa_data

    return combined_data

if __name__ == "__main__":
    result = asyncio.run(pipeline())