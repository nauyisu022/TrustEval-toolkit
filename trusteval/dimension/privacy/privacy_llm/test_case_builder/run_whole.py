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


file_type = os.getenv('FILE_TYPE', 'all')

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

        os.makedirs(output_dir, exist_ok=True)

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

        await extractor.run(output_file=output_path)
        print(f"Processed: {e}")
    except Exception as exc:
        import traceback
        print(f"Detailed error for {e}:")
        print(traceback.format_exc())
        raise

def process_data_type(data_type):
    if data_type not in data_mapping:
        print(f"Data type '{data_type}' is not supported.")
        return

    aspect_file = data_mapping[data_type]["aspect_file"]
    output_dir = data_mapping[data_type]["output_dir"]
    instruction_template = data_mapping[data_type]["instruction_template"]

    with open(aspect_file, 'r') as f:
        data = json.load(f)

    tasks = [
        run_pipeline(el, instruction_template, output_dir)
        for v in data.values() for el in v
    ]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if tasks:
        loop.run_until_complete(asyncio.gather(*tasks))

    loop.close()

def main():
    with ThreadPoolExecutor() as executor:
        if file_type == "all":
            futures = [
                executor.submit(process_data_type, "people"),
                executor.submit(process_data_type, "organization")
            ]
        elif file_type in data_mapping:
            futures = [executor.submit(process_data_type, file_type)]
        else:
            print(f"Skipping unsupported file type: {file_type}")
            futures = []

        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
