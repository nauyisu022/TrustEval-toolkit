import asyncio
import re
import json
from PIL import Image
import random
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../src"))
sys.path.append(project_root)
sys.path.append(current_dir)
from saver import Saver
from utils import generate_and_save_image, call_gpt4o_api

class DataProcessor:
    def __init__(self, base_folder_path):
        self.BASE_FOLDER_PATH = base_folder_path
        self.ORIGINAL_PATH = "social_chem_101.json"
        self.PROCESSED_PATH = "processed_ori_data.json"
        self.IMAGE_SAVE_FOLDER = os.path.join(base_folder_path, "generated_images")
        self.MERGED_IMAGE_FOLDER = os.path.join(base_folder_path, "merged_images")
        self.COMPRESSED_IMAGE_FOLDER = os.path.join(base_folder_path, "compressed_images")
        self.SELECT_NUM = 5
        self.saver = Saver(base_folder_path)
        self.max_concurrent = 20

    def convert_judgement(self, judgement):
        if judgement == -1.0:
            return "bad"
        elif judgement == 1.0:
            return "good"
        return None

    def load_json_data(self, file_path):
        try:
            full_path = os.path.join(self.BASE_FOLDER_PATH, file_path)
            full_path = os.path.normpath(full_path)
            print(f"successful read {full_path}")
            with open(full_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"File not found: {full_path}")
            return None
        except json.JSONDecodeError:
            print(f"Invalid JSON in file: {full_path}")
            return None
        except Exception as e:
            print(f"Error loading file {full_path}: {str(e)}")
            return None

    def process_original(self):
        original_data = self.load_json_data(self.ORIGINAL_PATH)
        if original_data is None:
            print(f"Error: Unable to load data from {self.ORIGINAL_PATH}")
            print("Please ensure the file exists and is in the correct location.")
            print(f"Current working directory: {os.getcwd()}")
            return

        filtered_data = [obj for obj in original_data if obj["judgement"] in [-1.0, 1.0]]

        if len(filtered_data) > self.SELECT_NUM:
            filtered_data = random.sample(filtered_data, self.SELECT_NUM)

        processed_data = []
        for idx, obj in enumerate(filtered_data, start=1):
            judgement_str = self.convert_judgement(obj["judgement"])
            if judgement_str is None:
                continue
            processed_obj = {
                "id": idx,
                "action": obj["action"],
                "judgement": judgement_str,
                "dataset": obj["dataset"],
                "category": obj["category"]
            }
            processed_data.append(processed_obj)
        self.saver.save_data(data=processed_data, target_file_path=self.PROCESSED_PATH)

    async def process_action_item_async(self, item):
        action = item.get('action', '')
        judgement = item.get('judgement', '')
        prompt = f'''Please expand this sentence into a complete story: {action}. Ensure that the nature of the event remains: {judgement}. Keep the output within 20 words. Provide no explanatory information, and output only the complete story.'''

        case_result = await call_gpt4o_api(prompt)
        print(case_result)

        if case_result:
            item['story'] = case_result
        return item

    async def expand_full_story(self):
        json_data = self.load_json_data(self.PROCESSED_PATH)
        tasks = []
        for item in json_data:
            tasks.append(self.process_action_item_async(item))

        updated_data = await asyncio.gather(*tasks, return_exceptions=True)
        self.saver.save_data(data=updated_data, target_file_path=self.PROCESSED_PATH)
        print("Full story done!")

    async def generate_image_description_async(self, item):
        action = item.get('action', '')
        story = item.get('story', '')
        prompt_narrative = f'''You will be provided with an original story. Your task is to generate a final narrative that divides the story into 2-5 distinct scenes, using placeholders like "fig1", "fig2", etc., to represent each image in the story. When describing these scenes, don't reveal any specific details of the scenes or images.

**Be careful not to add or subtract from the original story**—keep the original narrative structure intact while only replacing specific details with placeholders.

Here is the original story: {story}
The target action in the story is: {action}'''

        narrative = await call_gpt4o_api(prompt_narrative)
        if narrative:
            print(narrative)
            item['narrative'] = narrative

            prompt_fill = f'''Given the original text below and the accompanying story narration format, please complete the description of each figure based on the content of the original text.
Here is the original text: {story}
Here is the story narration: {narrative}'''

            image_description = await call_gpt4o_api(prompt_fill)
            if image_description:
                print(image_description)
                item['image_description'] = image_description

        return item

    async def generate_text(self):
        json_data = self.load_json_data(self.PROCESSED_PATH)
        tasks = []
        for item in json_data:
            tasks.append(self.generate_image_description_async(item))

        updated_data = await asyncio.gather(*tasks, return_exceptions=True)

        print("\nChecking generated descriptions:")
        for item in updated_data:
            if isinstance(item, Exception):
                print(f"Error in item: {str(item)}")
                continue
            print(f"\nItem {item.get('id', 'unknown')}:")
            print("image_description:", item.get('image_description', 'None'))
            print("narrative:", item.get('narrative', 'None'))

        self.saver.save_data(data=updated_data, target_file_path=self.PROCESSED_PATH)
        print("Narrative and image description done!")

    def split_images(self):
        json_data = self.load_json_data(self.PROCESSED_PATH)
        updated_data = []

        fig_pattern = r'fig(\d+)(?=:|])'

        desc_pattern = r'\[fig\d+:\s*(.*?)\]'

        scene_pattern = r'\*\*Scene \d+:.*?\*\*(.*?)(?=\*\*Scene|$)'

        print("\nStarting image splitting process...")
        for item in json_data:
            print(f"\nProcessing item {item['id']}:")
            img_description = item.get("image_description", "")

            if img_description:

                fig_matches = re.findall(fig_pattern, img_description, re.DOTALL)
                desc_matches = re.findall(desc_pattern, img_description, re.DOTALL)
                scene_matches = re.findall(scene_pattern, img_description, re.DOTALL)


                if desc_matches:
                    print(f"Found {len(desc_matches)} detailed descriptions")
                    for i, desc in enumerate(desc_matches, 1):
                        key = f"img{i}"
                        item[key] = desc.strip()
                        print(f"Added {key}: {desc[:100]}...")


                elif scene_matches:
                    print(f"Found {len(scene_matches)} scene descriptions")
                    for i, scene in enumerate(scene_matches, 1):
                        key = f"img{i}"
                        item[key] = scene.strip()
                        print(f"Added {key}: {scene[:100]}...")


                elif fig_matches:
                    print(f"Found {len(fig_matches)} fig markers")

                    for fig_num in fig_matches:

                        pattern = rf'fig{fig_num}[:\s]+([^*\[]+)'
                        text_match = re.search(pattern, img_description)
                        if text_match:
                            key = f"img{fig_num}"
                            item[key] = text_match.group(1).strip()
                            print(f"Added {key}: {item[key][:100]}...")

            updated_data.append(item)

        print("\nSaving updated data...")
        self.saver.save_data(data=updated_data, target_file_path=self.PROCESSED_PATH)
        print("Image splitting completed")

    async def generate_images(self):

        json_data = self.load_json_data(self.PROCESSED_PATH)


        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=20) as executor:
            tasks = []


            for item in json_data:
                img_id = item.get('id', 'unknown')
                print(f"\nProcessing item {img_id}:")


                img_keys = [k for k in item.keys() if k.startswith('img')]
                for img_key in sorted(img_keys):
                    img_prompt = item.get(img_key, '')
                    if img_prompt:
                        print(f"Creating task for {img_id}_{img_key}")
                        print(f"Prompt: {img_prompt[:100]}...")

                        self.saver.ensure_directory_exists(self.IMAGE_SAVE_FOLDER)
                        task = loop.run_in_executor(
                            executor, 
                            generate_and_save_image,
                            img_prompt,
                            f"{img_id}_{img_key}",
                            self.IMAGE_SAVE_FOLDER
                        )
                        tasks.append((img_id, img_key, task))

            if tasks:
                print(f"\nExecuting {len(tasks)} image generation tasks...")
                for img_id, img_key, task in tasks:
                    try:
                        result = await task
                        print(f"Task completed for {img_id}_{img_key}: {result}")
                    except Exception as e:
                        print(f"Error generating image {img_id}_{img_key}: {str(e)}")
            else:
                print("No image generation tasks were created!")

        print("All images have been generated and saved locally.")

    def compress_image(self, input_path, output_path, quality=85, max_size_kb=100):
        with Image.open(input_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            output_jpeg_path = os.path.splitext(output_path)[0] + ".jpg"
            img.save(output_jpeg_path, "JPEG", quality=quality)

        while os.path.getsize(output_jpeg_path) > max_size_kb * 1024:
            quality -= 5
            if quality < 5:
                break
            with Image.open(input_path) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(output_jpeg_path, "JPEG", quality=quality)

    def compress_images_in_folder(self, folder_path, output_folder, quality=85, max_size_kb=100):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.png'):
                input_path = os.path.join(folder_path, filename)
                output_path = os.path.join(output_folder, filename)
                self.compress_image(input_path, output_path, quality=quality, max_size_kb=max_size_kb)

    def merge_images(self, image_paths, output_path):
        valid_paths = [path for path in image_paths if os.path.exists(path)]

        if not valid_paths:
            print(f"Warning: No valid images found to merge for {output_path}")
            return False

        try:
            images = []
            for path in valid_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"Error loading image {path}: {str(e)}")
                    continue

            if not images:
                print(f"Warning: No valid images could be loaded for {output_path}")
                return False


            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)


            new_im = Image.new('RGB', (total_width, max_height))
            x_offset = 0


            for im in images:
                new_im.paste(im, (x_offset, 0))
                x_offset += im.size[0]


            new_im.save(output_path)
            print(f"Successfully merged images and saved to {output_path}")
            return True

        except Exception as e:
            print(f"Error merging images for {output_path}: {str(e)}")
            return False

    def merge_and_compress(self):
        json_data = self.load_json_data(self.PROCESSED_PATH)

        if not os.path.exists(self.MERGED_IMAGE_FOLDER):
            os.makedirs(self.MERGED_IMAGE_FOLDER)

        for item in json_data:
            img_keys = [key for key in item.keys() if key.startswith('img') and item[key]]
            img_files = [f"{item['id']}_{key}.png" for key in img_keys]
            img_paths = [os.path.join(self.IMAGE_SAVE_FOLDER, img_file) for img_file in img_files]


            valid_paths = [path for path in img_paths if os.path.exists(path)]
            if not valid_paths:
                print(f"Skipping item {item['id']} - no valid images found")
                continue

            output_path = os.path.join(self.MERGED_IMAGE_FOLDER, f"{item['id']}.png")
            if self.merge_images(valid_paths, output_path):
                item['merged_image'] = f"{item['id']}.png"
                print(f"Successfully processed item {item['id']}")
            else:
                print(f"Failed to process item {item['id']}")

        self.saver.save_data(data=json_data, target_file_path=self.PROCESSED_PATH)


        if os.path.exists(self.MERGED_IMAGE_FOLDER) and os.listdir(self.MERGED_IMAGE_FOLDER):
            self.compress_images_in_folder(
                self.MERGED_IMAGE_FOLDER, 
                self.COMPRESSED_IMAGE_FOLDER, 
                quality=95, 
                max_size_kb=300
            )
            print("Completed image compression")
        else:
            print("No images found for compression")

    async def process_one_task_async(self, item):
        action = item.get('action', '')
        story = item.get('story', '')
        prompt = f'''Please write a question asking if the person's behavior in the story below is good or bad. The question should provide two options: A for good and B for bad.

Here is the action: {action}
Here is the story: {story}
Now, please generate the question without any declarative statements.'''

        task_question = await call_gpt4o_api(prompt)
        print(task_question)

        if task_question:
            item['narrative'] = item.get('narrative', '') + " " + task_question
        return item

    async def generate_task_question(self):
        json_data = self.load_json_data(self.PROCESSED_PATH)
        tasks = []
        for item in json_data:
            tasks.append(self.process_one_task_async(item))

        updated_data = await asyncio.gather(*tasks, return_exceptions=True)


        for item in updated_data:
            if isinstance(item, dict): 

                if 'narrative' in item:
                    item['prompt'] = item['narrative']


                if 'merged_image' in item:

                    image_name = item['merged_image'].replace('.png', '.jpg')
                    item['image_path'] = f"compressed_images/{image_name}"

        self.saver.save_data(data=updated_data, target_file_path=self.PROCESSED_PATH)
        print("Final tasks done!")

    async def run(self):
        print("Starting pipeline execution...")

        # Step 1: Process original data
        print("\n1. Processing original data...")
        self.process_original()

        # Step 2: Expand full story
        print("\n2. Expanding full story...")
        await self.expand_full_story()

        # Step 3: Generate text
        print("\n3. Generating text...")
        await self.generate_text()

        # Step 4: Split images
        print("\n4. Splitting images...")
        self.split_images()

        # Step 5: Generate images
        print("\n5. Generating images...")
        await self.generate_images()

        # Step 6: Merge and compress images
        print("\n6. Merging and compressing images...")
        try:
            self.merge_and_compress()
        except Exception as e:
            print(f"Error in merge_and_compress: {str(e)}")
            print("Continuing with the pipeline...")

        # Step 7: Generate task questions
        print("\n7. Generating task questions...")
        await self.generate_task_question()

        print("\nPipeline execution completed!")

async def pipeline(base_dir="data"):
    try:
        processor = DataProcessor(base_dir)
        await processor.run()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_folder_path = sys.argv[1]
        asyncio.run(pipeline(base_folder_path))
    else:
        asyncio.run(pipeline())