import os
import pandas as pd
import random
from .utils import FairnessSaver,get_api_res


class StereotypeGenerator:
    def __init__(self, base_dir) -> None:
        self.saver = FairnessSaver()
        self.base_dir = base_dir

        self.dataset_path = os.path.join(base_dir,'metadata')
        self.dataset_processed_path = os.path.join(self.dataset_path, 'stereotype_data', 'processed')
        self.select_dataset_path = os.path.join(self.dataset_path, 'stereotype_data', 'select')
        self.cases_output_path = os.path.join(self.dataset_path, 'stereotype_data', 'cases')
        self.merged_output_file = os.path.join(base_dir, 'final', 'stereotype.json')

        self._ensure_directories()

    def _ensure_directories(self):
        directories = [
            self.dataset_path,
            self.dataset_processed_path,
            self.select_dataset_path,
            self.cases_output_path,
            os.path.dirname(self.merged_output_file)
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def process_all_datasets(self):
        def process_crows(input_file, output_file):
            df = pd.read_csv(input_file)

            output = []
            for index, row in df.iterrows():
                if row['stereo_antistereo'] == 'stereo':
                    stereo_text = row['sent_more']
                    antistereo_text = row['sent_less']
                else:
                    stereo_text = row['sent_less']
                    antistereo_text = row['sent_more']

                json_object = {
                    "id": index + 1,
                    "stereo_text": stereo_text,
                    "antistereo_text": antistereo_text,
                    "bias_type": row['bias_type'],
                    "data_source": "crows_pairs"
                }

                output.append(json_object)

            self.saver.save_to_json(output, output_file)
            print(f"{output_file} have successful generate.")

        def process_stereoset(input_file, output_file):
            data = self.saver.load_json_data(input_file)

            new_data = []
            unique_id = 1

            for obj in data:
                context = obj.get("context", "")
                target = obj.get("target", "")
                bias_type = obj.get("bias_type", "")
                sentences = obj.get("sentences", {})
                sentence_list = sentences.get("sentence", [])
                gold_label_list = sentences.get("gold_label", [])

                stereo_text = ""
                antistereo_text = ""
                unrelated_text = ""

                for sentence, label in zip(sentence_list, gold_label_list):
                    if label == 1:
                        stereo_text = sentence
                    elif label == 0:
                        antistereo_text = sentence
                    elif label == 2:
                        unrelated_text = sentence

                new_item = {
                    "id": unique_id,
                    "text": context,
                    "stereo_text": stereo_text,
                    "antistereo_text": antistereo_text,
                    "unrelated_text": unrelated_text,
                    "target": target,
                    "bias_type": bias_type,
                    "data_source": "stereoset"
                }

                new_data.append(new_item)
                unique_id += 1

            self.saver.save_to_json(new_data, output_file)
            print(f"{output_file} have successful generate.")

        def process_bbq(input_file, output_file):
            processed_data = []
            id_counter = 1

            data = self.saver.load_jsonl_data(input_file)
            for line in data:
                new_entry = {
                    "id": id_counter,
                    "category": line["category"],
                    "target": ', '.join(line["additional_metadata"]["stereotyped_groups"]),
                    "context": line["context"],
                    "question": line["question"],
                    "ans_0": line["ans0"],
                    "ans_1": line["ans1"],
                    "ans_2": line["ans2"],
                    "label": line["label"],
                    "data_source": "BBQ"
                }
                processed_data.append(new_entry)
                id_counter += 1

            self.saver.save_to_json(processed_data, output_file)
            print(f"{output_file} have successful generate.")

        self.saver.create_output_directory(self.dataset_processed_path)

        crows_input = os.path.join(self.dataset_path, 'crows_pairs_anonymized.csv')
        crows_output = os.path.join(self.dataset_processed_path, 'crows.json')
        process_crows(crows_input, crows_output)

        stereoset_inputs = [
            ('stereoset_fill.json', 'stereoset_fill.json'),
            ('stereoset_complete.json', 'stereoset_complete.json')
        ]
        for input_file, output_file in stereoset_inputs:
            process_stereoset(
                os.path.join(self.dataset_path, input_file),
                os.path.join(self.dataset_processed_path, output_file)
            )

        bbq_input = os.path.join(self.dataset_path, 'bbq.jsonl')
        bbq_output = os.path.join(self.dataset_processed_path, 'bbq.json')
        process_bbq(bbq_input, bbq_output)

    def sample_all_datasets(self):
        def random_sample(input_file, output_file, sample_size):
            data = self.saver.load_json_data(input_file)
            sampled_data = random.sample(data, sample_size)
            self.saver.save_to_json(sampled_data, output_file)
            print(f"{output_file} have successful generate. {sample_size} samples")

        self.saver.create_output_directory(self.select_dataset_path)

        samples = [
            ('crows.json', 'crows_sample.json', 40),
            ('bbq.json', 'bbq_sample.json', 40),
            ('stereoset_fill.json', 'stereoset_fill_sample.json', 30),
            ('stereoset_complete.json', 'stereoset_complete_sample.json', 30),
        ]
        for input_file, output_file, sample_size in samples:
            random_sample(
                os.path.join(self.dataset_processed_path, input_file),
                os.path.join(self.select_dataset_path, output_file),
                sample_size
            )

    def test_case_builder(self):
        # Crows
        def generate_crows_cases():
            json_file = os.path.join(self.select_dataset_path, 'crows_sample.json')
            json_data = self.saver.load_json_data(json_file)

            updated_data = []
            for item in json_data:
                prompt = f"Prompt using stereo_text and antistereo_text from {item}"
                case_result = get_api_res(prompt)
                if case_result:
                    item['case'] = case_result
                updated_data.append(item)

            self.saver.save_to_json(updated_data, os.path.join(self.cases_output_path, 'crows_cases.json'))
            print(f"Crows cases have been generated.")

        # BBQ
        def generate_bbq_cases():
            json_file = os.path.join(self.select_dataset_path, 'bbq_sample.json')
            json_data = self.saver.load_json_data(json_file)

            updated_data = []
            for item in json_data:
                prompt = f"Prompt using context and question from {item}"
                case_result = get_api_res(prompt)
                if case_result:
                    item['case'] = case_result
                updated_data.append(item)

            self.saver.save_to_json(updated_data, os.path.join(self.cases_output_path, 'bbq_cases.json'))
            print(f"BBQ cases have been generated.")

        self.saver.create_output_directory(self.cases_output_path)
        generate_crows_cases()
        generate_bbq_cases()


    def run(self,):
        print("Step 1: Processing all original datasets...")
        self.process_all_datasets()

        print("Step 2: Randomly sampling from processed datasets...")
        self.sample_all_datasets()

        print("Step 3: Generating cases...")
        self.test_case_builder()

if __name__ == "__main__":
    base_dir = "intermediate"
    SG=StereotypeGenerator(base_dir)
    SG.run()
