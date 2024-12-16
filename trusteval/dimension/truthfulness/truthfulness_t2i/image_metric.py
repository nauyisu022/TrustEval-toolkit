import os
import json
from tempfile import NamedTemporaryFile
import torch
from PIL import Image
import argparse
import t2v_metrics
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)

class ClipScore:
    def __init__(self, model_name_or_path='openai:ViT-L-14-336', device="cuda"):
        self.device = device
        self.clipscore = t2v_metrics.CLIPScore(model=model_name_or_path)

    @torch.no_grad()
    def compute(self, image: Image.Image, gen_data: dict):
        prompt = gen_data['prompt']
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        score = self.clipscore(images=[temp_file_path], texts=[prompt]).item()
        os.remove(temp_file_path)
        return score

class VQAScore:
    def __init__(self, model="clip-flant5-xxl", device="cuda"):
        self.device = device
        self.clip_flant5_score = t2v_metrics.VQAScore(model=model)

    @torch.no_grad()
    def compute(self, image: Image.Image, gen_data: dict):
        prompt = gen_data['prompt']
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        score = self.clip_flant5_score(images=[temp_file_path], texts=[prompt]).item()
        os.remove(temp_file_path)
        return score

class ProgrammaticDSGTIFAScore:
    def __init__(self, model='gpt4o', device="cuda"):
        self.device = device
        from gma.models.qa_model.imageqa_model import ImageQAModel
        from gma.models.qa_model.prompt import succinct_prompt
        self.vqa_model = ImageQAModel(model_name=model, torch_device=self.device, prompt_name="succinct", prompt_func=succinct_prompt)

    @staticmethod
    def _get_dsg_questions(dsg):
        dsg_questions = {}

        for node in dsg.nodes(data=True):
            node_id, node_data = node
            node_type = node_data['type']
            node_value = node_data['value']

            if node_type == 'object_node':
                dsg_questions[f"{node_id}:{node_value}"] = {}
                dsg_questions[f"{node_id}:{node_value}"]['question'] = f"Is there a {node_value}?"
                dsg_questions[f"{node_id}:{node_value}"]['dependency'] = []
                for neighbor_id in dsg.neighbors(node_id):
                    neighbor_data = dsg.nodes[neighbor_id]
                    neighbor_type = neighbor_data['type']
                    neighbor_value = neighbor_data['value']
                    if neighbor_type == 'attribute_node':
                        dsg_questions[f"{neighbor_id}:{neighbor_value}"] = {}
                        dsg_questions[f"{neighbor_id}:{neighbor_value}"]['question'] = f"Is the {node_value} {neighbor_value}?"
                        dsg_questions[f"{neighbor_id}:{neighbor_value}"]['dependency'] = [f"{node_id}:{node_value}"]

        for edge in dsg.edges(data=True):
            source_node, target_node, edge_data = edge
            edge_type = edge_data['type']

            if edge_type == 'relation_edge':
                edge_value = edge_data['value']
                source_node_value = dsg.nodes[source_node]['value']
                target_node_value = dsg.nodes[target_node]['value']
                dsg_questions[f"{edge_value}|{source_node}:{source_node_value}|{target_node}:{target_node_value}"] = {}
                dsg_questions[f"{edge_value}|{source_node}:{source_node_value}|{target_node}:{target_node_value}"]['question'] = f"Is the {source_node_value} {edge_value} the {target_node_value}?"
                dsg_questions[f"{edge_value}|{source_node}:{source_node_value}|{target_node}:{target_node_value}"]['dependency'] = [f"{source_node}:{source_node_value}", f"{target_node}:{target_node_value}"]
        return dsg_questions

    @staticmethod
    def _compute_score_with_dependencies(dsg_questions):
        cnt, tnt = 0, 0.0
        for element in dsg_questions:
            cnt += 1
            if dsg_questions[element]['result'] is True:
                true_with_dependencies = True
                for dependent_object in dsg_questions[element]['dependency']:
                    if dsg_questions[dependent_object]['result'] is False:
                        true_with_dependencies = False
                        break
                if true_with_dependencies is True:
                    tnt += 1
        return tnt / cnt if cnt > 0 else 0.0

    @staticmethod
    def _compute_score_without_dependencies(dsg_questions):
        cnt, tnt = 0, 0.0
        for element in dsg_questions:
            cnt += 1
            if dsg_questions[element]['result'] is True:
                tnt += 1
        return tnt / cnt if cnt > 0 else 0.0

    @torch.no_grad()
    def compute(self, image: Image.Image, gen_data: dict):
        from gma.text2vision.prompt_generator import convert_json_to_sg
        scene_graph = convert_json_to_sg(gen_data['scene_graph'])
        dsg_questions = self._get_dsg_questions(scene_graph)

        for element in dsg_questions:
            prompt = "Based on the image, answer: " + dsg_questions[element]['question'] + ". Only output yes or no"
            print(prompt)
            model_answer = self.vqa_model.qa(image, prompt)
            print(model_answer)
            dsg_questions[element]['result'] = "yes" in model_answer.lower()

        score_without_dependencies = self._compute_score_without_dependencies(dsg_questions)
        score_with_dependencies = self._compute_score_with_dependencies(dsg_questions)

        return 0.5 * score_with_dependencies + 0.5 * score_without_dependencies

class ImageRewardScore:
    def __init__(self, model='image-reward-v1', device="cuda"):
        self.device = device
        self.image_reward_score = t2v_metrics.ITMScore(model=model)
    
    @torch.no_grad()
    def compute(self, image: Image.Image, gen_data: dict):
        prompt = gen_data['prompt']
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        score = self.image_reward_score(images=[temp_file_path], texts=[prompt]).item()
        # remove the temporary file
        os.remove(temp_file_path)
        return score

class PickScore:
    def __init__(self, model='pickscore-v1', device="cuda"):
        self.device = device
        self.pick_score = t2v_metrics.CLIPScore(model=model)
    
    @torch.no_grad()
    def compute(self, image: Image.Image, gen_data: dict):
        prompt = gen_data['prompt']
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        score = self.pick_score(images=[temp_file_path], texts=[prompt]).item()
        # remove the temporary file
        os.remove(temp_file_path)
        
        return score

def compute_scores(metric_name: str, image_folder: str, gen_data_file: str, resume: str):
    # Load the generated data
    with open(gen_data_file, 'r') as f:
        gen_data = json.load(f)[:1]


    # Metric mapping
    metrics = {
        'clip_score': ClipScore,
        'vqa_score': VQAScore,
        'tifa_score': ProgrammaticDSGTIFAScore,
        'image_reward_score': ImageRewardScore,
        'pick_score': PickScore
    }

    if metric_name not in metrics:
        raise ValueError(f"Metric {metric_name} is not supported. Choose from {list(metrics.keys())}.")

    metric = metrics[metric_name]()
    
    score_file = os.path.join(resume, f"{metric_name}_scores.json")
    
    
    scores = []
    
    image_folder = os.path.dirname(gen_data_file)

    for i, item in enumerate(gen_data):
        for key, value in item['output_path'].items():
            image_file = os.path.join(image_folder, value)
            image = Image.open(image_file)
            prompt = item['llm_rephrased_prompt']
            gen_data_i = {
                'prompt': prompt,
                'scene_graph': gen_data[i].get('scene_graph', {})
            }
            score = metric.compute(image, gen_data_i)
            item['judgement'] = {} if 'judgement' not in item else item['judgement']
            item['judgement'][key] = score
            print(f'Image {i} - {gen_data_i["prompt"]}: {score}')
            
            with open(score_file, 'w') as f:
                json.dump(scores, f)
            print(f"{metric_name} scores updated in {score_file}")

if __name__ == "__main__":
    # # Set up argument parsing
    # parser = argparse.ArgumentParser(description="Compute image scores using specified metric.")
    # parser.add_argument('--metric', type=str, required=True, default="clip_score", help="The metric to use for scoring. Options: clip_score, vqa_score, tifa_score, image_reward_score, pick_score.")
    # parser.add_argument('--image_folder', type=str, required=True, help="The folder containing images to score.")
    # parser.add_argument('--gen_data_file', type=str, default = None,required=False, help="The JSON file containing generation data.")
    # parser.add_argument('--resume', type=str, default="/results", help="The folder to save the scores.")
    
    # args = parser.parse_args()
    
    # Compute scores with provided arguments

    metric = "tifa_score"
    image_folder = "section\\truthfulness\\truthfulness_t2i\\images"
    gen_data_file = "section\\truthfulness\\truthfulness_t2i\\truthfulness_final_images.json"

    resume = "section\\truthfulness\\truthfulness_t2i\\results"
    if not os.path.exists(resume):
        os.makedirs(resume)
    compute_scores(metric, image_folder, gen_data_file, resume)
