import os
from tempfile import NamedTemporaryFile

import torch
from PIL import Image

metrics_dict = {
    "ClipScore"       : "ClipScore",
    "VQAScore"        : "VQAScore",
    "PickScore"       : "PickScore",
    "ImageRewardScore": "ImageRewardScore",
    "ProgrammaticDSGTIFAScore": "ProgrammaticDSGTIFAScore"
}


class Text2ImageEvalMetric():
    def __init__(
            self,
            are_metrics_preloaded: bool = True,
            selected_matrics: list = None,
            device: str = "cuda",
    ):
        self.device = device
        if selected_matrics is None:
            selected_matrics = list(metrics_dict.keys())  # use all the supported matrics

        # load all the metric class, might cause much more memory usage
        if are_metrics_preloaded is True:
            self.metrics = {}
            for metric_name in selected_matrics:
                self.metrics[metric_name] = eval(metrics_dict[metric_name])(device=self.device)
        else:
            self.metrics = {}
            for metric_name in selected_matrics:
                self.metrics[metric_name] = eval(metrics_dict[metric_name])(device=self.device)
            for metric_name in selected_matrics:
                self.metrics[metric_name] = metrics_dict[metric_name]
        print(f"use {self.metrics.keys()}, Are metrics preloaded?: {are_metrics_preloaded}")

    def eval_with_metrics(self, output: Image.Image, gen_data: dict):
        image = output.convert("RGB")
        results = {}
        for metric_name in self.metrics:
            if isinstance(self.metrics[metric_name], str):
                current_metric = eval(self.metrics[metric_name])(device=self.device)
                results[metric_name] = current_metric.compute(image, gen_data)
            else:
                results[metric_name] = self.metrics[metric_name].compute(image, gen_data)
        return results

    def list_metrics(self):
        return list(metrics_dict.keys())


class Metric:
    def __init__(self, device: str = "cuda"):
        self.device = device

    @torch.no_grad()
    def compute(self, prompt, image):
        "(Abstract method) abstract compute matric method"


class ClipScore(Metric):
    def __init__(self, model_name_or_path='openai:ViT-L-14-336', device="cuda"):
        super().__init__(device=device)
        if self.device.isdigit():
            original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device
            import t2v_metrics
            if original_cuda_visible_devices is None:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
        else:
            import t2v_metrics
        self.clipscore = t2v_metrics.CLIPScore(model='openai:ViT-L-14-336')

    def compute(self, image: Image.Image, gen_data: dict):
        prompt = gen_data['prompt']
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        # Calculate the score using the temporary file path
        score = self.clipscore(images=[temp_file_path], texts=[prompt]).item()

        return score


class VQAScore(Metric):
    def __init__(self, model="clip-flant5-xxl", device="cuda"):
        super().__init__(device=device)
        if self.device.isdigit():
            original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device
            import t2v_metrics
            if original_cuda_visible_devices is None:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
        else:
            import t2v_metrics
        self.clip_flant5_score = t2v_metrics.VQAScore(model=model)

    def compute(self, image: Image.Image, gen_data: dict):
        prompt = gen_data['prompt']
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        # Calculate the score using the temporary file path
        score = self.clip_flant5_score(images=[temp_file_path], texts=[prompt]).item()
        return score


class PickScore(Metric):
    def __init__(self, model='pickscore-v1', device="cuda"):
        super().__init__(device=device)
        if self.device.isdigit():
            original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device
            import t2v_metrics
            if original_cuda_visible_devices is None:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
        else:
            import t2v_metrics
        self.pick_score = t2v_metrics.CLIPScore(model=model)

    def compute(self, image: Image.Image, gen_data: dict):
        prompt = gen_data['prompt']
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        # Calculate the score using the temporary file path
        score = self.pick_score(images=[temp_file_path], texts=[prompt]).item()
        return score


class ImageRewardScore(Metric):
    def __init__(self, model='image-reward-v1', device="cuda"):
        super().__init__(device=device)
        if self.device.isdigit():
            original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device
            import t2v_metrics
            if original_cuda_visible_devices is None:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
        else:
            import t2v_metrics
        self.image_reward_score = t2v_metrics.ITMScore(model=model)

    def compute(self, image: Image.Image, gen_data: dict):
        prompt = gen_data['prompt']
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        # Calculate the score using the temporary file path
        score = self.image_reward_score(images=[temp_file_path], texts=[prompt]).item()
        return score


class ProgrammaticDSGTIFAScore(Metric):
    def __init__(self, model='Phi-3-vision-128k-instruct', device="cuda"):
        super().__init__(device=device)
        from ..qa_model.imageqa_model import ImageQAModel
        from ..qa_model.prompt import succinct_prompt
        self.vqa_model = ImageQAModel(model_name=model, torch_device = self.device, prompt_name="succinct", prompt_func=succinct_prompt)
        
    @ staticmethod
    def _get_dsg_questions(dsg):
        dsg_questions = {}

        for node in dsg.nodes(data=True):
            node_id, node_data = node
            node_type = node_data['type']
            node_value = node_data['value']

            if node_type == 'object_node':
                dsg_questions[f"{node_id}:{node_value}"] = {}
                # preposition
                dsg_questions[f"{node_id}:{node_value}"]['question'] = f"Is there a {node_value}?"
                dsg_questions[f"{node_id}:{node_value}"]['dependency'] = []
                for neighbor_id in dsg.neighbors(node_id):
                    neighbor_data = dsg.nodes[neighbor_id]
                    neighbor_type = neighbor_data['type']
                    neighbor_value = neighbor_data['value']
                    if neighbor_type == 'attribute_node':
                        # combine color
                        dsg_questions[f"{neighbor_id}:{neighbor_value}"]={}
                        dsg_questions[f"{neighbor_id}:{neighbor_value}"]['question']= f"Is the {node_value} {neighbor_value}?"
                        dsg_questions[f"{neighbor_id}:{neighbor_value}"]['dependency']= [f"{node_id}:{node_value}"]

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

    @ staticmethod
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
    
    @ staticmethod
    def _compute_score_without_dependencies(dsg_questions):
        cnt, tnt = 0, 0.0
        for element in dsg_questions:
            cnt += 1
            if dsg_questions[element]['result'] is True:
                tnt += 1
        return tnt / cnt if cnt > 0 else 0.0
        
    def compute(self, image: Image.Image, gen_data: dict):
        from ...text2vision.prompt_generator import convert_json_to_sg
        # the scene graph in gen_data is json format, we need to convert it to nx.DiGraph format here
        scene_graph = convert_json_to_sg(gen_data['scene_graph'])
        dsg_questions = self._get_dsg_questions(scene_graph)
        
        for element in dsg_questions:
            prompt = "Based on the image, answer: " + dsg_questions[element]['question'] + ". Only output yes or no"
            print(prompt)
            model_answer  = self.vqa_model.qa(image, prompt)
            print(model_answer)
            dsg_questions[element]['result'] = "yes" in model_answer.lower()

        score_without_dependencies = self._compute_score_without_dependencies(dsg_questions)
        score_with_dependencies = self._compute_score_with_dependencies(dsg_questions)

        # return the weighted score
        return 0.5 * score_with_dependencies + 0.5 * score_without_dependencies

