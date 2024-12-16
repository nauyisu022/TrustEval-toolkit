import torch
from PIL import Image
from tempfile import NamedTemporaryFile
import os

metrics_dict = {
    "ClipScore": "ClipScore",
    "VQAScore": "VQAScore",
    "PickScore": "PickScore",
    "ImageRewardScore": "ImageRewardScore"
}

class Textto3DEvalMetric():
    def __init__(
        self,
        are_metrics_preloaded: bool = True,
        selected_matrics: list = None,
        device: str = "cuda",
    ):
        self.device = device
        if selected_matrics is None:
            selected_matrics = list(metrics_dict.keys()) # use all the supported metrics
        
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


    def _decode_video(self, video_data):
        import imageio
        import numpy as np
        import io
        with io.BytesIO(video_data) as video_bytes:
            video_bytes.seek(0)
            reader = imageio.get_reader(video_bytes, format='mp4')
            frames = []
            for frame in reader:
                frame = Image.fromarray(frame)
                frames.append(frame)
            reader.close()
        return frames


    def eval_with_metrics(self, prompt: str, video: bytes):
        frames = self._decode_video(video)
        results = {}
        for metric_name in self.metrics:
            if isinstance(self.metrics[metric_name], str):
                current_metric = eval(self.metrics[metric_name])(device=self.device)
                scores = [current_metric.compute(prompt, frame) for frame in frames]
                results[metric_name] = sum(scores) / len(scores)
            else:
                scores = [self.metrics[metric_name].compute(prompt, frame) for frame in frames]
                results[metric_name] = sum(scores) / len(scores)
        return results

    def list_metrics(self):
        return list(metrics_dict.keys())

class Metric:
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    @torch.no_grad()
    def compute(self, prompt, image):
        "(Abstract method) abstract compute metric method"
    
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

    def compute(self, prompt, image: Image.Image):
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name
        
        # Calculate the score using the temporary file path
        score = self.clipscore(images=[temp_file_path], texts=[prompt]).item()

        os.remove(temp_file_path)
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
        
    def compute(self, prompt: str, image: Image.Image):
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name
        
        # Calculate the score using the temporary file path
        score = self.clip_flant5_score(images=[temp_file_path], texts=[prompt]).item()
        
        os.remove(temp_file_path)
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
        
    def compute(self, prompt: str, image: Image.Image):
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name
        
        # Calculate the score using the temporary file path
        score = self.pick_score(images=[temp_file_path], texts=[prompt]).item()
        
        os.remove(temp_file_path)
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
        
    def compute(self, prompt: str, image: Image.Image):
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name
        
        # Calculate the score using the temporary file path
        score = self.image_reward_score(images=[temp_file_path], texts=[prompt]).item()
        
        os.remove(temp_file_path)
        return score
