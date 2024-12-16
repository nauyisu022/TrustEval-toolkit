import torch
from PIL import Image
import tempfile
from tempfile import NamedTemporaryFile
import os

metrics_dict = {
    "ClipScore": "ClipScore",
    "VQAScore": "VQAScore",
    "PickScore": "PickScore",
    "ImageRewardScore": "ImageRewardScore",
    "VBenchMotionSmoothness": "VBenchMotionSmoothness",
    "VBenchDynamicDegree": "VBenchDynamicDegree",
    "VBenchAestheticQuality": "VBenchAestheticQuality",
    "VBenchImagingQuality": "VBenchImagingQuality",
    "VBenchTemporalFlickering": "VBenchTemporalFlickering",
    "VBenchOverallConsistency": "VBenchOverallConsistency",
    "VBenchSubjectConsistency": "VBenchSubjectConsistency",
    "VBenchBackgroundConsistency": "VBenchBackgroundConsistency",
}



class Text2VideoEvalMetric():
    def __init__(
        self,
        are_metrics_preloaded: bool = False,
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
                if metric_name.startswith("VBench"):
                    results[metric_name] = current_metric.compute(prompt, video)
                else:
                    scores = [current_metric.compute(prompt, frame) for frame in frames]
                    results[metric_name] = sum(scores) / len(scores)
            else:
                if metric_name.startswith("VBench"):
                    results[metric_name] = current_metric.compute(prompt, video)
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
    def compute(self, prompt, video):
        "(Abstract method) abstract compute metric method"
    
    


class VBench_model(Metric):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        if self.device.isdigit(): 
            original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device
            from .model_zoo.VBench.vbench import VBench 
            if original_cuda_visible_devices is None:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
        else:
            from .model_zoo.VBench.vbench import VBench 
        import datetime
        # get current path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.vbench_model = VBench(device, os.path.join(current_dir, "model_zoo/VBench/vbench/VBench_full_info.json"), os.path.join(current_dir, "model_zoo/VBench/evaluation_results/"))
        # self.vbench_model = VBench(device, "./model_zoo/VBench/vbench/VBench_full_info.json", "./model_zoo/VBench/evaluation_results/")
        
    def compute(self, prompt, video):
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
            with open(tmp.name, 'wb') as file:
                file.write(data)
            video_path = tmp.name
            results = self.vbench_model.evaluate(
                videos_path = video_path,
                name = f'results_{datetime.now()}',
                prompt_list=[prompt], # pass in [] to read prompt from filename
                dimension_list = [],
                local=False,
                read_frame=False,
                mode="custom_input",
                imaging_quality_preprocessing_mode="longer",
            )
        return results
     

class VBenchMotionSmoothness(VBench_model):
    def __init__(self, device="cuda"):
        super().__init__(device=device)
    def compute(self, prompt, video):
        from datetime import datetime
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
            with open(tmp.name, 'wb') as file:
                file.write(video)
            video_path = tmp.name
            results = self.vbench_model.evaluate(
                videos_path = video_path,
                name = f'results_{datetime.now()}',
                prompt_list=[prompt], # pass in [] to read prompt from filename
                dimension_list = ["motion_smoothness"],
                local=False,
                read_frame=False,
                mode="custom_input",
                imaging_quality_preprocessing_mode="longer",
            )
        return results["motion_smoothness"][0]
    
class VBenchDynamicDegree(VBench_model):
    def __init__(self, device="cuda"):
        super().__init__(device=device)
    def compute(self, prompt, video):
        from datetime import datetime
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
            with open(tmp.name, 'wb') as file:
                file.write(video)
            video_path = tmp.name
            results = self.vbench_model.evaluate(
                videos_path = video_path,
                name = f'results_{datetime.now()}',
                prompt_list=[prompt], # pass in [] to read prompt from filename
                dimension_list = ["dynamic_degree"],
                local=False,
                read_frame=False,
                mode="custom_input",
                imaging_quality_preprocessing_mode="longer",
            )
        return results["dynamic_degree"][0]

class VBenchAestheticQuality(VBench_model):
    def __init__(self, device="cuda"):
        super().__init__(device=device)
    def compute(self, prompt, video):
        from datetime import datetime
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
            with open(tmp.name, 'wb') as file:
                file.write(video)
            video_path = tmp.name
            results = self.vbench_model.evaluate(
                videos_path = video_path,
                name = f'results_{datetime.now()}',
                prompt_list=[prompt], # pass in [] to read prompt from filename
                dimension_list = ["aesthetic_quality"],
                local=False,
                read_frame=False,
                mode="custom_input",
                imaging_quality_preprocessing_mode="longer",
            )
        return results["aesthetic_quality"][0]
    
class VBenchImagingQuality(VBench_model):
    def __init__(self, device="cuda"):
        super().__init__(device=device)
    def compute(self, prompt, video):
        from datetime import datetime
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
            with open(tmp.name, 'wb') as file:
                file.write(video)
            video_path = tmp.name
            results = self.vbench_model.evaluate(
                videos_path = video_path,
                name = f'results_{datetime.now()}',
                prompt_list=[prompt], # pass in [] to read prompt from filename
                dimension_list = ["imaging_quality"],
                local=False,
                read_frame=False,
                mode="custom_input",
                imaging_quality_preprocessing_mode="longer",
            )
        return results["imaging_quality"][0]


class VBenchTemporalFlickering(VBench_model):
    def __init__(self, device="cuda"):
        super().__init__(device=device)
    def compute(self, prompt, video):
        from datetime import datetime
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
            with open(tmp.name, 'wb') as file:
                file.write(video)
            video_path = tmp.name
            results = self.vbench_model.evaluate(
                videos_path = video_path,
                name = f'results_{datetime.now()}',
                prompt_list=[prompt], # pass in [] to read prompt from filename
                dimension_list = ["temporal_flickering"],
                local=False,
                read_frame=False,
                mode="custom_input",
                imaging_quality_preprocessing_mode="longer",
            )
        return results["temporal_flickering"][0]

class VBenchOverallConsistency(VBench_model):
    def __init__(self, device="cuda"):
        super().__init__(device=device)
    def compute(self, prompt, video):
        from datetime import datetime
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
            with open(tmp.name, 'wb') as file:
                file.write(video)
            video_path = tmp.name
            results = self.vbench_model.evaluate(
                videos_path = video_path,
                name = f'results_{datetime.now()}',
                prompt_list=[prompt], # pass in [] to read prompt from filename
                dimension_list = ["overall_consistency"],
                local=False,
                read_frame=False,
                mode="custom_input",
                imaging_quality_preprocessing_mode="longer",
            )
        return results["overall_consistency"][0]
    
class VBenchSubjectConsistency(VBench_model):
    def __init__(self, device="cuda"):
        super().__init__(device=device)
    def compute(self, prompt, video):
        from datetime import datetime
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
            with open(tmp.name, 'wb') as file:
                file.write(video)
            video_path = tmp.name
            results = self.vbench_model.evaluate(
                videos_path = video_path,
                name = f'results_{datetime.now()}',
                prompt_list=[prompt], # pass in [] to read prompt from filename
                dimension_list = ["subject_consistency"],
                local=False,
                read_frame=False,
                mode="custom_input",
                imaging_quality_preprocessing_mode="longer",
            )
        return results["subject_consistency"][0]

class VBenchBackgroundConsistency(VBench_model):
    def __init__(self, device="cuda"):
        super().__init__(device=device)
    def compute(self, prompt, video):
        from datetime import datetime
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
            with open(tmp.name, 'wb') as file:
                file.write(video)
            video_path = tmp.name
            results = self.vbench_model.evaluate(
                videos_path = video_path,
                name = f'results_{datetime.now()}',
                prompt_list=[prompt], # pass in [] to read prompt from filename
                dimension_list = ["background_consistency"],
                local=False,
                read_frame=False,
                mode="custom_input",
                imaging_quality_preprocessing_mode="longer",
            )
        return results["background_consistency"][0]

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
