import base64
import json
import os
import random
from typing import Union
import diffusers
import numpy as np
import torch
from PIL import Image
import io
import imageio 

import sys
import os
import torch
import sys
sys.path.append('t3d')
from launch import *


from .base_gen_model import GenModel, GenModelInstance
from .textto3d_metric import Textto3DEvalMetric


# Todo: Integrate threestudio into this file
textto3d_model = {
    "dreamfusion-sd": ("Dreamfusion_sd", "t3d/configs/dreamfusion-sd.yaml"),
    "dreamfusion-if": ("Dreamfusion_if", "t3d/configs/dreamfusion-if.yaml"),
    "prolificdreamer": (
        "Prolificdreamer",
        [
            "t3d/configs/prolificdreamer.yaml",
            "t3d/configs/prolificdreamer-geometry.yaml",
            "t3d/configs/prolificdreamer-texture.yaml",
        ],
    ),
    "magic3d-if": (
        "Magic3D",
        ["t3d/configs/magic3d-coarse-if.yaml", "t3d/configs/magic3d-refine-sd.yaml"],
    ),
    "magic3d-sd": (
        "Magic3D",
        ["t3d/configs/magic3d-coarse-sd.yaml", "t3d/configs/magic3d-refine-sd.yaml"],
    ),
    "sjc": ("SJC", "t3d/configs/sjc.yaml"),
    "latentnerf": (
        "LatentNeRF",
        ["t3d/configs/latentnerf.yaml", "t3d/configs/latentnerf-refine.yaml"],
    ),
    "fantasia3d": (
        "Fantasia3D",
        ["t3d/configs/fantasia3d.yaml", "t3d/configs/fantasia3d-texture.yaml"],
    ),
}


def set_model_key(model_name, key):
    textto3d_model[model_name] = (textto3d_model[model_name][0], key)

def list_textto3d_models():
    return list(textto3d_model.keys())


class Textto3DModel(GenModel):
    def __init__(
        self,
        model_name: str,
        model: GenModelInstance = None,
        metrics: list[str] = None,
        metrics_device: str = "cuda",
        precision: torch.dtype = torch.float16,# None means using all the default metrics
        torch_device: str = "cuda",
        cache_path: str = None,
    ):
        super().__init__(model_name, cache_path)
        if metrics is not None:
            self.metrics = Textto3DEvalMetric(selected_matrics=metrics, device=metrics_device)
        else: 
            self.metrics = "No Metrics"

        if isinstance(torch_device, str):
            torch_device = torch.device(torch_device)
        else:
            if torch_device == -1:
                torch_device = (
                    torch.device("cuda") if torch.cuda.is_available() else "cpu"
                )
            else:
                torch_device = torch.device(f"cuda:{torch_device}")
        if model is None:
            print(f"Loading {model_name} ...")
            class_name, ckpt = textto3d_model[model_name]
            self.model_presision = precision
            self.model = eval(class_name)(ckpt, precision, torch_device)
            print(f"Finish loading {model_name}")
        else:
            print("Using provided model...")
            self.model = model
            
    @torch.no_grad()
    def gen(self, prompt):
        output = self._gen(prompt)
        if self.metrics == "No Metrics":
            result = {"output": output}
        else:
            result = {"output": output, "metrics": self.metrics.eval_with_metrics(prompt, output)}
        return result
            
    def _data_to_str(self, data):
        if isinstance(data, str):
            return data
        else:
            raise ValueError("Invalid data type")


class Dreamfusion_sd(GenModelInstance):
    def __init__(self, ckpt: str = "t3d/configs/dreamfusion-sd.yaml", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        self.config_f = ckpt
        self.device = device
    def gen(self, prompt):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f
        input_ns.gpu = self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        img_list = main(input_ns, extras)[0]
        with io.BytesIO() as video_bytes:
            # Use imageio to write frames to a video
            writer = imageio.get_writer(video_bytes, format='mp4', fps=30)
            
            for frame in img_list:
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        return video_data
    
class Dreamfusion_if(GenModelInstance):
    def __init__(self, ckpt: str = "t3d/configs/dreamfusion-if.yaml", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        self.config_f = ckpt
        self.device = device
    def gen(self, prompt):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f
        input_ns.gpu = self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        img_list = main(input_ns, extras)[0]
        with io.BytesIO() as video_bytes:
            # Use imageio to write frames to a video
            writer = imageio.get_writer(video_bytes, format='mp4', fps=30)
            
            for frame in img_list:
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        return video_data
        # return result[0]

class Prolificdreamer(GenModelInstance):
    def __init__(self, ckpt: list = ["t3d/configs/prolificdreamer.yaml", "t3d/configs/prolificdreamer-geometry.yaml","t3d/configs/prolificdreamer-texture.yaml"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), geometry_refine = True, texturing = True):
        self.config_f = ckpt
        self.device = device
        self.geometry_refine = geometry_refine
        self.texturing = texturing
    def gen(self, prompt):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f[0]
        input_ns.gpu =self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        # pass kwargs to extras
            
        result = main(input_ns, extras)
        if self.geometry_refine:
            input_ns.config = self.config_f[1]
            extras.append(f"system.geometry_convert_from={result[1]}")
            # system.geometry_convert_override.isosurface_threshold=some_value 0-20.
            # extras.append(f"system.geometry_convert_override.isosurface_threshold={20.0}")
            result = main(input_ns, extras)
            
            if self.texturing:
                input_ns.config = self.config_f[2]
                extras.pop()
                extras.append(f"system.geometry_convert_from={result[1]}")
                result = main(input_ns, extras)
                img_list = result[0]
                with io.BytesIO() as video_bytes:
                    # Use imageio to write frames to a video
                    writer = imageio.get_writer(video_bytes, format='mp4', fps=30)
                    
                    for frame in img_list:
                        writer.append_data(frame)
                    
                    writer.close()
                    video_bytes.seek(0)
                    video_data = video_bytes.read()
                return video_data
            else:
                img_list = result[0]
                with io.BytesIO() as video_bytes:
                    # Use imageio to write frames to a video
                    writer = imageio.get_writer(video_bytes, format='mp4', fps=30)
                    
                    for frame in img_list:
                        writer.append_data(frame)
                    
                    writer.close()
                    video_bytes.seek(0)
                    video_data = video_bytes.read()
                return video_data
        else:
            img_list = result[0]
            with io.BytesIO() as video_bytes:
                # Use imageio to write frames to a video
                writer = imageio.get_writer(video_bytes, format='mp4', fps=30)
                
                for frame in img_list:
                    writer.append_data(frame)
                
                writer.close()
                video_bytes.seek(0)
                video_data = video_bytes.read()
            return video_data


class Magic3D(GenModelInstance):
    def __init__(self, ckpt: str = ["t3d/configs/magic3d-coarse-if.yaml", "t3d/configs/magic3d-refine-sd.yaml"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), refine = True):
        self.config_f = ckpt
        self.device = device
        self.refine = refine
    def gen(self, prompt):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f[0]
        input_ns.gpu =self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        
        result = main(input_ns, extras)
        if self.refine:
            input_ns.config = self.config_f[1]
            extras.append(f"system.geometry_convert_from={result[1]}")
            # system.geometry_convert_override.isosurface_threshold=some_value 0-20.
            # extras.append(f"system.geometry_convert_override.isosurface_threshold={20.0}")
            img_list = main(input_ns, extras)[0]
            with io.BytesIO() as video_bytes:
                # Use imageio to write frames to a video
                writer = imageio.get_writer(video_bytes, format='mp4', fps=30)
                
                for frame in img_list:
                    writer.append_data(frame)
                
                writer.close()
                video_bytes.seek(0)
                video_data = video_bytes.read()
            return video_data
        else:
            img_list = result[0]
            with io.BytesIO() as video_bytes:
                # Use imageio to write frames to a video
                writer = imageio.get_writer(video_bytes, format='mp4', fps=30)
                
                for frame in img_list:
                    writer.append_data(frame)
                
                writer.close()
                video_bytes.seek(0)
                video_data = video_bytes.read()
            return video_data
           
class SJC(GenModelInstance):
    def __init__(self, ckpt: str = "t3d/configs/sjc.yaml", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        self.config_f = ckpt
        self.device = device
    def gen(self, prompt):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f
        input_ns.gpu =self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        # pass kwargs to extras
            
        result = main(input_ns, extras)
        with io.BytesIO() as video_bytes:
            # Use imageio to write frames to a video
            writer = imageio.get_writer(video_bytes, format='mp4', fps=30)
            
            for frame in result[0]:
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        return video_data
        # return result[0]
    
class LatentNeRF(GenModelInstance):
    def __init__(self, ckpt: list = ["t3d/configs/latentnerf.yaml", "t3d/configs/latentnerf-refine.yaml"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), refine = True):
        self.config_f = ckpt
        self.device = device
        self.refine = refine
    def gen(self, prompt):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f[0]
        input_ns.gpu =self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
        
            
        result = main(input_ns, extras)
        if self.refine:
            input_ns.config = self.config_f[1]
            extras.append(f"system.weights={result[1]}")
            # system.geometry_convert_override.isosurface_threshold=some_value 0-20.
            # extras.append(f"system.geometry_convert_override.isosurface_threshold={20.0}")
            img_list = main(input_ns, extras)[0]
            with io.BytesIO() as video_bytes:
                # Use imageio to write frames to a video
                writer = imageio.get_writer(video_bytes, format='mp4', fps=30)
                
                for frame in img_list:
                    writer.append_data(frame)
                
                writer.close()
                video_bytes.seek(0)
                video_data = video_bytes.read()
            return video_data
        else:
            img_list = result[0]
            with io.BytesIO() as video_bytes:
                # Use imageio to write frames to a video
                writer = imageio.get_writer(video_bytes, format='mp4', fps=30)
                
                for frame in img_list:
                    writer.append_data(frame)
                
                writer.close()
                video_bytes.seek(0)
                video_data = video_bytes.read()
            return video_data
            
    
class Fantasia3D(GenModelInstance):
    def __init__(self, ckpt: list = ["t3d/configs/fantasia3d.yaml", "t3d/configs/fantasia3d-texture.yaml"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), texture = True):
        self.config_f = ckpt
        self.device = device
        self.texture = texture
    def gen(self, prompt):
        import argparse
        input_ns = argparse.Namespace(**{})
        input_ns.config = self.config_f[0]
        input_ns.gpu =self.device
        input_ns.train = True
        input_ns.validate = False
        input_ns.test = False
        input_ns.export = False
        input_ns.gradio = False
        input_ns.typecheck = False
        input_ns.verbose = False 
        extras = [f'system.prompt_processor.prompt={prompt}']
            
        result = main(input_ns, extras)
        if self.texture:
            input_ns.config = self.config_f[1]
            extras.append(f"system.geometry_convert_from={result[1]}")
            img_list = main(input_ns, extras)[0]
            with io.BytesIO() as video_bytes:
                # Use imageio to write frames to a video
                writer = imageio.get_writer(video_bytes, format='mp4', fps=30)
                
                for frame in img_list:
                    writer.append_data(frame)
                
                writer.close()
                video_bytes.seek(0)
                video_data = video_bytes.read()
            return video_data
        else:
            img_list = result[0]
            with io.BytesIO() as video_bytes:
                # Use imageio to write frames to a video
                writer = imageio.get_writer(video_bytes, format='mp4', fps=30)
                
                for frame in img_list:
                    writer.append_data(frame)
                
                writer.close()
                video_bytes.seek(0)
                video_data = video_bytes.read()
            return video_data
    