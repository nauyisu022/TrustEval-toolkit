import base64
import json
import os
import random
from typing import Union

import numpy as np
import torch
from PIL import Image
import sys

import argparse, os, sys, glob, yaml, math, random
import datetime, time
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import repeat
from einops import rearrange, repeat
from functools import partial
from pytorch_lightning import seed_everything
import io
import imageio

from .base_gen_model import GenModel, GenModelInstance
from .text2video_metric import Text2VideoEvalMetric


text2video_models = {
    "text2vid-zero": (
        "Text2VideoZero", 
        "runwayml/stable-diffusion-v1-5"
    ),
    
    "text2vid-zero-sdxl": (
        "Text2VideoZeroSDXL", 
        "stabilityai/stable-diffusion-xl-base-1.0"
    ),

    "zeroscope-xl": (
        "ZeroScopeXL", [
            "cerspense/zeroscope_v2_576w",
            "cerspense/zeroscope_v2_XL"
        ]
    ),
    "modelscope-t2v": (
        "ModelScopeT2V", 
        "damo-vilab/text-to-video-ms-1.7b"
    ),
    
    "animatediff": (
        "AnimateDiff", [
            "guoyww/animatediff-motion-adapter-v1-5-2", 
            "SG161222/Realistic_Vision_V5.1_noVAE"
        ]
    ),
    
    "animateLCM": (
        "AnimateLCM", [
            "wangfuyun/AnimateLCM", 
            "emilianJR/epiCRealism", 
            "AnimateLCM_sd15_t2v_lora.safetensors", 
            "lcm-lora"
        ]
    ),
    "free-init": (
        "FreeInit", [
            "guoyww/animatediff-motion-adapter-v1-5-2", 
            "SG161222/Realistic_Vision_V5.1_noVAE"
        ]
    ),
    "VideoCraft2": (
        "VideoCraft2",[
            "model_zoo/VideoCrafter/configs/inference_t2v_512_v2.0.yaml",
            "model_zoo/VideoCrafter/checkpoints/base_512_v2/model.ckpt"
        ]
    )
}


def set_model_key(model_name, key):
    text2video_models[model_name] = (text2video_models[model_name][0], key)

def list_text2video_models():
    return list(text2video_models.keys())


class Text2VideoModel(GenModel):
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
            self.metrics = Text2VideoEvalMetric(selected_matrics=metrics, device=metrics_device)
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
            class_name, ckpt = text2video_models[model_name]
            self.model_presision = precision
            self.model = eval(class_name)(ckpt, precision, torch_device)
            print(f"Finish loading {model_name}")
        else:
            print("Using provided model...")
            self.model = model
            
    @torch.no_grad()
    def gen(self, prompt):
        output = self._gen(prompt)
        # release the memory
        torch.cuda.empty_cache()
        del self.model
        if self.metrics == "No Metrics":
            result = {"output": output}
        else:
            result = {"output": output, "metrics": self.metrics.eval_with_metrics(prompt, output)}
        return result
            
    def _data_to_str(self, data):
        # the data here for text2image model is the prompt, so it should be a str
        if isinstance(data, str):
            return data
        else:
            raise ValueError("Invalid data type")


class Text2VideoZero(GenModelInstance):
    def __init__(self, ckpt:str = "runwayml/stable-diffusion-v1-5", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), fps = 16, video_length = 32, height = 256, width = 256):
        from diffusers import TextToVideoZeroPipeline
        self.pipeline = TextToVideoZeroPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
        self.fps = fps
        self.video_length = video_length
        self.height = height
        self.width = width
        
        self.pipeline.safety_checker = lambda images, clip_input: (images, False)
    def gen(self, prompt):
        result = self.pipeline(prompt, video_length = self.video_length, height = self.height, width = self.width).images
        with io.BytesIO() as video_bytes:
            writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
            
            for frame in result:
                frame = (frame * 255).astype(np.uint8)
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        
        return video_data

class Text2VideoZeroSDXL(GenModelInstance):
    def __init__(self, ckpt:str = "stabilityai/stable-diffusion-xl-base-1.0", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), fps = 16, video_length = 32, height = 256, width = 256):
        from diffusers import TextToVideoZeroSDXLPipeline
        self.pipeline = TextToVideoZeroSDXLPipeline.from_pretrained(ckpt, torch_dtype=precision, use_safetensors=True).to(device)
        self.pipeline.enable_model_cpu_offload()
        self.fps = fps  
        self.video_length = video_length
        self.height = height
        self.width = width
        
        self.pipeline.safety_checker = lambda images, clip_input: (images, False)
    def gen(self, prompt):
        result = self.pipeline(prompt,video_length = self.video_length,height = self.height,width = self.width).images
        with io.BytesIO() as video_bytes:
            writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
            
            for frame in result:
                frame = (frame * 255).astype(np.uint8)
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        
        return video_data
    
class ZeroScopeXL(GenModelInstance):
    def __init__(self, ckpt: list = ["cerspense/zeroscope_v2_576w","cerspense/zeroscope_v2_XL"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), fps = 16, video_length = 32, height = 576, width = 1024):
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

        self.pipeline = DiffusionPipeline.from_pretrained(ckpt[0], torch_dtype=precision).to(device)
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)
        self.pipeline.enable_vae_slicing() 

        self.upscale = DiffusionPipeline.from_pretrained(ckpt[1], torch_dtype=torch.float16).to(device)
        self.upscale.scheduler = DPMSolverMultistepScheduler.from_config(self.upscale.scheduler.config)
        self.upscale.enable_model_cpu_offload()
        self.upscale.unet.enable_forward_chunking(chunk_size=1, dim=1)
        self.upscale.enable_vae_slicing()
        
        self.fps = fps
        self.video_length = video_length
        self.height = height
        self.width = width
        
        self.pipeline.safety_checker = lambda images, clip_input: (images, False)

    def gen(self, prompt):
        video_frames = self.pipeline(prompt, num_frames=self.video_length).frames[0]
        video = [Image.fromarray((frame*255).astype(np.uint8)).resize((self.width, self.height)) for frame in video_frames]
        video_frames = self.upscale(prompt, video=video, strength=0.6).frames[0]
        with io.BytesIO() as video_bytes:
            writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
            
            for frame in video_frames:
                frame = (frame * 255).astype(np.uint8)
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        
        return video_data


class ModelScopeT2V(GenModelInstance):
    def __init__(self, ckpt: str = "damo-vilab/text-to-video-ms-1.7b", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"),EnableVAESlicing=True, fps = 16, video_length = 32, height = 256, width = 256):
        from diffusers import DiffusionPipeline
        self.pipeline = DiffusionPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
        self.pipeline.enable_model_cpu_offload()
        if EnableVAESlicing:
            self.pipeline.enable_vae_slicing()
            
        self.fps = fps
        self.video_length = video_length
        self.height = height
        self.width = width

        self.pipeline.safety_checker = lambda images, clip_input: (images, False)
    def gen(self, prompt):
        video_frames = self.pipeline(prompt, num_frames = self.video_length, height = self.height, width = self.width).frames[0]
        with io.BytesIO() as video_bytes:
            writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
            
            for frame in video_frames:
                frame = (frame * 255).astype(np.uint8)
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        return video_data
    

class AnimateDiff(GenModelInstance):
    def __init__(self, ckpt: list = ["guoyww/animatediff-motion-adapter-v1-5-2", "SG161222/Realistic_Vision_V5.1_noVAE"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), fps = 16, video_length = 32, height = 256, width = 256):
        from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
        adapter = MotionAdapter.from_pretrained(ckpt[0])
        model_id = ckpt[1]
        self.pipeline = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=precision).to(device)
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1
        )
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        self.pipeline.enable_model_cpu_offload()
        
        self.fps = fps
        self.video_length = video_length
        self.height = height
        self.width = width
        
        self.pipeline.safety_checker = lambda images, clip_input: (images, False)

    def gen(self, prompt):
        output = self.pipeline(prompt, num_frames=self.video_length, guidance_scale=7.5, num_inference_steps=25, generator=torch.Generator("cpu").manual_seed(42), height = self.height, width = self.width)
        result = output.frames[0]
        frame_array = [np.array(frame) for frame in result]
        with io.BytesIO() as video_bytes:
            # Use imageio to write frames to a video
            writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
            
            for frame in frame_array:
                frame = (frame * 255).astype(np.uint8)
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        
        return video_data
       

class AnimateLCM(GenModelInstance):
    def __init__(self, ckpt: list = ["wangfuyun/AnimateLCM", "emilianJR/epiCRealism", "AnimateLCM_sd15_t2v_lora.safetensors", "lcm-lora"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), fps = 16, video_length = 32, height = 256, width = 256):
        from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
        adapter = MotionAdapter.from_pretrained(ckpt[0])
        self.pipeline = AnimateDiffPipeline.from_pretrained(ckpt[1], motion_adapter=adapter, torch_dtype=precision).to(device)
        self.pipeline.scheduler = LCMScheduler.from_config(self.pipeline.scheduler.config, beta_schedule="linear")
        self.pipeline.load_lora_weights(ckpt[0], weight_name=ckpt[2], adapter_name=ckpt[3])
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_model_cpu_offload()
        
        self.fps = fps
        self.video_length = video_length
        self.height = height
        self.width = width
        
        self.pipeline.safety_checker = lambda images, clip_input: (images, False)

    def gen(self, prompt):
        output = self.pipeline(prompt, num_frames=self.video_length, guidance_scale=1.5, num_inference_steps=6, generator=torch.Generator("cpu").manual_seed(0), height = self.height, width = self.width)
        result = output.frames[0]
        frame_array = [np.array(frame) for frame in result]
        with io.BytesIO() as video_bytes:
            # Use imageio to write frames to a video
            writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
            
            for frame in frame_array:
                frame = (frame * 255).astype(np.uint8)
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        
        return video_data

class FreeInit(GenModelInstance):
    def __init__(self, ckpt: list = ["guoyww/animatediff-motion-adapter-v1-5-2", "SG161222/Realistic_Vision_V5.1_noVAE"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), fps = 16, video_length = 32, height = 256, width = 256):
        from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
        from diffusers.utils import export_to_gif
        adapter = MotionAdapter.from_pretrained(ckpt[0])
        model_id = ckpt[1]
        self.pipeline = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=precision).to(device)
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1
        )
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        self.pipeline.enable_free_init(method="butterworth", use_fast_sampling=True)
        
        self.fps = fps
        self.video_length = video_length
        self.height = height
        self.width = width
        
        self.pipeline.safety_checker = lambda images, clip_input: (images, False)

    def gen(self, prompt):
        output = self.pipeline(prompt, num_frames=self.video_length, guidance_scale=7.5, num_inference_steps=20, generator=torch.Generator("cpu").manual_seed(666), height = self.height, width = self.width)
        self.pipeline.disable_free_init()

        result = output.frames[0]
        frame_array = [np.array(frame) for frame in result]
        with io.BytesIO() as video_bytes:
            writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
            
            for frame in frame_array:
                
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        
        return video_data
    
    
class VideoCraft2(GenModelInstance):
    def __init__(self, ckpt: list = ["model_zoo/VideoCrafter/configs/inference_t2v_512_v2.0.yaml","model_zoo/VideoCrafter/checkpoints/base_512_v2/model.ckpt"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), fps = 28, video_length = 32, height = 320, width = 512):
        self.input_ns = argparse.Namespace(**{})
        self.input_ns.seed = 123
        self.input_ns.mode = 'base'
        self.input_ns.ckpt_path = 'model_zoo/VideoCrafter/checkpoints/base_512_v2/model.ckpt'
        self.input_ns.config = 'model_zoo/VideoCrafter/configs/inference_t2v_512_v2.0.yaml'
        self.input_ns.savefps = 10
        self.input_ns.ddim_steps = 50
        self.input_ns.n_samples = 1
        self.input_ns.ddim_eta = 1.0
        self.input_ns.bs = 1
        self.input_ns.height = height
        self.input_ns.width = width
        # self.input_ns.frames = -1
        self.input_ns.frames = video_length
        self.input_ns.fps = fps
        self.input_ns.unconditional_guidance_scale = 12.0
        seed_everything(self.input_ns.seed)            
        self.gpu_no = 0
        
    
    def gen(self, prompt, **kwargs):
        args = self.input_ns
        sys.path.append(os.path.join(sys.path[0], "model_zoo/VideoCrafter/scripts/evaluation"))
        sys.path.append(os.path.join(sys.path[0], "model_zoo/VideoCrafter"))
        sys.path.append(os.path.join(sys.path[0], "model_zoo/VideoCrafter/"))
        import lvdm
        from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
        from funcs import batch_ddim_sampling
        from utils.utils import instantiate_from_config
        config = OmegaConf.load(args.config)
        model_config = config.pop("model", OmegaConf.create())
        model = instantiate_from_config(model_config)
        model = model.cuda(self.gpu_no)
        assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
        model = load_model_checkpoint(model, args.ckpt_path)
        model.eval()

        assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"

        h, w = args.height // 8, args.width // 8
        frames = model.temporal_length if args.frames < 0 else args.frames
        channels = model.channels

        ## step 2: load data
        # -----------------------------------------------------------------
        # assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"
        prompts = [prompt]

        ## step 3: run over samples
        ## -----------------------------------------------------------------
        start = time.time()
        noise_shape = [1, channels, frames, h, w]
        fps = torch.tensor([args.fps]*1).to(model.device).long()

        if isinstance(prompts, str):
            prompts = [prompts]
        text_emb = model.get_learned_conditioning(prompts)

        cond = {"c_crossattn": [text_emb], "fps": fps}

        ## inference
        batch_samples = batch_ddim_sampling(model, cond, noise_shape, args.n_samples, \
                                                args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, **kwargs)
        
        video = batch_samples[0][0].detach().cpu() # [c, t, h, w]
        video = torch.clamp(video.float(), -1., 1.)
        video = (video + 1.0) / 2.0
        
        video = video.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).numpy().astype(np.uint8) 
        
        with io.BytesIO() as video_bytes:
            # Use imageio to write frames to a video
            writer = imageio.get_writer(video_bytes, format='mp4', fps=args.fps)
            
            for frame in video:
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        return video_data


