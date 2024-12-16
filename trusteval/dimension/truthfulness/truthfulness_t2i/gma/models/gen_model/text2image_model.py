import torch

from .base_gen_model import GenModel, GenModelInstance
from .text2image_metric import Text2ImageEvalMetric

# DALLE, Stable Diffusion
text2image_models = {
	"stable-diffusion-3"  : (
		"StableDiffusion3",
		"stabilityai/stable-diffusion-3-medium-diffusers",
	),
	"stable-diffusion-2-1": (
		"StableDiffusion2",
		"stabilityai/stable-diffusion-2-1",
	),
	"stable-diffusion-xl" : (
		"StableDiffusionXL",
		"stabilityai/stable-diffusion-xl-base-1.0",
	),
	"pixart-alpha"        : (
		"PixArtAlpha",
		"PixArt-alpha/PixArt-XL-2-1024-MS",
	),
    "pixart_Sigma"        : (
        "PixArtSigma",
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    ),
    "deepfloyd-if-xl"     : (
        "DeepFloydIF",
        [
            "DeepFloyd/IF-I-XL-v1.0",
            "DeepFloyd/IF-II-L-v1.0",
            "stabilityai/stable-diffusion-x4-upscaler",
        ],
    ),
    "stable-cascade"      : (
        "StableCascade",
        "stabilityai/stable-cascade"
    ),
    "wuerstchen-V2"       : (
        "Wuerstchen",
        "warp-ai/wuerstchen",
    ),
    "playground-V2.5"     : (
        "Playground",
        "playgroundai/playground-v2.5-1024px-aesthetic",
    ),
}


def set_model_key(model_name, key):
	text2image_models[model_name] = (text2image_models[model_name][0], key)


def list_text2image_models():
	return list(text2image_models.keys())


class Text2ImageModel(GenModel):
	def __init__(
			self,
			model_name: str,
			model: GenModelInstance = None,
			metrics: list[str] = None,
			metrics_device: str = "cuda",
			precision: torch.dtype = torch.float16,  # None means using all the default metrics
			torch_device: str = "cuda",
			cache_path: str = None,
	):
		super().__init__(model_name, cache_path)
		if metrics is not None:
			self.metrics = Text2ImageEvalMetric(selected_matrics=metrics, device=metrics_device)
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
			class_name, ckpt = text2image_models[model_name]
			self.model_presision = precision
			self.model = eval(class_name)(ckpt, precision, torch_device)
			print(f"Finish loading {model_name}")
		else:
			print("Using provided model...")
			self.model = model

	@torch.no_grad()
	def gen(self, gen_data: dict):# -> dict[str, Any | tuple[Any | None, None, None] | tuple[Any...:
		prompt = gen_data['prompt']
		output = self._gen(prompt)
		if self.metrics == "No Metrics":
			result = {"output": output}
		else:
			result = {"output": output, "metrics": self.metrics.eval_with_metrics(output, gen_data)}
		return result

	def _data_to_str(self, data):
		# the data here for text2image model is the prompt, so it should be a str
		if isinstance(data, str):
			return data
		else:
			raise ValueError("Invalid data type")


class StableDiffusion2(GenModelInstance):
	def __init__(
			self,
			ckpt: str = "stabilityai/stable-diffusion-2-base",
			precision: torch.dtype = torch.float16,
			device: torch.device = torch.device("cuda"),
	):
		from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

		self.pipeline = DiffusionPipeline.from_pretrained(
			ckpt, torch_dtype=precision, revision="fp16"
		)
		self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
			self.pipeline.scheduler.config
		)
		self.pipeline = self.pipeline.to(device)

	def gen(self, prompt):
		return self.pipeline(prompt, num_inference_steps=25).images[0]


class StableDiffusionXL(GenModelInstance):
	def __init__(
			self,
			ckpt: str = "stabilityai/stable-diffusion-xl-base-1.0",
			precision: torch.dtype = torch.float16,
			device: torch.device = torch.device("cuda"),
	):
		from diffusers import StableDiffusionXLPipeline

		self.pipeline = StableDiffusionXLPipeline.from_pretrained(
			ckpt, torch_dtype=precision
		).to(device)

	def gen(self, prompt):
		return self.pipeline(prompt).images[0]


class StableDiffusion3(GenModelInstance):
	def __init__(
			self,
			ckpt: str = "stabilityai/stable-diffusion-3-medium-diffusers",
			precision: torch.dtype = torch.float16,
			device: torch.device = torch.device("cuda"),
	):
		from diffusers import StableDiffusion3Pipeline

		self.pipeline = StableDiffusion3Pipeline.from_pretrained(
			ckpt, torch_dtype=precision
		)
		self.pipeline.enable_model_cpu_offload()

	def gen(self, prompt):
		return self.pipeline(prompt=prompt).images[0]


class PixArtAlpha(GenModelInstance):
	def __init__(
			self,
			ckpt: str = "PixArt-alpha/PixArt-XL-2-1024-MS",
			precision: torch.dtype = torch.float16,
			device: torch.device = torch.device("cuda"),
	):
		from diffusers import PixArtAlphaPipeline
		self.pipe = PixArtAlphaPipeline.from_pretrained(ckpt, torch_dtype=torch.float16, use_safetensors=True).to(device)

	def gen(self, prompt):
		image = self.pipe(prompt).images[0]
		return image

class PixArtSigma(GenModelInstance):
    def __init__(
        self,
        ckpt: str = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        precision: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
    ):
        from diffusers import PixArtSigmaPipeline

        self.pipeline = PixArtSigmaPipeline.from_pretrained(
            ckpt, torch_dtype=precision
        ).to(device)
        self.pipeline.enable_model_cpu_offload()

    def gen(self, prompt):
        return self.pipeline(prompt).images[0]
    
    
class StableCascade(GenModelInstance):
    def __init__(
        self,
        ckpt: str = "stabilityai/stable-cascade",
        precision: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ):
        from diffusers import StableCascadePriorPipeline

        self.pipeline = StableCascadePriorPipeline.from_pretrained(
            ckpt, torch_dtype=precision
        ).to(device)
        self.pipeline.enable_model_cpu_offload()

    def gen(self, prompt):
        return self.pipeline(prompt).images[0]

        
class Wuerstchen(GenModelInstance):
    def __init__(
        self,
        ckpt: str = "warp-ai/wuerstchen",
        precision: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
    ):
        from diffusers import AutoPipelineForText2Image

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            ckpt, torch_dtype=precision
        ).to(device)

    def gen(self, prompt):
        from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
        return self.pipeline(
            prompt,
            width=1024,
            height=1024,
            prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
            prior_guidance_scale=4.0,
            num_images_per_prompt=2,
        ).images[0]
    

class Playground(GenModelInstance):
    def __init__(
        self,
        ckpt: str = "playgroundai/playground-v2.5-1024px-aesthetic",
        precision: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
    ):
        from diffusers import DiffusionPipeline

        self.pipeline = DiffusionPipeline.from_pretrained(
            ckpt, torch_dtype=precision, variant="fp16"
        ).to(device)

    def gen(self, prompt):
        return self.pipeline(prompt=prompt, num_inference_steps=50, guidance_scale=3).images[0]
    