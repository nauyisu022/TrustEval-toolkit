import tempfile
from typing import Callable, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .base_qa_model import QAModel, QAModelInstance
from .imageqa_model import ImageQAModel

videoqa_models = {

}


def list_videoqa_models():
	return list(videoqa_models.keys())


class VideoQAModel(QAModel):
	def __init__(
			self,
			model_name,
			prompt_name: str,
			prompt_func: Callable,
			model: QAModelInstance = None,
			torch_device: Union[int, str] = -1,
			precision=torch.bfloat16,
			choice_format='letter',
			enable_choice_search: bool = False,
	):
		super().__init__(model_name, prompt_name, prompt_func, choice_format, enable_choice_search)

		if isinstance(torch_device, str):
			torch_device = torch.device(torch_device)
		else:
			if torch_device == -1:
				torch_device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
			else:
				torch_device = torch.device(f"cuda:{torch_device}")

		if model is None:
			print(f"Loading {model_name}...")
			class_name, ckpt = videoqa_models[model_name]
			self.model_precision = precision
			self.model = eval(class_name)(ckpt, torch_device, self.model_precision)
			print(f"Finish loading {model_name}")
		else:
			print(f"Using provided self.model...")
			self.model = model

	@torch.no_grad()
	def _qa(self, data, prompt):
		if isinstance(data, str):
			return self.model.qa(data, prompt)
		else:
			with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
				with open(tmp.name, 'wb') as file:
					file.write(data)
				video_path = tmp.name
				answer = self.model.qa(video_path, prompt)
			return answer


def sample_frames(video_path, n):
	import cv2
	# Open the video file
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print("Error: Could not open video.")
		return []

	# Calculate total number of frames and video FPS
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	# Calculate interval in terms of frames
	interval = max(1, total_frames // n)

	# Sample frames
	sampled_frames = []
	for i in range(0, total_frames, interval):
		# Set the current frame position
		cap.set(cv2.CAP_PROP_POS_FRAMES, i)

		# Read the frame
		ret, frame = cap.read()
		if not ret:
			print(f"Error: Could not read frame {i}.")
			break

		# Convert the frame to PIL Image
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		pil_img = Image.fromarray(frame_rgb)
		sampled_frames.append(pil_img)

		# Stop if we have collected n frames
		if len(sampled_frames) >= n:
			break

	# Release the video capture object
	cap.release()

	return sampled_frames


def get_contrasting_color(image, x, y, width, height):
	"""
	Determine a contrasting color (black or white) based on the average color of a specified area in the image.
	"""
	# Crop the relevant part of the image
	cropped_image = image.crop((x, y, x + width, y + height))
	# Convert to numpy array for analysis
	np_image = np.array(cropped_image)
	# Calculate the average color
	average_color = np.mean(np_image, axis=(0, 1))
	# Brightness calculation based on perceived luminance
	brightness = np.sqrt(0.299 * average_color[0] ** 2 + 0.587 * average_color[1] ** 2 + 0.114 * average_color[2] ** 2)
	# Return white for dark backgrounds and black for light backgrounds
	return 'white' if brightness < 128 else 'black'


def concatenate_image(images, rows, columns, separator_width=10):
	# Ensure we have the exact number of images needed
	if len(images) != rows * columns:
		raise ValueError(f"Expected {rows * columns} images, but got {len(images)}.")

	# Calculate the max width and height of images to standardize sizes
	max_width = max(img.width for img in images)
	max_height = max(img.height for img in images)

	# Resize images to the max width and height
	resized_images = [img.resize((max_width, max_height), Image.Resampling.LANCZOS) for img in images]

	# Calculate the total width and height for the combined image
	total_width = max_width * columns + separator_width * (columns - 1)
	total_height = max_height * rows + separator_width * (rows - 1)
	combined_image = Image.new('RGB', (total_width, total_height), color='white')

	# Place images in the specified grid
	x_offset = 0
	y_offset = 0
	for i, img in enumerate(resized_images):
		combined_image.paste(img, (x_offset, y_offset))
		if (i + 1) % columns == 0:  # Move to the next row after the last column
			x_offset = 0
			y_offset += img.height + separator_width
		else:  # Move to the next column
			x_offset += img.width + separator_width

	# Add numbers to each image for identification
	draw = ImageDraw.Draw(combined_image)
	try:
		font_size = (max_width + max_height) // 2 // 12
		font = ImageFont.load_default(size=font_size)
	except IOError:
		font = ImageFont.truetype("arial", 20)

	x_offset = 0
	y_offset = 0
	for i, img in enumerate(resized_images):
		text = str(i + 1)
		text_x = x_offset + 10
		text_y = y_offset + 10
		text_width, text_height = font_size, font_size
		font_color = get_contrasting_color(combined_image, text_x, text_y, text_width, text_height)
		draw.text((text_x, text_y), text, fill=font_color, font=font)
		if (i + 1) % columns == 0:
			x_offset = 0
			y_offset += img.height + separator_width
		else:
			x_offset += img.width + separator_width

	return combined_image


def video_to_concat_image(video_path, num_rows, num_columns):
	return concatenate_image(sample_frames(video_path, num_rows * num_columns), num_rows, num_columns)


class ImageQAModel4Video(VideoQAModel):
	def __init__(
			self,
			model: ImageQAModel,
			prompt_name: str,
			prompt_func: Callable,
			num_rows: int = 2,
			num_columns: int = 2,
			choice_format='letter',
			enable_choice_search: bool = False,
	):
		super(VideoQAModel, self).__init__(model.model_name, prompt_name, prompt_func, choice_format, enable_choice_search)
		self.num_rows = num_rows
		self.num_columns = num_columns
		self.num_frames = self.num_rows * self.num_columns
		self.model = model

	@torch.no_grad()
	def _qa(self, data, prompt):
		if isinstance(data, Image.Image):
			return self.model._qa(data, prompt)
		elif isinstance(data, str):
			return self.model._qa(video_to_concat_image(data, self.num_rows, self.num_columns), prompt)
		else:
			with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
				with open(tmp.name, 'wb') as file:
					file.write(data)
				video_path = tmp.name
				answer = self.model._qa(video_to_concat_image(video_path, self.num_rows, self.num_columns), prompt)
			return answer
