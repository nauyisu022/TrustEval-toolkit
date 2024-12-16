import io
import os
import base64
import requests
from abc import ABC, abstractmethod
from PIL import Image
from ..utils.tools import retry_on_failure,retry_on_failure_async,sync_timeout,async_timeout

class RequestHandler(ABC):
    @staticmethod
    def encode_image_to_base64(image):
        buffered = io.BytesIO()
        image.save(buffered, format=image.format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    @staticmethod
    def check_image_format(image_path):
        with open(image_path, 'rb') as f:
            file_header = f.read(12) 
            
        if file_header.startswith(b'\x89PNG\r\n\x1a\n'):
            return "png"
        elif file_header.startswith(b'\xFF\xD8'):
            return "jpeg"
        elif file_header[:4] == b'RIFF' and file_header[8:12] == b'WEBP':
            return "webp"
        else:
            return "unknown"
        
    @staticmethod
    def download_image(url, save_path):
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            response = requests.get(url)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return save_path
        except Exception as e:
            raise ValueError(f"Error downloading image: {save_path}")

    @staticmethod
    def generate_image_messages(self, image_urls):
        if isinstance(image_urls, str):
            image_urls = [image_urls]
        image_messages = []
        for image_url in image_urls:
            image = Image.open(image_url)
            base64_image = self.encode_image_to_base64(image)
            image_message = {"type": "image_url", "image_url": {"url": f"data:image/{self.check_image_format(image_url)};base64,{base64_image}"}}
            image_messages.append(image_message)
        
        return image_messages