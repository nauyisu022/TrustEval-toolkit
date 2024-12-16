
import os,sys
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# Adjust the sys.path before importing local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)

from src.generation import ModelService

class APIService:
    def __init__(self,model_name='gpt-4o',temperature=0.6,):
        self.service = ModelService(
            request_type='llm',
            handler_type='api',
            config_path='src/config/config.yaml',
            model_name=model_name,
            temperature=temperature,
            top_p=1,
        )
        
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=8),  
    )
    def get_response(self, prompt: str,system_prompt:str=None) -> str:
        try:
            response = self.service.process(prompt=prompt,system_prompt=system_prompt)
            return response
        except Exception as e:
            print(f"Error during API call: {e}")
            return "" 
        