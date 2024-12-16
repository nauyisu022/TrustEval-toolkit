from abc import ABC, abstractmethod

# Parameters: https://platform.openai.com/docs/api-reference/chat/create

class ModelRequest(ABC):
    def __init__(self, model_name, prompt, **kwargs):
        self.model_name = model_name
        self.prompt = prompt
        self.kwargs = kwargs

    @abstractmethod
    def send_request(self, request_handler):
        pass
