from .base_request import ModelRequest

class T2IRequest(ModelRequest):
    def __init__(self, model_name, prompt, **kwargs):
        super().__init__(model_name, prompt, **kwargs)
        self.save_folder = kwargs.get('save_folder', '')
        self.file_name = kwargs.get('file_name', '')
        self.kwargs.pop('save_folder', None)
        self.kwargs.pop('file_name', None)
    
    def send_request(self, request_handler):
        return request_handler.handle_request(self)
    