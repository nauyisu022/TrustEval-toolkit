from .base_request import ModelRequest

class VLMRequest(ModelRequest):
    def __init__(self, model_name, prompt, **kwargs):
        super().__init__(model_name, prompt, **kwargs)
        self.image_urls = kwargs.get('image_urls')
        self.kwargs.pop('image_urls', None)

    def send_request(self, request_handler):
        return request_handler.handle_request(self)
    
    async def send_request_async(self, request_handler):
        return await request_handler.handle_request_async(self)
