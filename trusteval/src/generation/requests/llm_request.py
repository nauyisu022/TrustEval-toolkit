from .base_request import ModelRequest

class LLMRequest(ModelRequest):
    def send_request(self, request_handler):
        return request_handler.handle_request(self)
    
    async def send_request_async(self, request_handler):
        return await request_handler.handle_request_async(self)