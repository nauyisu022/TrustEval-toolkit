from ..requests import T2IRequest, VLMRequest, LLMRequest

class ModelRequestFactory:
    @staticmethod
    def create_request(request_type, model_name, prompt, **kwargs):
        if request_type == 't2i':
            return T2IRequest(model_name, prompt, **kwargs)
        elif request_type == 'vlm':
            return VLMRequest(model_name, prompt, **kwargs)
        elif request_type == 'llm':
            return LLMRequest(model_name, prompt, **kwargs)
        else:
            raise ValueError(f"Unknown request type: {request_type}")
        
    @staticmethod
    async def create_request_async(request_type, model_name, prompt, **kwargs):
        if request_type == 't2i':
            return T2IRequest(model_name, prompt, **kwargs)
        elif request_type == 'vlm':
            return VLMRequest(model_name, prompt, **kwargs)
        elif request_type == 'llm':
            return LLMRequest(model_name, prompt, **kwargs)
        else:
            raise ValueError(f"Unknown request type: {request_type}")
