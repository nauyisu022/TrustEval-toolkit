import yaml
import aiofiles
from ..handlers import LocalRequestHandler, OpenAISDKHandler, OtherSDKHandler, OpenAIVisionSDKHandler, OtherVisionSDKHandler, T2IHandler,LocalOpenAISDKHandler

def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

async def load_config_async(config_file):
    async with aiofiles.open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(await f.read())
    return config

class RequestHandlerFactory:
    @staticmethod
    def create_handler(request_type, handler_type, model_name, config_path, pipe=None):
        config = load_config(config_path)
        t2i_models = config['t2i_models']
        if handler_type == 'local':
            if model_name in t2i_models.keys() and request_type == 't2i':
                return LocalRequestHandler(config, pipe)
            elif request_type == 'llm':
                return LocalOpenAISDKHandler(config)
        elif handler_type == 'api':
            openai_sdk_llms = config['openai_sdk_llms']
            other_sdk_llms = config['other_sdk_llms']
            openai_sdk_vlms = config['openai_sdk_vlms']
            other_sdk_vlms = config['other_sdk_vlms']
            if model_name in openai_sdk_llms.keys() and request_type == 'llm':
                return OpenAISDKHandler(config)
            elif model_name in other_sdk_llms.keys() and request_type == 'llm':
                return OtherSDKHandler(config)
            elif model_name in openai_sdk_vlms.keys() and request_type == 'vlm':
                return OpenAIVisionSDKHandler(config)
            elif model_name in other_sdk_vlms.keys() and request_type == 'vlm':
                return OtherVisionSDKHandler(config)
            elif model_name in t2i_models.keys() and request_type == 't2i':
                return T2IHandler(config)
            else:
                raise ValueError(f"Unknown model name: {model_name} for request type: {request_type}")
        else:
            raise ValueError(f"Unknown handler type: {handler_type}")

    @staticmethod
    async def create_handler_async(request_type, handler_type, model_name, config_path):
        if handler_type == 'local':
            return LocalRequestHandler()
        
        elif handler_type == 'api':
            config = await load_config_async(config_path)
            openai_sdk_llms = config['openai_sdk_llms']
            other_sdk_llms = config['other_sdk_llms']
            openai_sdk_vlms = config['openai_sdk_vlms']
            other_sdk_vlms = config['other_sdk_vlms']
            if model_name in openai_sdk_llms.keys() and request_type == 'llm':
                return OpenAISDKHandler(config)
            elif model_name in other_sdk_llms.keys() and request_type == 'llm':
                return OtherSDKHandler(config)
            elif model_name in openai_sdk_vlms.keys() and request_type == 'vlm':
                return OpenAIVisionSDKHandler(config)
            elif model_name in other_sdk_vlms.keys() and request_type == 'vlm':
                return OtherVisionSDKHandler(config)
            else:
                raise ValueError(f"Unknown model name: {model_name} for request type: {request_type}")
        else:
            raise ValueError(f"Unknown handler type: {handler_type}")