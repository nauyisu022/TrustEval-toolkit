import os
import sys
import json
import asyncio
import logging
from typing import List, Optional, Dict, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm_asyncio

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Adjust project root and include in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.generation import ModelService  # Ensure this import works based on your project structure

# Proxy settings (if necessary)


class ResponseProcessor:
    """
    Processes responses from different models, handling both asynchronous and synchronous services.
    """

    def __init__(
        self,
        request_type: str,
        async_list: Optional[List[str]] = None,
        sync_list: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Initializes the ResponseProcessor.

        :param request_type: Type of request ('llm' or 'vlm').
        :param async_list: List of asynchronous model names.
        :param sync_list: List of synchronous model names.
        :param save_path: Path to save the processed responses.
        """
        self.request_type = request_type
        self.async_service_list = [
            ModelService(
                request_type=request_type,
                handler_type='api',
                model_name=model,
                config_path=os.path.join(PROJECT_ROOT, 'src/config/config.yaml'),
                temperature=0.0
            )
            for model in async_list or []
        ]
        self.sync_service_list = [
            ModelService(
                request_type=request_type,
                handler_type='local',
                model_name=model,
                config_path=os.path.join(PROJECT_ROOT, 'src/config/config.yaml'),
                temperature=0.0
            )
            for model in sync_list or []
        ]

        self.async_model_list = async_list or []
        self.sync_model_list = sync_list or []
        self.save_path = save_path
        self.base_path = os.path.dirname(save_path) if save_path else ''

    async def process_async_service(
        self,
        service: ModelService,
        model_name: str,
        prompt: str,
        image_urls: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Processes an asynchronous service request.

        :param service: The ModelService instance.
        :param model_name: Name of the model.
        :param prompt: The prompt to send.
        :param image_urls: List of image URLs (if applicable).
        :return: A dictionary with model name as key and response as value.
        """
        try:
            if self.request_type == 'llm':
                response = await service.process_async(prompt)
            elif self.request_type == 'vlm':
                response = await service.process_async(prompt, image_urls=image_urls)
            else:
                logger.warning(f"Unknown request type: {self.request_type}")
                response = None
            return {model_name: response}
        except Exception as e:
            logger.error(f"Error processing async model {model_name}: {e}")
            return {model_name: None}

    def process_sync_service(
        self,
        service: ModelService,
        model_name: str,
        prompt: str,
        image_urls: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Processes a synchronous service request.

        :param service: The ModelService instance.
        :param model_name: Name of the model.
        :param prompt: The prompt to send.
        :param image_urls: List of image URLs (if applicable).
        :return: A dictionary with model name as key and response as value.
        """
        try:
            if self.request_type == 'llm':
                response = service.process(prompt)
            elif self.request_type == 'vlm':
                response = service.process(prompt, image_urls=image_urls)
            else:
                logger.warning(f"Unknown request type: {self.request_type}")
                response = None
            return {model_name: response}
        except Exception as e:
            logger.error(f"Error processing sync model {model_name}: {e}")
            return {model_name: None}

    async def get_single_response(
        self,
        item: Dict[str, Any],
        prompt_key: str = 'enhanced_prompt',
        result_key: str = 'responses',
        base_path: str = '',
        image_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetches a single response for an item from all applicable models.

        :param item: The data item containing the prompt and possibly image URLs.
        :param prompt_key: Key to extract the prompt from the item.
        :param result_key: Key under which to store the responses.
        :param base_path: Base path for resolving image URLs.
        :param image_key: Key to extract image URLs from the item.
        :return: The updated item with responses.
        """
        prompt = item.get(prompt_key, item.get('prompt')) if prompt_key == 'enhanced_prompt'  else item.get(prompt_key)
        if not prompt:
            logger.debug("No prompt found; skipping item.")
            return item

        image_urls = None
        if self.request_type == 'vlm' and image_key:
            image_list = item.get(image_key, [])
            if image_list:
                if isinstance(image_list, str):
                    image_list = [image_list]
                image_urls = [os.path.join(base_path, url) for url in image_list]

        # Prepare async tasks
        async_tasks = [
            self.process_async_service(service, model, prompt, image_urls)
            for service, model in zip(self.async_service_list, self.async_model_list)
            if item.get(result_key, {}).get(model) is None
        ]

        # Prepare sync tasks
        sync_services_and_models = [
            (service, model)
            for service, model in zip(self.sync_service_list, self.sync_model_list)
            if item.get(result_key, {}).get(model) is None
        ]
        sync_tasks = []
        if sync_services_and_models:
            services, models = zip(*sync_services_and_models)
            with ThreadPoolExecutor(max_workers=10) as executor:
                sync_tasks = list(
                    executor.map(
                        lambda p: self.process_sync_service(*p),
                        zip(
                            services,
                            models,
                            [prompt] * len(models),
                            [image_urls] * len(models)
                        )
                    )
                )

        # Execute async tasks
        async_responses = await asyncio.gather(*async_tasks) if async_tasks else []

        # Combine responses
        responses = async_responses + sync_tasks

        # Update item with responses
        if result_key not in item:
            item[result_key] = {}

        for response in responses:
            if isinstance(response, dict):
                for model, response_text in response.items():
                    item[result_key][model] = response_text

        return item

    async def get_responses(
        self,
        data: List[Dict[str, Any]],
        prompt_key: str = 'prompt',
        result_key: str = 'responses',
        image_key: Optional[str] = None,
        max_concurrent_tasks: int = 5,
        base_path: str = ''
    ) -> None:
        """
        Processes all data items to fetch responses from models.

        :param data: List of data items to process.
        :param prompt_key: Key to extract the prompt from each item.
        :param result_key: Key under which to store the responses.
        :param image_key: Key to extract image URLs from each item.
        :param max_concurrent_tasks: Maximum number of concurrent tasks.
        :param base_path: Base path for resolving image URLs.
        """
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        async def process_item(item: Dict[str, Any], index: int) -> None:
            async with semaphore:
                updated_item = await self.get_single_response(
                    item,
                    prompt_key=prompt_key,
                    result_key=result_key,
                    base_path=base_path,
                    image_key=image_key
                )
                data[index] = updated_item
                self._save_data(data)  # Pass the data explicitly here

        async def main():
            tasks = [
                process_item(item, idx)
                for idx, item in enumerate(data)
            ]
            for future in tqdm_asyncio.as_completed(tasks, desc="Processing items"):
                await future

        await main()

    def _save_data(self, data: List[Dict[str, Any]]) -> None:
        """
        Saves the current state of the data to the save_path in JSON format.
        """
        if self.save_path:
            try:
                with open(self.save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                logger.info(f"Data saved to {self.save_path}")
            except Exception as e:
                logger.error(f"Failed to save data: {e}")

    def load_existing_responses(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Loads existing responses from the save_path if the file exists and matches the data length.

        :param data: Original data.
        :return: Data enriched with existing responses if available.
        """
        if self.save_path and os.path.exists(self.save_path):
            logger.info(f"Loading existing responses from {self.save_path}")
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    existing_responses = json.load(f)
                if len(existing_responses) == len(data):
                    return existing_responses
                else:
                    logger.warning("Existing responses length mismatch. Proceeding with original data.")
            except Exception as e:
                logger.error(f"Failed to load existing responses: {e}")
        return data


def count_invalid_responses(
    data: List[Dict[str, Any]],
    response_key: str = 'responses',
    invalid_responses: Optional[List[Any]] = None
) -> Dict[str, int]:
    """
    Counts the number of invalid responses per model.

    :param data: List of data items.
    :param response_key: Key under which responses are stored.
    :param invalid_responses: List of values considered as invalid.
    :return: Dictionary mapping model names to their invalid response counts.
    """
    if invalid_responses is None:
        invalid_responses = ["", None]

    invalid_counts = defaultdict(int)

    for entry in data:
        responses = entry.get(response_key, {})
        for model_name, response in responses.items():
            if response in invalid_responses:
                invalid_counts[model_name] += 1

    return invalid_counts


class MultiProcessor:
    """
    Handles processing of multiple data files and manages different ResponseProcessors.
    """

    def __init__(
        self,
        data_path: str,
        request_types: List[str],
        prompt_keys: List[str],
        result_keys: List[str],
        image_keys: Optional[List[Optional[str]]] = None,
        async_list: Optional[List[str]] = None,
        sync_list: Optional[List[str]] = None,
        max_concurrent_tasks: int = 5,
        file_name_extension: str = ''
    ) -> None:
        """
        Initializes the MultiProcessor.

        :param data_path: Path to the data file.
        :param request_types: List of request types corresponding to processors.
        :param prompt_keys: List of prompt keys corresponding to processors.
        :param result_keys: List of result keys corresponding to processors.
        :param image_keys: List of image keys corresponding to processors.
        :param async_list: List of asynchronous model names.
        :param sync_list: List of synchronous model names.
        :param max_concurrent_tasks: Maximum number of concurrent tasks.
        :param file_name_extension: Extension to append to the output file name.
        """
        self.data_path = data_path
        self.request_types = request_types
        self.prompt_keys = prompt_keys
        self.result_keys = result_keys
        self.image_keys = image_keys or [None] * len(request_types)
        self.async_list = async_list or []
        self.sync_list = sync_list or []
        self.max_concurrent_tasks = max_concurrent_tasks
        self.file_name_extension = file_name_extension
        self.save_path = self._construct_save_path()
        self.processors = self._create_processors()
        self._data = []

    def _construct_save_path(self) -> str:
        """
        Constructs the save path based on the data path and file name extension.

        :return: Save path string.
        """
        base, ext = os.path.splitext(self.data_path)
        return f"{base}{self.file_name_extension}{ext}"

    def _create_processors(self) -> List[ResponseProcessor]:
        """
        Creates a list of ResponseProcessor instances based on request types.

        :return: List of ResponseProcessor instances.
        """
        processors = []
        for request_type in self.request_types:
            processor = ResponseProcessor(
                request_type=request_type,
                async_list=self.async_list,
                sync_list=self.sync_list,
                save_path=self.save_path
            )
            processors.append(processor)
        return processors

    def _load_existing_responses(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Loads existing responses if available.

        :param data: Original data.
        :return: Data enriched with existing responses if available.
        """
        for processor in self.processors:
            data = processor.load_existing_responses(data)
        return data

    async def process_all(self, analyze_only: bool = False) -> None:
        """
        Processes all data items, optionally only analyzing invalid responses.

        :param analyze_only: If True, skips processing and only analyzes existing responses.
        """
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded data from {self.data_path} with {len(data)} items.")
        except Exception as e:
            logger.error(f"Failed to load data from {self.data_path}: {e}")
            return

        self._data = self._load_existing_responses(data)

        if not analyze_only:
            tasks = []
            for processor, prompt_key, result_key, image_key in zip(
                self.processors,
                self.prompt_keys,
                self.result_keys,
                self.image_keys
            ):
                tasks.append(
                    processor.get_responses(
                        self._data,
                        prompt_key=prompt_key,
                        result_key=result_key,
                        image_key=image_key,
                        max_concurrent_tasks=self.max_concurrent_tasks,
                        base_path=os.path.dirname(self.data_path)
                    )
                )
            await asyncio.gather(*tasks)

            logger.info("All responses have been processed and saved.")

        # Analyze invalid responses
        invalid_counts = count_invalid_responses(self._data, response_key='responses')
        for model_name, count in invalid_counts.items():
            logger.info(f"Model: {model_name}, Invalid Response Count: {count}")


async def generate_responses(
    data_folder: str,
    request_type: List[str],
    async_list: List[str],
    sync_list: List[str]=[],
    prompt_key: List[str]='enhanced_prompt',
    result_key: List[str]='responses',
    file_name_extension: str='_responses',
    image_key: Optional[List[Optional[str]]] = None
) -> None:
    """
    Processes all data files in a specified directory.

    :param data_folder: Path to the folder containing data files.
    :param request_type: List of request types.
    :param async_list: List of asynchronous model names.
    :param sync_list: List of synchronous model names.
    :param prompt_key: List of prompt keys.
    :param result_key: List of result keys.
    :param file_name_extension: Extension to append to output file names.
    :param image_key: List of image keys.
    """
    try:
        with open(os.path.join(data_folder, 'file_config.json'), 'r', encoding='utf-8') as f:
            file_config = json.load(f)
        files = [file['file_name'] for file in file_config]
        logger.info(f"Found {len(files)} files to process in {data_folder}.")
    except Exception as e:
        logger.error(f"Failed to load file_config.json: {e}")
        return

    for data_file in tqdm_asyncio(files, desc="Processing files"):
        if not data_file.endswith('.json'):
            logger.warning(f"Skipping non-JSON file: {data_file}")
            continue
        
        if os.path.exists(os.path.join(data_folder, data_file.replace('.json', f'_enhanced.json'))):
            data_path = os.path.join(data_folder, data_file.replace('.json', f'_enhanced.json'))
        else:
            data_path = os.path.join(data_folder, data_file)
        logger.info(f"Processing file: {data_file} -> {data_path}")

        try:
            processor = MultiProcessor(
                data_path=data_path,
                request_types=request_type if isinstance(request_type, list) else [request_type],
                prompt_keys=prompt_key if isinstance(prompt_key, list) else [prompt_key],
                result_keys=result_key if isinstance(result_key, list) else [result_key],
                image_keys=image_key if isinstance(image_key, list) else [image_key],
                async_list=async_list,
                sync_list=sync_list,
                max_concurrent_tasks=20,
                file_name_extension=file_name_extension
            )
            await processor.process_all(analyze_only=False)
        except Exception as e:
            logger.error(f"Failed to process file {data_file}: {e}", exc_info=True)

if __name__ == '__main__':

    request_type = ['llm',]

    folder_path='D:\Paper\TrustAGI-code\examples/truthfulness_llm/final'
    # async_list = ['gpt-4o', 'gpt-4o-mini', 
    #                       'glm-4v-plus','llama-3.2-90B-V',
    #                       'claude-3.5-sonnet','qwen-2-vl-72B', 'gemini-1.5-pro',
    #                        ]
    # async_list=['gpt-4o', 'gpt-4o-mini',"gpt-3.5-turbo",
    #               'claude-3.5-sonnet', 'claude-3-haiku',
    #             'gemini-1.5-pro', 'gemini-1.5-flash', 'gemma-2-27B',
    #               'llama-3.1-70B','llama-3.1-8B',
    #               'glm-4-plus', 'qwen-2.5-72B', 
    #               'mistral-8x7B' ,"mistral-8x22B", 
    #               "yi-lightning", 'deepseek-chat']
    async_list = ['llama-3.1-8B']
    sync_list=[]

    prompt_key = ['prompt',]
    result_key = ['responses',]
    image_key = ['image_urls']
    file_name_extension = '_responses'

    for request_type, prompt_key, result_key in zip(request_type, prompt_key, result_key):
        asyncio.get_event_loop().run_until_complete(process_data_folder(folder_path, request_type, async_list, sync_list, prompt_key, result_key, file_name_extension, image_key=image_key))
    print("All files processed.")