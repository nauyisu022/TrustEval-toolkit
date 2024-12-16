import os
import json
import re
from dotenv import load_dotenv
from src.metadata_curator.utils import (
    bing_search,
    process_search_results,
    get_search_keyword,
    generate_jsonformat_prompt
)
from src.generation import ModelService
# Load environment variables from a .env file
load_dotenv()

service = ModelService(
    request_type='llm',         # The type of request (e.g., 'llm' for language models)
    handler_type='api',         # The type of handler (e.g., 'api' for API-based requests)
    model_name='gpt-4o-mini',        # The name of the model to use
    config_path='src/config/config.yaml'
)

class TextWebSearchPipeline:
    def __init__(self, instruction, basic_information, need_azure, output_format, keyword_model="gpt-4o", response_model="gpt-4o", include_url=True, include_summary=True, include_original_html=True, include_access_time=True, direct_search_keyword=None, direct_site=None):
        """
        Initialize the TextWebSearchPipeline with various parameters and settings.

        :param instruction: Instruction for the pipeline.
        :param basic_information: basic_information to consider during the search.
        :param output_format: User-specific dictionary for response formatting.
        :param direct_search_keyword: Direct keyword for the search, default is None.
        :param direct_site: Specific site to search within, default is None.
        :param keyword_model: Model name for generating keywords.
        :param response_model: Model name for generating responses.
        :param include_url: Whether to include URLs in the output.
        :param include_summary: Whether to include summaries in the output.
        :param include_original_html: Whether to include original HTML in the output.
        :param include_access_time: Whether to include access times in the output.
        """
        self.instruction = instruction
        self.basic_information = '. '.join([f"{k} is {v}" for k, v in basic_information.items()])
        # self.get_response = get_openai_text_response
        self.get_response = service.process_async
        self.need_azure = need_azure
        self.output_format = output_format
        self.keyword_model = keyword_model
        self.response_model = response_model
        self.summaries = []
        self.responses = []
        self.original_html = []
        self.url = []
        self.access_time = []
        self.include_url = include_url
        self.include_summary = include_summary
        self.include_original_html = include_original_html
        self.include_access_time = include_access_time
        self.direct_search_keyword = direct_search_keyword
        self.direct_site = direct_site

        # Load API keys and endpoints from environment variables
        self.bing_subscription_key = os.getenv('BING_SEARCH_V7_SUBSCRIPTION_KEY')
        self.bing_endpoint = os.getenv('BING_SEARCH_V7_ENDPOINT')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_api_base_url = os.getenv('OPENAI_API_BASE_URL')

    def extract_keywords_from_response(self, response):
        """
        Extract keywords from the response using a regular expression.

        :param response: The response string containing keywords.
        :return: A list of extracted keywords.
        """
        try:
            match = re.search(r'\[\[(.*?)\]\]', response.strip())
            if match:
                keywords = match.group(1).split(',')
                keywords = [keyword.strip() for keyword in keywords]
                return keywords
            else:
                raise ValueError("Keywords not found in response.")
        except ValueError as e:
            print(e)
            return None

    async def run_search_pipeline(self):
        """
        Run the search pipeline to obtain search results.

        :return: Processed search results.
        """
        if self.direct_search_keyword:
            keywords_query = self.direct_search_keyword
        else:
            user_input = f"{self.instruction} {self.basic_information}"
            keyword_prompt = get_search_keyword(user_input)
            keywords_response = await self.get_response(model=self.keyword_model, prompt=keyword_prompt)
            try:
                keywords = self.extract_keywords_from_response(keywords_response)
            except ValueError as e:
                print(e)
                return None

            if not keywords:
                print("No keywords extracted.")
                return None

            keywords_query = ','.join(keywords)

        if self.direct_site:
            keywords_query += f"+site:{self.direct_site}"

        print(f"Keywords: {keywords_query}")

        try:
            search_results = bing_search(keywords_query)
        except Exception as e:
            print(f"Error during Bing search: {e}")
            return None

        processed_results = await process_search_results(search_results, self.need_azure)
        return processed_results

    async def generate_output_json(self, output_path):
        """
        Generate the output JSON file from the search pipeline results.

        :param output_path: The file path to save the output JSON.
        :return: The path to the output JSON file.
        """
        search_results = await self.run_search_pipeline()
        if not search_results:
            print("No search results to save.")
            return None

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(search_results, f, ensure_ascii=False, indent=4)
        print(f"Web results saved to {output_path}, total results: {len(search_results)}")
        return output_path

    def extract_summaries(self, output_file):
        """
        Extract summaries, original HTML, URLs, and access times from the output JSON file.

        :param output_file: The file path to the output JSON.
        """
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.summaries = [item['summary'] for item in data]
        self.original_html = [item['original_html'] for item in data]
        self.url = [item['url'] for item in data]
        self.access_time = [item['access_time'] for item in data]

    async def generate_responses(self):
        """
        Generate responses based on the extracted summaries and user dictionary.
        """
        for summary in self.summaries:
            formatted_dict = str(self.output_format)
            prompt = generate_jsonformat_prompt(
                instruction=self.instruction,
                basic_information="",
                summary=summary,
                jsonformat=formatted_dict
            )
            try:
                response = await self.get_response(model=self.response_model, prompt=prompt)
                response = re.sub(r"```json(.*?)```", r"\1", response, flags=re.DOTALL).strip()

                if response.startswith("{") and response.endswith("}"):
                    self.responses.append(json.loads(response))
                else:
                    print(f"Invalid JSON response: {response}")
            except Exception as e:
                print(f"Error while generating response: {e}")

    def merge_responses(self):
        """
        Merge the generated responses with the additional metadata.

        :return: A list of merged responses.
        """
        merged_responses = []
        for idx, response in enumerate(self.responses):
            if self.include_url:
                response["url"] = self.url[idx]
            if self.include_summary:
                response["summary"] = self.summaries[idx]
            if self.include_original_html:
                response["original_html"] = self.original_html[idx]
            if self.include_access_time:
                response["access_time"] = self.access_time[idx]
            merged_responses.append(response)
        return merged_responses

    async def run(self, output_file="./final_output.json"):
        """
        Run the entire pipeline and save the final output to a JSON file.

        :param output_file: The file path to save the final output JSON.
        """
        await self.generate_output_json(output_file)
        self.extract_summaries(output_file)
        await self.generate_responses()
        merged_responses = self.merge_responses()

        # with open(output_file, 'w', encoding='utf-8') as f:
        #     json.dump(merged_responses, f, indent=4)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_responses, f, indent=4)
        print(f"Final results saved to {output_file}, total responses: {len(merged_responses)}")