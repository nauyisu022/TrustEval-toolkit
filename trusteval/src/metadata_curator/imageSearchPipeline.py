import os
import json
import re
import asyncio
from datetime import datetime
import pytz
import requests
from dotenv import load_dotenv
from src.metadata_curator.utils import get_azure_openai_text_response, get_search_keyword

load_dotenv()

class ImageWebSearchPipeline:
    def __init__(self, instruction, basic_information, output_path="a.json", keyword_model="gpt-4o", include_access_time=True, direct_search_keyword=None):
        self.instruction = instruction
        self.basic_information = '. '.join([f"{k} is {v}" for k, v in basic_information.items()])
        self.keyword_model = keyword_model
        self.include_access_time = include_access_time
        self.output_path = output_path or 'processed_image_results.json'
        self.direct_search_keyword = direct_search_keyword

        self.subscription_key = os.getenv('BING_SEARCH_V7_SUBSCRIPTION_KEY')
        self.endpoint = os.getenv('BING_SEARCH_V7_ENDPOINT') + "/v7.0/images/search"

        if not self.subscription_key:
            raise ValueError("Bing Search V7 subscription key is not provided and not found in environment variables.")
        if not self.endpoint:
            raise ValueError("Bing Search V7 endpoint is not provided and not found in environment variables.")

    def extract_keywords_from_response(self, response):
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

    async def get_search_keywords(self):
        user_input = f"{self.instruction} {self.basic_information}"
        keyword_prompt = get_search_keyword(user_input)
        keywords_response = await get_azure_openai_text_response(model=self.keyword_model, prompt=keyword_prompt)

        keywords = self.extract_keywords_from_response(keywords_response)
        if not keywords:
            print("No keywords extracted.")
            return None

        return ','.join(keywords)

    def search_images(self, query, mkt='en-US'):
        print(f"Searching for images: {query}")
        params = {'q': query, 'mkt': mkt}
        headers = {'Ocp-Apim-Subscription-Key': self.subscription_key}

        try:
            response = requests.get(self.endpoint, headers=headers, params=params)
            response.raise_for_status()
            return response.json()['value']
        except Exception as ex:
            raise ex

    def process_results(self, search_results):
        processed_results = []
        access_time = datetime.now(pytz.utc).isoformat() if self.include_access_time else None

        for item in search_results:
            processed_item = {
                "name": item.get("name"),
                "contentUrl": item.get("contentUrl"),
                "thumbnailUrl": item.get("thumbnailUrl"),
                "hostPageUrl": item.get("hostPageUrl"),
                "encodingFormat": item.get("encodingFormat"),
                "datePublished": item.get("datePublished"),
            }
            if self.include_access_time:
                processed_item["accessTime"] = access_time
            processed_results.append(processed_item)

        return processed_results

    def save_to_json(self, data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def run(self):
        
        if self.direct_search_keyword:
            keywords = self.direct_search_keyword
        else:
            keywords = await self.get_search_keywords()
            if not keywords:
                return None
        print("keywords:"+keywords)
        search_results = self.search_images(keywords)
        processed_data = self.process_results(search_results)
        self.save_to_json(processed_data, self.output_path)
        print(f"Processed {len(processed_data)} results and saved to {self.output_path}")
        return processed_data

async def search_images_method(instruction, basic_information, custom_output_path):
    pipeline_custom = ImageWebSearchPipeline(instruction, basic_information, output_path=custom_output_path)
    results_custom = await pipeline_custom.run()
    if results_custom:
        print(f"Found {len(results_custom)} images (custom output)")