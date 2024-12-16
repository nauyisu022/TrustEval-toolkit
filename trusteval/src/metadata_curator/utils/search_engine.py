import os
import requests
import yaml
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config():
    config_name = 'config/config.yaml'
    with open(os.path.join(parent_dir, config_name), 'r') as file:
        config = yaml.safe_load(file)
    return config

def bing_search(query, mkt='en-US'):
    print(query)
    config = load_config()
    subscription_key = config['BING_SEARCH_V7_SUBSCRIPTION_KEY']
    endpoint = config['BING_SEARCH_V7_ENDPOINT'] + "/v7.0/search"

    if not subscription_key:
        raise ValueError("Bing Search V7 subscription key is not provided and not found in environment variables.")
    if not endpoint:
        raise ValueError("Bing Search V7 endpoint is not provided and not found in environment variables.")

    params = { 'q': query, 'mkt': mkt }
    headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        return response.json()['webPages']
    except Exception as ex:
        raise ex

def bing_image_search(query, mkt='en-US'):
    print(query)
    subscription_key = os.getenv('BING_SEARCH_V7_SUBSCRIPTION_KEY')
    endpoint = os.getenv('BING_SEARCH_V7_ENDPOINT') + "/v7.0/images/search"

    if not subscription_key:
        raise ValueError("Bing Search V7 subscription key is not provided and not found in environment variables.")
    if not endpoint:
        raise ValueError("Bing Search V7 endpoint is not provided and not found in environment variables.")

    params = { 'q': query, 'mkt': mkt }
    headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        return response.json()['value']
    except Exception as ex:
        raise ex