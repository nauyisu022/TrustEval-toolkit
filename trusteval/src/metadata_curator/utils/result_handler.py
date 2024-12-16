import asyncio
from datetime import datetime
from .fetcher import fetch_and_extract_text
from .openai_handler import get_openai_text_response, get_azure_openai_text_response
from .prompt_template import summary_html_content

async def process_single_result(result, need_azure=True):
    get_response = get_azure_openai_text_response if need_azure else get_openai_text_response
    try:
        cached_url = result['cachedPageUrl']
    except KeyError as e:
        cached_url = None
        print(f"Key error while extracting URLs from search result: {e}")

    try:
        original_url = result['url']
    except KeyError as e:
        print(f"No valid URL found for search result: {e}, skipping result.")
        return None

    url_to_fetch = cached_url if cached_url else original_url

    html_content = await fetch_and_extract_text(url_to_fetch)
    if html_content is None:
        print(f"Failed to fetch content from both original and cached URLs: {result}")
        return None

    summary_prompt = summary_html_content(html_content)
    try:
        summary = await get_response(model="gpt-4o-mini", prompt=summary_prompt)
    except Exception as e:
        print(f"Error during OpenAI summarization: {e}")
        return None

    access_time = datetime.now().isoformat()

    result_dict = {
        "original_html": html_content,
        "summary": summary,
        "url": original_url,
        "access_time": access_time
    }

    return result_dict

async def process_search_results(search_results, need_azure=True):
    tasks = [process_single_result(result, need_azure) for result in search_results.get('value', [])]
    results = await asyncio.gather(*tasks)
    # Filter out None results
    return [result for result in results if result is not None]
