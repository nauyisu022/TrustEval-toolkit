import asyncio
from aiohttp import ClientSession
from typing import List, Any, Callable, Tuple,Dict
import asyncio
from typing import List, Dict, Callable, Any
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
load_dotenv()



async def apply_function_concurrently(
    func: Callable[..., Dict[str, Any]],
    elements: List[Dict[str, Any]],
    max_concurrency: int
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency)
    results = [None] * len(elements)

    async def bound_function(index: int, element: Dict[str, Any]):
        async with semaphore:
            result = await func(**element)
            results[index] = result

    tasks = [bound_function(index, element) for index, element in enumerate(elements)]
    await tqdm_asyncio.gather(*tasks)
    return results




## usage example：
if __name__=='__main__':
    # Sample asynchronous function to process each element
    async def process_element(element: Dict[str, Any]) -> Dict[str, Any]:
        # For demonstration purposes, let's add a new key-value pair to each element
        # For example, count the number of words in the prompt
        text = element.get('text', '')
        result = await enhancer.enhance_diversity(
            text, 
            current_format="multiple_choice",
        )
        ## async function
        element['enhanced_results']=result 
        return element

    # Main function to run the concurrent processing
    async def main():
        data_list=load_json('/media/sata1/wtm/TrustAGI-code/MMLU_300.json')
        max_concurrency = 5  # Set the maximum number of concurrent tasks
        results = await apply_function_concurrently(
            func=process_element,
            elements=data_list,
            max_concurrency=max_concurrency
        )
        save_json(results,'/media/sata1/wtm/TrustAGI-code/MMLU_300_enhanced.json')


    if __name__ == "__main__":
        asyncio.get_event_loop().run_until_complete(main())