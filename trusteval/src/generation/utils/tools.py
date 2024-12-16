import asyncio
import functools
import traceback
import concurrent.futures
import time
import openai, zhipuai
import functools
import asyncio
import traceback
import replicate.exceptions


def retry_on_failure(max_retries=2, delay=2, backoff=1.1):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    result = func(*args, **kwargs)
                    if result is not None:
                        return result
                except openai.BadRequestError as e:
                    print(f"OpenAI BadRequestError: {e}")
                    return None
                except zhipuai.core._errors.APIRequestFailedError as e:
                    print(f"ZhipuAI APIRequestFailedError: {e}")
                    return None
                except replicate.exceptions.ModelError as e:
                    print(f"Replicate ModelError: {e}")
                    return None
                except Exception as e:
                    print(traceback.format_exc())
                
                retries += 1
                if retries < max_retries:
                    print(f"Retrying ({retries}/{max_retries}) in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            return None
        return wrapper_retry
    return decorator_retry

def retry_on_failure_async(max_retries=2, delay=1, backoff=1.1):
    def decorator_retry(func):
        @functools.wraps(func)
        async def wrapper_retry(*args, **kwargs):
            retries = 1
            current_delay = delay
            while retries <= max_retries:
                try:
                    result = await func(*args, **kwargs)
                    if result is not None:
                        return result
                except Exception as e:
                    print(f"Model {args[0].model_name} failed with error: {e}")
                    print(traceback.format_exc())
                retries += 1
                if retries <= max_retries:
                    print(f"Retrying ({retries}/{max_retries}) in {current_delay} seconds...")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            return None
        return wrapper_retry
    return decorator_retry



def sync_timeout(timeout):
    def decorator_timeout(func):
        @functools.wraps(func)
        def wrapper_timeout(*args, **kwargs):
            def inner_process():
                return func(*args, **kwargs)    
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(inner_process)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    print(f"Function '{func.__name__}' timed out after {timeout} seconds")
                    return None
        return wrapper_timeout
    return decorator_timeout

def async_timeout(timeout):
    def decorator_timeout(func):
        @functools.wraps(func)
        async def wrapper_timeout(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                print(f"Function '{func.__name__}' timed out after {timeout} seconds")
                return None
        return wrapper_timeout
    return decorator_timeout

