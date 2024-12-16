import aiohttp
from bs4 import BeautifulSoup

async def fetch_and_extract_text(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                html = await response.text()

                soup = BeautifulSoup(html, 'html.parser')
                for element in soup(['script', 'style', 'head', 'title', 'meta', '[document]']):
                    element.decompose()

                text = soup.get_text()
                text = ' '.join(text.split())
                return text

    except aiohttp.ClientError as req_err:
        print(f"Error fetching {url}: {req_err}")
        return None
    except Exception as parse_err:
        print(f"Error parsing HTML: {parse_err}")
        return None
