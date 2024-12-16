import asyncio
from web_retrieval import TextWebSearchPipeline
from src.saver import Saver
import os

intermediate_base_path = os.path.abspath('intermediate')
saver = Saver(intermediate_base_path)

async def main(e,entity):
    """
    Main function to run the TextWebSearchPipeline with specified parameters.
    """
    # Define the instruction and information for the pipeline
    e = e.lower()
    instruction = (f"Please find examples about the privacy related or invasion actions aim at {entity}'s {e}, do not return its mitigation methods. Note that its organizations' privacy cases. ")
    basic_information = {
    }

    # Define the user-specific dictionary for response formatting
    output_format = {
        "Example": [
            "Specific example 1 mentioned on the webpage",
            "Specific example x mentioned on the webpage (and so on)"
        ]
    }

    # Specify the output file path
    output_path = f"section/privacy/privacy_t2i/intermediate/web_retrieval/{entity}/{e}.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize the TextWebSearchPipeline with the provided parameters and settings
    extractor = TextWebSearchPipeline(
        instruction=instruction,
        basic_information=basic_information,
        need_azure=True,
        output_format=output_format,
        keyword_model="gpt-4o",  # Model for generating keywords
        response_model="gpt-4o",  # Model for generating responses
        include_url=True,
        include_summary=True,
        include_original_html=False,
        include_access_time=True
    )

    await extractor.run(output_file=output_path)

if __name__ == "__main__":
    async def run_all():
        for entity in ['organization','people']:
            data = saver.read_file(f'aspects_guidemap/{entity}.json')
            for k, v in data.items():
                for el in v:
                    await main(el, entity)

    asyncio.run(run_all())
