import asyncio
from pipeline import TextWebSearchPipeline
import json

def main(e):
    """
    Main function to run the TextWebSearchPipeline with specified parameters.
    """
    # Define the instruction and information for the pipeline
    instruction = ("Please find examples about the malicious actions aim at {}, do not return its mitigation methods. ".format(e.lower()))
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
    output_path = "generated_raw_data/safety/{}.json".format(e)

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

    # Run the pipeline and save the output to the specified file
    asyncio.run(extractor.run(output_file=output_path))

if __name__ == "__main__":
    with open('../metadata/LLM/safety/taxonomy_harmful.json', 'r') as f:
        data = json.load(f)
    for k, v in data.items():
        for el in v:
            # judge whether the file exists
            import os
            if not os.path.exists('generated_raw_data/safety/{}.json'.format(el)):
                print(el)
                main(el)
