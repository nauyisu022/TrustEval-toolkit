import os
import json

import asyncio
from pipeline import TextWebSearchPipeline


def web_retrieval(e, instruction=None,
                  basic_information=None,
                  output_format=None,
                  output_path=None,
                  need_azure=True,
                  keyword_model="gpt-4o",
                  response_model="gpt-4o",
                  include_url=True,
                  include_summary=True,
                  include_original_html=False,
                  include_access_time=True):
    """
    Main function to run the TextWebSearchPipeline with specified parameters.
    """
    # Define the instruction and information for the pipeline
    # instruction = ("Please find examples about the malicious actions aim at {}, do not return its mitigation methods. ".format(e.lower()))
    # basic_information = {
    # }
    if output_format is None:
        noutput_format = {
            "Example": [
                "Specific example 1 mentioned on the webpage",
                "Specific example x mentioned on the webpage (and so on)"
            ]
        }
    if instruction is None:
        instruction = ("Please find Harmful word about the malicious actions about {}, do not return its mitigation methods. ".format(e.lower()))

    if basic_information is None:
        basic_information = {
        }

    # Specify the output file path
    # output_path = "generated_raw_data/safety/{}.json".format(e)
    if output_path is None:
        output_path = "generated_raw_data/safety/{}.json".format(e)

    # Initialize the TextWebSearchPipeline with the provided parameters and settings
    extractor = TextWebSearchPipeline(
        instruction=instruction,
        basic_information=basic_information,
        need_azure=True,
        output_format=output_format,
        keyword_model=keyword_model,
        response_model=response_model,
        include_url=include_url,
        include_summary=include_summary,
        include_original_html=include_original_html,
        include_access_time=include_access_time
    )

    # Run the pipeline and save the output to the specified file
    asyncio.run(extractor.run(output_file=output_path))


def run_web_retrieval(taxonomy_path='../metadata/LLM/safety/taxonomy_harmful.json'):
    with open(taxonomy_path, 'r') as f:
        data = json.load(f)
    for k, v in data.items():
        for el in v:
            # judge whether the file exists
            if not os.path.exists('generated_raw_data/safety/{}.json'.format(el)):
                print('Running web retrieval for {}'.format(el))
                web_retrieval(el)


