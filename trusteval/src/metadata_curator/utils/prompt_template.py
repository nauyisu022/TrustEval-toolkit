def get_search_keyword(topic):
    return f"""
[System]
You are a professional Search Engine Optimization (SEO) expert, skilled at generating relevant search keywords for various topics.

[Background]
When using the Bing Search API, search keywords need to be provided in a specific format. Good search keywords should accurately reflect the topic content while aligning with common user search habits.

[Instruction]
1. I will give you a topic.
2. Please generate 1-3 relevant search keywords for this topic. 
3. These keywords should be words or phrases that people are most likely to use when searching for this topic.
4. Each keyword should be highly relevant to the topic and commonly used in searches.
5. Keywords can be single words or phrases, but should not exceed 5 words.
6. Please give as few search keywords as possible to enhance the accuracy of the search.

[Output Format]
Please output in the following format:

[[keyword1,...]]

Note:
- Do not include numbering.
- Each keyword should be surrounded by double square brackets.
- The entire output should be enclosed in a pair of square brackets.
- Keywords should be separated by commas.
- Please focus your search on the subject information provided by the user and the content related to the subject information, such as the description, definition, behavior or other of the subject information. 
- If the topic contains information such as "basic content of the webpage", this is an error caused when intercepting the user input, please ignore it.
Now, please generate search keywords for the following topic:
{topic}
    """

def summary_html_content(content):
    return f'''
[System]
You are a professional information extraction and summarization assistant, skilled at identifying and extracting key information from large volumes of text.

[Background]
The user has a web scraping script that can extract large amounts of text content from HTML web pages. Your task is to process this text, extracting all important information without over-condensing it, maintaining sufficient information density and context.

[Instruction]
1. Carefully read all the provided text content.
2. Identify and extract all key information, including but not limited to: main topics, important facts, statistical data, names, places, dates, events, etc.
3. Retain important details and background information; avoid oversimplification.
4. Organize the extracted information in a clear, easy-to-understand structure.
5. If there are contradictory or inconsistent pieces of information in the text, please point them out.
6. Maintain the core meaning of the original text without adding your own interpretations or speculations.

[Output Format]
Please output the extracted information in the following format:
[[Summary: Output Content]]

[User Input]
{content}
'''

def generate_jsonformat_prompt(instruction,basic_information,summary,jsonformat):
    return f'''
You are an AI assistant tasked with analyzing web content based on specific user requirements. Your job is to extract relevant information from a given web page summary according to provided themes and format the output as requested.

Input parameters:
1. Instruction: A partial instruction describing the task 
2. Basic Informations: A list of information or topics to focus on
3. Web Page Summary: A string containing a summary of the web page content
4. Output Format: A string containing a JSON format example and descriptions for each key

Your task:
1. Carefully read the instruction, basic information, and web page summary.
2. Identify and extract information from the web page summary that relates to the given information.
3. Organize the extracted information according to the provided output format.
4. Return a JSON object that follows the structure of the given output format, populated with the relevant information from the web page summary.

Please ensure that:
- You only include information that is directly related to the given information.
- You format the output exactly as specified in the output format example.
- If no relevant information is, include an "None" for that information in the JSON output.
- You do not add any additional commentary or explanation outside of the JSON structure.

Respond with the requested JSON object based on the provided inputs.

Here are the User Input:
Instruction: {instruction}
Basic Information: {basic_information}
Web Page Summary: {summary}
Output Format: {jsonformat}
'''

