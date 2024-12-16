# Changelog

All notable changes to this project will be documented in this file.

## 2024-09-25
### Added
- Multiple images in annotation platform.
- Modelservice class supported multiple images in VLM.
- Add `extra_instructions` key in dataset submission's `file_config.json` if needed.

### Fixed
- Resolved `show_status` module error in annotation platform.



# Code Submission Guidelines 
## Project Structure

```python
project_root/
│
├── src/                    # Main code directory
│   ├── __init__.py         # Makes src a package
│   ├── utils.py            # Utility functions
│   ├── saver.py            # Data saving class
│   ├── data_processing/    # Data processing module
│   │   ├── __init__.py
│   │   ├── cleaner.py      # Data cleaning
│   │   └── craft.py        # Core data processing logic
│
├── cache/                  # Intermediate data storage
│
├── fairness/               # Contributor's submitted code directory (decided to your task)
├── ethics/
├── .../
│
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── main.py                 # Main program entry point
```

## Guidelines for Code Submission

### Import Statements

To avoid import errors, always use absolute imports when importing modules from the `src` directory. For example:

```python
from src.utils import some_function
from src.data_processing.cleaner import clean_data
```

### Using the `Saver` Class

We have a unified class `Saver` in `src/saver.py` for saving intermediate and final datasets. This class ensures that all data is stored in a consistent manner and in a specified directory. The `Saver` class takes an absolute path during initialization, and all saved files will be stored in this base directory.

### Example Directory Structure for Saved Files

When using the `Saver` class, ensure that your intermediate and final datasets are stored in specific directories. For example:

```python
project_root/
│
├── fairness/              # Data storage directory
│   ├── intermediate/      # Intermediate data
│   │   └── module1/       # Intermediate data for module1
│   │       ├── file1.json
│   │       └── file2.csv
│   └── final/             # Final datasets
│       ├── module1/       # Final data for module1
│       │   ├── result1.json
│       │   └── result2.json
│
└── ...
```

### Example Usage of the `Saver` Class

Below is a detailed example showing how to use the `Saver` class to save data in different formats:

```python
import os
from src.saver import Saver

# Initialize Saver with the base directory for intermediate files
intermediate_base_path = os.path.abspath('fairness/intermediate/module1')
saver = Saver(intermediate_base_path)

# Save intermediate JSON data
json_data = {"theme": "dark", "notifications": True}
json_intermediate_path = 'intermediate_data.json'
saver.save_data(json_intermediate_path, json_data)

# Initialize Saver with the base directory for final datasets
final_base_path = os.path.abspath('fairness/final/module1')
saver = Saver(final_base_path)

# Save intermediate JSON data
json_data = {"theme": "dark", "notifications": True}
json_intermediate_path = 'intermediate_data.json'
saver.save_data(json_intermediate_path, json_data)

# Read file example
read_data = saver.read_file(json_intermediate_path)
print(read_data)
```

### Example Module in `fairness/`

Here is an example module located in the `fairness/` directory that demonstrates how to use the `Saver` class:

```python
# your_submission_code//example_module.py

import os
from src.saver import Saver

def main():
    # Initialize Saver with the base directory for intermediate files
    intermediate_base_path = os.path.abspath('fairness/intermediate/example_module')
    saver = Saver(intermediate_base_path)

    # Save some intermediate data
    data = {"step": 1, "status": "processing"}
    saver.save_data('intermediate_step1.json', data)

    # Processing logic here...

    # Initialize Saver with the base directory for final datasets
    final_base_path = os.path.abspath('fairness/final/example_module')
    saver = Saver(final_base_path)

    # Save final results
    result = {"step": 1, "status": "completed"}
    saver.save_data('final_result.json', result)

if __name__ == "__main__":
    main()
```

## Contact

If you have any questions or need further clarification, please contact [Yanbo Wang](wyf23187@gmail.com).




# Data Crafter
## Environment Variables

Before running the scripts, ensure you have set the necessary environment variables in the `.env` file located in the `src/metadata_curator/` directory:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE_URL=your_openai_api_base_url // Initially set to https://api.openai.com/
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint // Initially set to https://trustagi2.openai.azure.com/
BING_SEARCH_V7_SUBSCRIPTION_KEY=your_bing_search_v7_subscription_key
BING_SEARCH_V7_ENDPOINT=your_bing_search_v7_endpoint // Initially set to https://api.bing.microsoft.com/
```

## Running the Unified Pipeline

The `pipeline.py` script automates the process of web searching and result processing. It extracts keywords from user input, performs a Bing search, and processes the results into JSON format.

### Parameters

When running the unified pipeline, you need to provide several parameters to control its behavior:

- **instruction**: A string that specifies the user's instruction for what to find on the web pages.
- **basic information**: A dictionary that defines the specific information for the search.
- **need_azure**: A boolean that indicates whether to use the Azure API for generating responses.
- **output_format**: A dictionary that outlines the structure of the expected results.
- **keyword_model**: The model to use for generating keywords (e.g., "gpt-4o").
- **response_model**: The model to use for generating responses (e.g., "gpt-4o").
- **include_url**: A boolean that indicates whether to include the URL of the web pages in the output.
- **include_summary**: A boolean that indicates whether to include a summary of the web pages in the output.
- **include_original_html**: A boolean that indicates whether to include the original HTML of the web pages in the output.
- **include_access_time**: A boolean that indicates whether to include the access time of the web pages in the output.
- **output_file**: A string that specifies the name of the output file where the results will be saved.

- **direct_search_keyword (optional)**: Direct keyword for the search. If provided, this keyword will be used directly. 
- **direct_site (optional)**: Specific site to search within. If provided, the search will be limited to this site. If there are multiple specified websites, please separate them with commas, for example`"www.wikipedia.com,www.nytimes.com"`

### Key Workflow Description in the TextWebSearchPipeline
#### User Input Construction
The first step involves constructing user input by combining the user's `instruction` and `basic information`. The basic information, represented as a dictionary, is transformed into a string format. This combined input is then used to generate search keywords through the `get_search_keyword` prompt template. After that, using the generated search keywords, a search is performed in Bing to retrieve relevant web pages.
**Example：**
Suppose we have the following input:
- `instruction`: "Please summarize the search results."
- `basic_information`: 
```python
    {
        "Topic": "Artificial Intelligence",
        "Year": "2024",
        "Region": "Global"
    }
```
The `basic_information` would be processed as:`"Topic is Artificial Intelligence. Year is 2024. Region is Global."`
The final `user_input` would be:`"Please summarize the search results. Topic is Artificial Intelligence. Year is 2024. Region is Global."`
Then we use this `user_input` to generate the search keyword

#### Summarizing Web Content
The search results are further processed using the `process_search_results` prompt template, which enables a LLM to create summaries for the content of each web page. This results in summaries that are comprehensible to the model, drawn from the original HTML text of each web page.

#### Generating Responses
With the user's instruction, basic information, output format, and the corresponding web page summaries, the `generate_jsonformat_prompt` prompt template is used to generate structured responses from the LLM. 
Here is our prompt template for generating structured responses:
```markdown
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
```

#### Response Cleaning and Storage
The generated responses are then cleaned to ensure they are in valid JSON format before being stored in the `responses` attribute of the class.

#### Merging Responses
Finally, the various responses are merged together into a cohesive output format.

### Example

Here's how to use the pipeline with an example script (already in `src/metadata_curator/run.py`):

```python
import asyncio
from pipeline import TextWebSearchPipeline

def main():
    """
    Main function to run the TextWebSearchPipeline with specified parameters.
    """
    # Define the instruction and information for the pipeline
    instruction = "Please find examples of unfair treatment that match the given information."
    basic_information = {
        "Gender" : "Female",
        "Nation" : "United States",
    }

    # Define the user-specific dictionary for response formatting
    output_format = {
        "Example": [
            "Specific example 1 mentioned on the webpage",
            "Specific example x mentioned on the webpage (and so on)"
        ]
    }

    # Specify the output file path
    output_path = "a.json"

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
    main()
```

### Output Structure

The generated JSON file will have the following structure:

```json
[
    {
        "Example": [
            "25% of working women have earned less than male counterparts for the same job, while only 5% of working men report earning less than female peers.",
            "Women are four times more likely than men to feel treated as incompetent due to gender (23% vs. 6%).",
            "16% of women report experiencing repeated small slights at work due to their gender, compared to 5% of men.",
            "15% of working women say they received less support from senior leaders than male counterparts; only 7% of men report similar experiences.",
            "10% of working women have been passed over for important assignments due to gender, compared to 5% of men.",
            "22% of women have personally experienced sexual harassment compared to 7% of men.",
            "53% of employed black women report experiencing gender discrimination, compared to 40% of white and Hispanic women.",
            "22% of black women report being passed over for important assignments due to gender, compared to 8% of white and 9% of Hispanic women."
        ],
        "url": "https://www.pewresearch.org/short-reads/2017/12/14/gender-discrimination-comes-in-many-forms-for-todays-working-women/",
        "summary": "[[Summary: \n\n**Main Topic: Gender Discrimination in the Workplace**\n\n1. **Prevalence of Discrimination:**\n   - Approximately 42% of working women in the U.S. report experiencing gender discrimination at work.\n   - Women are about twice as likely as men (42% vs. 22%) to report experiencing at least one of eight specific forms of gender discrimination.\n\n2. **Forms of Discrimination:**\n   - 25% of working women have earned less than male counterparts for the same job, while only 5% of working men report earning less than female peers.\n   - Women are four times more likely than men to feel treated as incompetent due to gender (23% vs. 6%).\n   - 16% of women report experiencing repeated small slights at work due to their gender, compared to 5% of men.\n   - 15% of working women say they received less support from senior leaders than male counterparts; only 7% of men report similar experiences.\n   - 10% of working women have been passed over for important assignments due to gender, compared to 5% of men.\n\n3. **Sexual Harassment:**\n   - 36% of women and 35% of men believe sexual harassment is a problem in their workplace; however, 22% of women have personally experienced it compared to 7% of men.\n   - Variability in reports of sexual harassment exists depending on survey questions.\n\n4. **Differences by Education:**\n   - Women with a postgraduate degree report higher rates of discrimination compared to those with less education: 57% vs. 40% (bachelor\u2019s degree) and 39% (less than bachelor\u2019s).\n   - 29% of women with postgraduate degrees experience repeated small slights compared to 18% (bachelor\u2019s) and 12% (less education).\n\n5. **Income Disparities:**\n   - 30% of women with family incomes of $100,000 or higher report earning less than male counterparts, compared to 21% of women with lower incomes.\n\n6. **Racial and Ethnic Differences:**\n   - 53% of employed black women report experiencing gender discrimination, compared to 40% of white and Hispanic women.\n   - 22% of black women report being passed over for important assignments due to gender, compared to 8% of white and 9% of Hispanic women.\n\n7. **Political Party Differences:**\n   - 48% of working Democratic women report experiencing gender discrimination, compared to one-third of Republican women.\n\n8. **Survey Details:**\n   - The survey was conducted from July 11 to August 10, 2017, with a representative sample of 4,914 adults, including 4,702 employed adults.\n   - The margin of error is \u00b12.0 percentage points for the total sample and \u00b13.0 for employed women.\n\n**Authors:** Kim Parker and Cary Funk, Pew Research Center. \n**Publication Date:** December 14, 2017.]]",
        "access_time": "2024-08-10T05:21:17.497674"
    },
    {
        // There will be multiple such blocks for different search results.
    }
]
```

## Running the imageSearchPipeline
The `imageSearchPipeline.py` script automates the process of extracting keywords from user input, performing a Bing image search, and processing the results into JSON format.

### Parameters

When running the pipeline, you need to provide several parameters to control its behavior:

- **instruction**: A string that specifies the user's instruction for what kind of images to find.
- **basic_information**: A dictionary that defines the specific information for the search (e.g., breed of dog, age, etc.).
- **output_path**: (Optional) A string that specifies the name of the output file where the results will be saved. Defaults to "processed_image_results.json".
- **keyword_model**: The model to use for generating keywords (e.g., "gpt-4o").
- **include_access_time**: A boolean that indicates whether to include the access time of the web pages in the output. Defaults to `True`.

- **direct_search_keyword (optional)**: Direct keyword for the search. If provided, this keyword will be used directly. 

### Key Workflow Description in the imageSearchPipeline

#### Keyword Extraction
The first step involves constructing user input by combining the user's `instruction` and `basic_information`. This combined input is then used to generate search keywords through an OpenAI model.

#### Image Search
The script performs image searches on Bing using the extracted keywords and the Bing Search API.

#### Processing Search Results
The search results are processed to extract relevant details such as image name, content URL, thumbnail URL, host page URL, encoding format, and date published.

#### Saving to JSON
The processed results are saved to a specified JSON file.

### Example

Here's how to use the `imageSearchPipeline.py` with an example script:

```python
import asyncio
from ImageWebSearchPipeline import ImageWebSearchPipeline

def main():
    """
    Main function to run the ImageWebSearchPipeline with specified parameters.
    """
    # Define the instruction and basic information for the pipeline
    instruction = "Find images of cute puppies"
    basic_information = {"breed": "Golden Retriever", "age": "2 months"}

    # Use default output path
    pipeline = ImageWebSearchPipeline(instruction, basic_information)
    asyncio.run(pipeline.run())

    # Use custom output path
    custom_output_path = "custom_puppy_images.json"
    pipeline_with_custom_output = ImageWebSearchPipeline(instruction, basic_information, output_path=custom_output_path)
    asyncio.run(pipeline_with_custom_output.run())

if __name__ == "__main__":
    main()
```

### Output Structure

The generated JSON file will have the following structure:
```json
[
    {
        "name": "Image Name",
        "contentUrl": "https://example.com/contentUrl", //The original image URL is large and may be inaccessible.
        "thumbnailUrl": "https://example.com/thumbnailUrl",//The thumbnail URL generated by bing search can theoretically be accessed directly
        "hostPageUrl": "https://example.com/hostPageUrl",//Original URL of the webpage where the image is located
        "encodingFormat": "jpg",
        "datePublished": "2024-08-11T14:57:45.000Z",//Image published time
        "accessTime": "2024-08-11T14:57:45.000Z"  
    },
    {
        // There will be multiple such blocks for different search results.
    }
]
```



# Diversity Enhancer

Diversity Enhancer 是一个 Python 类，旨在通过多种方式增强文本的多样性。它提供了对有固定format (选择题、开放性问答或者判断题) 以及没有固定fomat的一系列enhance的操作，下面来进行详细介绍。

## 使用方法

首先，导入 ContextualVariator 类：

```python
from contextual_variator import ContextualVariator
```

然后，创建一个 ContextualVariator 实例：

```python
enhancer = ContextualVariator()
```

ContextualVariator 初始化的时候需要接受一个可选参数，这个可选参数是一个list，包含了所有在你的query上支持的操作方式（你可以自己指定，也可以不指定），以此来限制对query的处理。如果没有指定的话，默认初始化为下面的列表（所有的操作方式），并会进行随机选择一种策略进行enhance。

```python
supported_operations = [
    "transform_expression",
    "paraphrase_sentence",
    "modify_sentence_length",
    "transform_to_multiple_choice",
    "transform_to_true_false",
    "transform_to_open_ended"
]
```
您也可以只让他支持部分的操作
```python
operations = [
    "transform_to_multiple_choice"
]
enhancer = ContextualVariator(operations)
```

### 0. 注意事项

#### 基本用法

- 固定format是指你的原始query/sentence是 `["multiple_choice", "true_false", "open_ended"]`的其中之一，此时你在使用enhance方法时必须提供原始query的current format 作为`current_format`的参数。`answer`参数是可选提供的，如果提供就会根据问题形式的变化给出新问题下的answer，如果未提供就不会返回`answer`键。
- 非固定format就只用提供原始query/sentence即可

**固定format支持的操作**

```python
format_operations = [
    "transform_expression",
    "paraphrase_sentence",
    "modify_sentence_length",
    "transform_to_multiple_choice",
    "transform_to_true_false",
    "transform_to_open_ended"
]
```

**非固定format支持的操作**

```python
non_format_operations = [
    "transform_expression",
    "paraphrase_sentence",
    "modify_sentence_length"
]
```

使用 `enhance_diversity` 方法：

enhance方法针对你在实例化ContextualVariator时设置的方式来进行参数传递，如果你包含了固定format专属的三种操作`[transform_to_multiple_choice","transform_to_true_false","transform_to_open_ended"]`，你必须指定**current_format**以及**answer**两个参数。否则只会调用`"transform_expression", "paraphrase_sentence", "modify_sentence_length"`三种通用enhance方法。
- 同时我们可以传入一个keep_original参数，默认为`True`，如果为`True`的话则有与其他操作等概率出现一个操作方法为keep_original的操作，输出的结果与原来保持不变。为`False`则关闭这个操作。

`e.g`

```python
import json
from contextual_variator import ContextualVariator
enhancer = ContextualVariator()
async def main():

    # non_format query.sentence
    result = await enhancer.enhance_diversity("This is a test sentence.")
    print(json.dumps(result, indent=4))


    # offer 'current_format' and 'answer'
    result = await enhancer.enhance_diversity(
        "What is the capital of France?", 
        current_format="Open ended question",
        answer="Paris"
    )

    print(json.dumps(result, indent=4))

    # only offer 'current_format'
    result = await enhancer.enhance_diversity(
        "What is the meaning of life?", 
        current_format="Open ended question",
    )
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

```

`output`

```json
{
    "sentence": "I can't believe this is just a test sentence!",
    "structure_type": "emotion",
    "enhancement_method": "transform_expression"
}
{
    "sentence": "What is the capital of France?\nA) Berlin\nB) Madrid\nC) Paris\nD) Rome",
    "answer": "Paris",
    "format": "Multiple choice question",
    "enhancement_method": "transform_to_multiple_choice"
}
{
    "sentence": "What is the meaning of life? a) Happiness b) Success c) 42 d) Love",
    "format": "Multiple choice question",
    "enhancement_method": "transform_to_multiple_choice"
}
```

#### 参数说明

- `sentence`: 要增强多样性的句子或问题（字符串）
- `current_format`: 当前问题的格式（可选，仅在处理固定format时需要）。对于问题格式转换，必须提供 `current_format` 参数
- `answer`: 可选，如果当前任务为一个问题，如果存在的话可以给 `answer`（ground truth），输出的时候也会将 ground truth 转化为对应的问题形式下的答案
- `extra_instructions` 可选，如果你想要给模型一些额外的指导，可以传入这个参数，这个参数是一个字符串，默认为空
#### 输出格式
所有输出的 JSON 结构与每个详细输出结构类似，可以参考下文1、2、3、4小节中的输出部分，只是会多出一个 `enhancement_method` 来说明随机选择了哪种方法来增强多样性。

### 1. 转换句子结构
`transform_expression` 函数可以将输入的句子转换为不同的结构或风格，同时保持原始含义。

#### 函数参数
- `sentence`: 要转换的原始句子 (必需)
- `structure_type`: 指定转换的结构类型 (必需)
- `example`: 用于 "example_type" 结构类型的示例句子 (可选)
- `custom_structures`: 自定义结构类型字典 (可选)

#### 使用方法
1. LLM自动选择模式(默认模式,推荐) 
使用 "select" 模式,让模型从可用结构中选择最合适的类型。此时会随机将可用模型列表中的**一半**在**打乱顺序**后传给LLM,让它自行决定哪个修改方式更合适。
```python
result = await enhancer.transform_expression(sentence, structure_type="select")
```
2. 随机转换 
使用 "random" 模式,函数会从默认的9种句式类型中随机选择一种进行转换。
```python
result = await enhancer.transform_expression(sentence,structure_type="random")
```
3. 指定结构类型
通过 `structure_type` 参数指定特定的转换类型。
```python
result = await enhancer.transform_expression(sentence, structure_type="passive_voice")
```
4. 示例类型转换
使用 `example_type` 结构类型,并提供 `example` 参数。
```python
result = await enhancer.transform_expression(sentence, structure_type="example_type", example="As the saying goes, 'The early bird catches the worm.'")
```
5. 自定义结构列表
提供 `custom_structures` 参数来使用自定义的结构类型。
```python
custom_structures = {
    "metaphor": "Rewrite the sentence as a metaphor that conveys the same meaning.",
    "alliteration": "Rewrite the sentence using alliteration while preserving its meaning.",
    "personification": "Rewrite the sentence using personification to convey the same idea."
}
result = await enhancer.transform_expression(sentence,  structure_type="select",custom_structures=custom_structures)
```


默认结构类型:
- declarative
<!-- - rhetorical_question -->
- imperative
- conditional
- passive_voice
- active_voice
<!-- - double_negative -->
- emphasize
- emotion

#### 输出格式
函数返回一个包含转换后句子和结构类型的JSON对象:
```json
{
  "sentence": "转换后的句子",
  "structure_type": "使用的结构类型"
}
```
对于 "select" 模式,输出还会包含所选的结构类型:
```json
{
  "selected_structure": "模型选择的结构类型",
  "sentence": "转换后的句子",
  "structure_type": "select"
}
```
注意事项: 

- 使用 "example_type" 时必须提供 `example` 参数。
- 提供 `custom_structures` 时,随机选择将从这些自定义结构中进行,而非默认结构。
- "select" 模式会自动选择最合适的结构类型,并在输出中指明。
- 所有转换都会尽可能保持原始句子的核心含义。

### 2. 改写句子

```python
sentence = "Life is like a box of chocolates."
result = await enhancer.paraphrase_sentence(sentence)
print(result)
```

#### 输出格式：
```json
{
  "Sentence": "Life resembles a box of chocolates; you never know what you're going to get."
}
```

### 3. 修改句子长度

```python
sentence = "The quick brown fox jumps over the lazy dog."
result = await enhancer.modify_sentence_length(sentence)
print(result)

# 指定增加或减少长度
result = await enhancer.modify_sentence_length(sentence, "lengthen")
print(result)
```
可选的长度修改操作：
- "lengthen"
- "shorten"

#### 输出格式：
```json
{
  "Sentence": "The swift and agile brown fox gracefully leaps over the indolent and sluggish canine.",
  "operation": "lengthen"
}
```

### 4. 转换问题格式

```python
current_format = "Multiple choice question"
current_question = "What is the capital of France? a) Berlin b) Madrid c) Paris d) Rome"
answer="c) Paris" #可选，如果存在ground truth则可以传入，没有可以省略。

# 随机选择新的问题格式
result = await enhancer.transform_question_format(current_format, current_question=current_question, answer=answer)
print(result)

# 指定特定的问题格式
result = await enhancer.transform_question_format(current_format, target_format="True/False question", current_question=current_question, answer=answer)
print(result)
```

可选的问题格式：
- "Multiple choice question"
- "True/False question"
- "Open ended question"

#### 输出格式：
```json
{
    "sentence": "What is the capital of France?",
    "answer": "Paris.",
    "format": "Open ended question"
}

{
  "sentence": "What is the capital of France? a) Berlin b) Madrid c) Paris d) Rome",
  "format": "Multiple choice question",
  "answer": "Paris"
}

{
  "sentence": "Paris is the capital of France. True or False",
  "answer": "True",
  "format": "True/false question"
}
```
当你选择随机选择格式的时候，随机的问题格式会是非当前格式的另外两种形式之一。
如果你没有输入 ground truth，则输出也不会包含 `answer`。

## 注意事项

- 所有方法都是异步的，需要在异步环境中使用（使用 `async/await`）。
- 当不指定特定类型时，方法会随机选择一个可用的转换类型。



# ModelService Class README

## Overview
The `ModelService` class provides a flexible and powerful interface for interacting with various language models. It supports synchronous and asynchronous requests, as well as handling multi-turn conversations. This document outlines the usage, input formats, and output formats of the `ModelService` class.

## Features
- **Synchronous and Asynchronous Processing**: The class can handle both synchronous and asynchronous requests.
- **Multi-Turn Conversations**: The service supports multi-turn conversations, maintaining the context of the conversation history.
- **Customizable Request and Handler Types**: The class allows customization of request and handler types to suit various deployment scenarios.
- **Extensible for Future Implementations**: The class is designed to be extensible, with plans to include text-to-image (t2i) and local deployment functionalities in future updates.

## Usage

### Initialization
Initialize the `ModelService` with the required parameters:
```python
from generation import ModelService

service = ModelService(
    request_type='llm',         # The type of request (e.g., 'llm' for language models)
    handler_type='api',         # The type of handler (e.g., 'api' for API-based requests)
    model_name='gpt-4o',        # The name of the model to use
    config_path='src/config/config.yaml',  # Path to the configuration file
    **kwargs                    # Additional keyword arguments
)
```

### Synchronous Processing
#### Single Prompt
To process a single prompt synchronously:
```python
response = service.process("What is the capital of France?")
print(response)
```

#### Multi-Turn Conversation
To process multiple prompts in a conversation synchronously:
```python
prompts = [
    "Hello, who are you?",
    "What can you do?",
    "Tell me a joke."
]
responses = service.process(prompts)
for response in responses:
    print(response)
```

### Asynchronous Processing
#### Single Prompt
To process a single prompt asynchronously:
```python
import asyncio

async def main():
    response = await service.process_async("What is the capital of France?")
    print(response)

asyncio.get_event_loop().run_until_complete(main())
```

#### Multi-Turn Conversation
To process multiple prompts in a conversation asynchronously:
```python
import asyncio

async def main():
    prompts = [
        "Hello, who are you?",
        "What can you do?",
        "Tell me a joke."
    ]
    responses = await service.process_async(prompts)
    for response in responses:
        print(response)

asyncio.get_event_loop().run_until_complete(main())
```

### Concurrent Processing
To apply a function concurrently to a list of elements with a maximum concurrency limit:
```python
   import asyncio, json
   from generation import ModelService, apply_function_concurrently
   from typing import List, Dict, Any

    # process_element function is defined by you in step 1.
    async def process_element(element: Dict[str, Any]) -> Dict[str, Any]:
        # Example operation involving asynchronous processing, such as an API call
        text = element.get("prompt", "")
        # your customized asynchronous logic here
        result = await some_async_api_call(text)
        element["result"] = result
        return element
    
   async def main():
       with open('path_to_your_dataset.json', 'r', encoding='utf-8') as file:
           elements = json.load(file)
       results = await apply_function_concurrently(
           process_element, elements, max_concurrency=5
       )
       with open('path_to_generation_result.json', 'w') as file:
           json.dump(results, file, indent=4)

   if __name__ == "__main__":
       asyncio.get_event_loop().run_until_complete(main())

```

## Input and Output Formats
### Input
- **Single Prompt**: A single string.
- **Multi-Turn Conversation**: A list of strings, where each string is a user prompt.

### Output
- **Single Prompt**: A single response string.
- **Multi-Turn Conversation**: A list of response strings corresponding to each user prompt.

## Code & Data submission format

1. **Provide the `process_element` Function: (Your work)**

   Define the `process_element` function that specifies how each data should be processed. You should customize this function based on your requirements. For example:

   ```python
    async_service = ModelService(
        request_type='llm',
        handler_type='api',
        model_name='gpt-4o',
        config_path='src/config/config.yaml'
    )
    
    async def process_element(element: Dict[str, Any]) -> Dict[str, Any]:
        """
        This function processes an element (a dictionary) 
        It performs an asynchronous operation (e.g., an API call) and then stores the result in the element.
        Args:
            element (Dict[str, Any]): A dictionary that contains the data to be processed. 

        Returns:
            Dict[str, Any]: The processed element with the 'result' field added after modification.
        """
        # Example operation involving asynchronous processing, such as an API call
        text = element.get("prompt", "")
        # Replace this with your customized asynchronous logic
        result = await async_service.process_async(text)
        element["result"] = result
        return element
   ```

2. **Prepare Your Data: (Your work)**

   Ensure your data is in a JSON file where each element is a dictionary. For example:

   ```json
   [
       {"prompt": "What is the capital of France?", "options": ["Paris", "London", "Berlin", "Madrid"]},
       {"prompt": "What is 2 + 2?", "options": ["3", "4", "5", "6"]}
   ]
   ```

3. **Run the Main Script: (Coding team's work)**

   The main script reads your JSON file, processes the elements using the `process_element` function, and writes the enhanced elements to a new JSON file. For example:

   ```python
   import asyncio, json
   from generation import ModelService, apply_function_concurrently
   from typing import List, Dict, Any

    # process_element function is defined by you in step 1.
    async def process_element(element: Dict[str, Any]) -> Dict[str, Any]:
        # Example operation involving asynchronous processing, such as an API call
        text = element.get("prompt", "")
        # your customized asynchronous logic here
        result = await some_async_api_call(text)
        element["result"] = result
        return element
    
   async def main():
       with open('path_to_your_dataset.json', 'r', encoding='utf-8') as file:
           elements = json.load(file)
       results = await apply_function_concurrently(
           process_element, elements, max_concurrency=5
       )
       with open('path_to_generation_result.json', 'w') as file:
           json.dump(results, file, indent=4)

   if __name__ == "__main__":
       asyncio.get_event_loop().run_until_complete(main())
   ```

4. **Customize as Needed: (Optional)**

   If you have any special requirements or need to adjust the format, please contact the coding team for assistance.


## Future Work
The following features are currently under development and will be included in future updates:
- **Text-to-Image (t2i) Functionality**: Support for generating images from text descriptions.
- **Local Deployment**: Options for running models locally without API calls.

For any issues, questions, or special requirements, please contact the coding team. Stay tuned for more updates!


# Human Annotation Platform

This project offers a platform for human annotation of textual and image data using Streamlit. The platform allows users to upload JSON data files, configure the annotation tasks, and annotate each item with predefined options and feedback.

## Features

- **Configuration Page:** Upload JSON data files and configure the keys to display.
- **Text Annotation Platform:** Annotate textual data with predefined options and provide feedback.
- **Image Annotation Platform:** Annotate images with predefined options and provide feedback.
- **Automatic Saving:** Annotations are automatically saved after each change.
- **Overall Status:** View the overall status of annotations, including counts and percentages for each option.

## Project Structure
```
.
├── src
│   ├── annotation
│   │   ├── annotation.css
│   │   └── annotation.py
│   ├── config
│   │   └── annotation_config.yaml
├── data
│   └── {dataset_name}
│       ├── images
│       └── {dataset_name}_annotation.json
|       └── {dataset_name}.json
└── README.md
```

- **src/annotation/annotation.css:** Custom CSS for the annotation interface.
- **src/config/annotation_config.yaml:** Configuration file specifying the annotation options.
- **data/{dataset_name}/images: (Your task)** Directory to store image files for image annotation, where `{dataset_name}` is the name of your dataset.
- **data/{dataset_name}/{dataset_name}.json: (Your task)** JSON file containing the dataset to be annotated.
- **data/{dataset_name}/{dataset_name}_annotation.json:** JSON file to store annotations for the dataset.

## Annotation Data Format

The annotation platform expects the data in a JSON file with each item being a dictionary. Example JSON format:

```json
[
    {
        "text": "Sample text 1",
        "image_path": "image1.jpg"
    },
    {
        "text": "Sample text 2",
        "image_path": "image2.jpg"
    }
]
```

### Key Requirements:

- For image annotation, include an `image_path` key with the path to the image file relative to the `data/{dataset_name}/images` directory.

## Configuration File

The `annotation_config.yaml` file defines the options for annotation. Example:

```yaml
Semantic shift:
  - True
  - False

Quality assessment:
  - True
  - False
```

## Usage

1. **Running the Application:**
   
   To start the Streamlit application, run:

   ```bash
   streamlit run src/annotation/annotation.py --server.enableXsrfProtection=false
   ```

2. **Upload JSON Data File:**

   On the Configuration page, upload your JSON data file. The platform will create an annotation file if it doesn't exist, or load an existing one.

3. **Select Keys:**

   Choose the keys you want to display for annotation from the loaded data. Click "Load Selected Keys" to initialize the annotation state.

4. **Annotate Items:**

   Navigate to the "Text Annotation Platform" or "Image Annotation Platform" to start annotating the items. Use the provided options and feedback text area to annotate each item.

5. **Save Annotations:**

   Annotations are automatically saved after each change.

6. **Check Status:**

   Use the "Show Status" button to display the overall annotation status, including counts and percentages for each option.

## Notes

- Ensure that the image files are placed in the `data/{dataset_name}/images` directory and the paths in the JSON file are correctly specified.
- The platform supports multiple annotation categories as defined in the configuration file.
- Annotations are saved automatically, providing a smooth and efficient annotation process.

We release a video tutorial on how to use the platform for detailed instructions, which url is [here](https://drive.google.com/file/d/17ra2e2mTkp4C_lw9WHMPXqDoBsWLJ1A4/view?usp=sharing). For any issues or feature requests, please contact Yanbo Wang.


