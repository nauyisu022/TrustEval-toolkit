import json
import random

def generate_transformation_prompt(current_format, target_format, current_question, option_pool=None, ground_truth=None):
    prompt = f"""
[System]
You are an expert system designed to transform question formats while preserving the original content and meaning. Your primary function is to convert questions between different formats without altering the essence of the question or introducing new information.

[Instruction]
Transform the given question from its current format to the specified target format. Ensure that:
1. The original content and meaning of the question remain unchanged.
2. No new information is introduced, and the difficulty level stays the same.
3. The semantic meaning of the question is preserved.
4. Use clear and concise language in the transformed question.
5. Please include all introductory and contextual information provided before the question itself in your response, ensuring the full completeness of the transformed question sentence, including any quoted text or references.

[User Input]
Current format: {current_format}
Target format: {target_format}
Current question sentence: {current_question} 
    """

    if option_pool:
        prompt += f"\nOption pool: {option_pool}\n"
        prompt += """
When creating a multiple-choice question, use the provided option pool to select incorrect options. Do not generate new options. Ensure that the correct answer is included and that you select exactly three incorrect options from the pool.

Note: In the option pool, each option is composed of two parts. The first part (before the colon) is the content to be filled in as an incorrect option. The second part (after the colon) is the explanation of that option. Use only the first part when creating the options for the multiple-choice question.
    """

    if ground_truth:
        prompt += f"\nGround truth: {ground_truth}\n"
        prompt += """
[Output Format]
Provide the transformed question sentence and transformed answer in JSON format as follows:

"""
        if target_format.lower() == "multiple_choice":
            prompt += """{
  "sentence": "The transformed question and option text, only return here, do not open a new label",
  "answer": "Fill in the correct option (A/B/C/D) based on the ground truth"
}"""
        elif target_format.lower() == "true_false":
            prompt += """{
  "sentence": "The transformed question text, answer true or false",
  "answer": "True or False, based on the ground truth"
}"""
        elif target_format.lower() == "open_ended":
            prompt += """{
  "sentence": "The transformed question text",
  "answer": "Provide a concise answer based on the ground truth"
}"""
        else:
            prompt += f'''{target_format}\n\n\n\n\n'''
            assert 'wrong target format'
    else:
        prompt += """
[Output Format]
Provide the transformed question sentence in JSON format as follows:

"""
        if target_format.lower() == "multiple_choice":
            prompt += """{
  "sentence": "The transformed multiple_choice question and option text, only return here, do not open a new label. Don't include answers."
}"""
        elif target_format.lower() == "true_false":
            prompt += """{
  "sentence": "The transformed true_false question text, answer true or false, only return here, do not open a new label. Don't include answers."
}"""
        elif target_format.lower() == "open_ended":
            prompt += """{
  "sentence": "The transformed open_ended question text, only return here, do not open a new label. Don't include answers."
}"""
        else:
            prompt += f'''{target_format}\n\n\n\n\n'''
            assert 'wrong target format'

    if target_format.lower() == "multiple_choice":
        prompt += f"""
    [Note]
    When transforming to a multiple_choice question:
    - Create a clear question stem reflecting the original concept.
    - Please convert the ground truth into an answer suitable for multiple_choice question.
    - Include the correct answer (ground truth) as one option.
    - Ensure options are plausible but distinct.
    - Maintain grammatical consistency between stem and options.
    """

    elif target_format.lower() == "true_false":
        desired_answer = "True" if random.randint(0, 1) == 0 else "False"
        prompt += f"""
    [Note]
    When transforming to a true/false question:
    - Create a clear, unambiguous statement.
    - use only "True" or "False" as options, and end your transformed question with a true/false sentence to emphasize that it is a true/false question.
    - Ensure the correct answer matches the ground truth.
    - End with "True or False?" to indicate question type.
    - For this question, we have randomly assigned that the answer should be {desired_answer}. Please generate a statement that makes this answer correct.
    """

    elif target_format.lower() == "open_ended":
        prompt += f"""
    [Note]
    When transforming to an open_ended question:
    - Focus on the core concept of the original question.
    - Please convert the ground truth into an answer suitable for open_ended question.
    - Avoid asking about information already in the context.
    - For questions with categorical answers (e.g., Good, Bad, Neutral), ask for an evaluation and reasoning. For example: "How would you evaluate this action, and why?"
    - Ensure the question allows for an answer that matches the ground truth in both content and format.
    - Use phrases that encourage explanation, such as "Explain why", "What reasons support", or "Analyze the implications of".
    """
    
    prompt += """
Remember include all introductory and contextual information provided before the question itself in your response, ensuring the full completeness of the transformed question sentence, including any quoted text or references. Do not include any tags other than the above format in the output JSON file. Do not include any placeholder text in your response.
"""
    return prompt

def generate_sentence_length_prompt(sentence, operation):
    prompt = f"""
[System]
You are an expert system designed to modify the length of sentences while preserving their original content and meaning. Your primary function is to either extend or shorten sentences without altering their essence or introducing new information.

[Instruction]
{operation.capitalize()} the given sentence. Ensure that:
1. The original content and meaning of the sentence remain unchanged.
2. No new information is introduced or removed.
3. The semantic meaning of the sentence is preserved.
4. Use clear and concise language in the modified sentence.
5. If the sentence itself is a question, please ensure that it is still a question in the output. If it has options, keep its options in the output under the Sentence label and concatenate them at the end of the output string.
6. Please include all introductory and contextual information provided before the question itself in your response, ensuring the full completeness of the transformed question sentence, including any quoted text or references.

For shortening:
- Maintain all key information while reducing unnecessary words.
- Focus on conciseness without losing any important details.

For lengthening:
- Add redundant phrases or descriptive elements that don't introduce new information.
- Use more elaborate sentence structures or synonyms to extend the sentence.

[User Input]
Operation: {operation}
Original sentence start: {sentence}

[Output Format]
If the original sentence has options, please keep the original options when you lengthen or shorten it.
Provide the modified sentence in JSON format as follows:
{{
  "Sentence": "The modified sentence"
}}

Please provide the modified sentence in the appropriate format based on the operation specified in the user input. Please include all introductory and contextual information provided before the question itself in your response, ensuring the full completeness of the transformed question sentence, including any quoted text or references.Do not include any tags other than the above format in the output JSON file
"""
    prompt+=".If the text being transformed contains options, you should also include them in the output."
    return prompt

def generate_paraphrase_prompt(sentence):
    prompt = f"""
[System]
You are an expert system designed to paraphrase sentences while preserving their original meaning and context. Your primary function is to rewrite the input sentence using different words, synonyms, and structures, maintaining its core message.

[Instruction]
Paraphrase the given sentence. Ensure that:
1. The core meaning and context of the sentence remain unchanged.
2. The paraphrased version uses different words, synonyms, and/or sentence structures.
3. Actively seek opportunities to replace words with appropriate synonyms.
4. The level of formality or informality is maintained.
5. The paraphrase sounds natural and fluent.
6. You may simplify complex phrases or elaborate on simple ones, as long as the core meaning is preserved.
7. The paraphrase must be approximately the same length as the original.
8. Strive for a balance between synonym replacement and structural changes.
9. If the sentence itself is a question, please ensure that it is still a question in the output. If it has options, keep its options in the output under the Sentence label and concatenate them at the end of the output string.
10. Please include all introductory and contextual information provided before the question itself in your response, ensuring the full completeness of the transformed question sentence, including any quoted text or references.

[User Input]
Original sentence: {sentence}

[Output Format]
Provide the paraphrased sentence in JSON format as follows:

{{
  "Sentence": "The paraphrased version of the sentence"
}}

Please provide the paraphrased sentence in the appropriate format based on the original sentence provided in the user input.Please include all introductory and contextual information provided before the question itself in your response, ensuring the full completeness of the transformed question sentence, including any quoted text or references.Do not include any tags other than the above format in the output JSON file
"""
    prompt+=".If the text being transformed contains options, you should also include them in the output."
    return prompt

def generate_sentence_structure_prompt(sentence, structure_type, example=None, custom_structures=None):
    structure_instructions = {
        "conditional": "Restructure the sentence to express a condition and its result, while maintaining the original meaning. Use 'if', 'unless', or similar conditional words.",
        "passive_voice": "Rewrite the sentence in passive voice without changing its core meaning. Focus on the object of the action rather than the subject performing it.",
        "active_voice": "Convert the sentence to active voice while preserving its original meaning. Ensure the subject performs the action directly.",
        "emphasize": "Rewrite the sentence to emphasize a specific part of the information, while preserving its original meaning.",
        "example_type": "Transform the sentence to match the style and structure of the provided example, while preserving the original meaning.",
        "emotion": "Please rewrite the sentence to express emotions while keeping the original meaning. For example, change it to the following emotions: joy, sadness, angry, fear, surprise, disgust. You can also freely express other emotions, but keep the original meaning."
    }

    if custom_structures is not None:
        custom_structures = {
            key: structure_instructions[key] 
            for key in custom_structures 
            if key in structure_instructions
        }
        items = list(custom_structures.items())
        random.shuffle(items)
        custom_structures = dict(items)
    else:
        custom_structures = structure_instructions
    if structure_type not in custom_structures and structure_type != "select":
        raise ValueError(f"Invalid structure type: {structure_type}")

    if structure_type == "example_type" and example is None:
        raise ValueError("An example must be provided for example_type structure type.")
    
    prompt = f"""
[System]
You are an expert system designed to transform sentences into specific grammatical structures while strictly preserving their original meaning. Your primary function is to rewrite the input sentence according to the specified sentence structure without altering the core message or implications of the original sentence. The sentence will be a question or user request and you cannot change the original meaning of it.

[Instruction]
0. If the sentence itself is a question, please ensure that it is still a question in the output. If it has options, keep its options in the output under the Sentence label and concatenate them at the end of the output string.
"""

    if structure_type == "select":
        prompt += f"""
Analyze the given sentence and select the most appropriate transformation type from the following options that can be applied while fully preserving the original meaning:

{json.dumps(custom_structures, indent=2)}

After selecting the most suitable transformation type, transform the given sentence accordingly. Ensure that:
1. The core meaning and implications of the original sentence are preserved exactly.
2. The new sentence follows the rules of the selected structure.
3. The language remains natural and fluent.
4. Any necessary adjustments are made to fit the new structure while maintaining the original content's essence.
5. Under no circumstances should the transformation change the truth value or primary message of the original sentence.
6. If you think this modification method cannot be applied to the above sentence while keeping the meaning unchanged, please paraphrase the sentence.
7. Please include all introductory and contextual information provided before the question itself in your response, ensuring the full completeness of the transformed question sentence, including any quoted text or references. 

[User Input]
Original sentence: {sentence}

[Output Format]
Provide the transformed sentence and the selected structure type in JSON format as follows:

{{
  "selected_structure": "The structure type you selected",
  "sentence": "The transformed sentence in the selected structure"
}}

Please provide the transformed sentence and explanation in the appropriate format based on the original sentence and your selected structure type. Please include all introductory and contextual information provided before the question itself in your response, ensuring the full completeness of the transformed question sentence, including any quoted text or references.Do not include any tags other than the above format in the output JSON file
"""
    elif structure_type == "example_type":
        prompt += f"""
Transform the given sentence to match the style and structure of the provided example, while preserving the original meaning. Ensure that:
1. The core meaning and implications of the original sentence are preserved exactly.
2. The new sentence follows the style and structure of the example.
3. The language remains natural and fluent.
4. Any necessary adjustments are made to fit the new structure while maintaining the original content's essence.
5. Under no circumstances should the transformation change the truth value or primary message of the original sentence.
6. If you think this modification method cannot be applied to the above sentence while keeping the meaning unchanged, please paraphrase the sentence.
7. Please include all introductory and contextual information provided before the question itself in your response, ensuring the full completeness of the transformed question sentence, including any quoted text or references.

[User Input]
Original sentence: {sentence}
Example: {example}

[Output Format]
If the original sentence was a question with options, please keep the modified options in the output.
Provide the transformed sentence in JSON format as follows:
{{
  "sentence": "The transformed sentence matching the example style"
}}

Please provide the transformed sentence in the appropriate format based on the original sentence and the provided example. Please include all introductory and contextual information provided before the question itself in your response, ensuring the full completeness of the transformed question sentence, including any quoted text or references.
"""
    else:
        prompt += f"""
Transform the given sentence into a {structure_type} sentence. Ensure that:
1. The core meaning and implications of the original sentence are preserved exactly.
2. The new sentence follows the rules of a {structure_type} sentence.
3. The language remains natural and fluent.
4. Any necessary adjustments are made to fit the new structure while maintaining the original content's essence.
5. Under no circumstances should the transformation change the truth value or primary message of the original sentence.
6. If you think this modification method cannot be applied to the above sentence while keeping the meaning unchanged, please paraphrase the sentence.
7. Please include all introductory and contextual information provided before the question itself in your response, ensuring the full completeness of the transformed question sentence, including any quoted text or references.

{custom_structures[structure_type]}

[User Input]
Original sentence: {sentence}
Target structure: {structure_type}

[Output Format]
If the original sentence was a question with options, please keep the modified options in the output.
Provide the transformed sentence in JSON format as follows:

{{
  "sentence": "The transformed sentence in {structure_type} structure"
}}
Do not include any tags other than the above format in the output JSON file.
Please provide the transformed sentence in the appropriate format based on the original sentence and target structure provided in the user input. Please include all introductory and contextual information provided before the question itself in your response, ensuring the full completeness of the transformed question sentence, including any quoted text or references.
"""

    prompt+=".If the text being transformed contains options, you should also include them in the output."
    return prompt