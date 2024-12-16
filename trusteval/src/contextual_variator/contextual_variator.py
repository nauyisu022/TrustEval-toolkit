from dataclasses import dataclass
import sys
import os
import yaml
import random
import asyncio
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .prompt_template import (
    generate_sentence_structure_prompt,
    generate_paraphrase_prompt,
    generate_sentence_length_prompt,
    generate_transformation_prompt
)
from generation.model_service import ModelService
from .clean_json import clean_json

@dataclass
class Format:
    name: str

    def __post_init__(self):
        self.name = self.name.lower()

    def __eq__(self, other):
        if isinstance(other, Format):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other.lower()
        return NotImplemented

    def __str__(self):
        return self.name

def lowercase_keys(obj):
    if isinstance(obj, dict):
        return {k.lower(): lowercase_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [lowercase_keys(elem) for elem in obj]
    else:
        return obj

class ContextualVariator:
    def __init__(self, supported_operations=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../"))
        self.config_file_path = os.path.join(project_root, "config", "config.yaml")

        with open(self.config_file_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.all_operations = [
            "transform_expression",
            "paraphrase_sentence",
            "modify_sentence_length",
            "transform_to_multiple_choice",
            "transform_to_true_false",
            "transform_to_open_ended"
        ]

        if supported_operations is None:
            self.supported_operations = self.all_operations
        else:
            self.supported_operations = [op for op in supported_operations if op in self.all_operations]

        #print("Initialized supported operations:", self.supported_operations)

    async def _get_model_service(self):
        return ModelService(
            request_type="llm",
            handler_type="api",
            model_name="gpt-4o",
            config_path=self.config_file_path,
            temperature=0.7,
            max_tokens=2048
        )

    async def enhance_diversity(self, sentence, current_format=None, answer=None, keep_original=True, extra_instructions=None):
        #print("Self supported operations:", self.supported_operations)
        if current_format:
            available_operations = self.supported_operations.copy()
        else:
            available_operations = [op for op in self.supported_operations if not op.startswith("transform_to_")]

        if keep_original:
            available_operations.append("keep_original")

        #print("Available operations:", available_operations)

        if not available_operations:
            print("No supported operations available.")
            return None

        max_attempts = 3
        for attempt in range(max_attempts):
            chosen_operation = random.choice(available_operations)
            #print(f"Attempt {attempt + 1}, chosen operation: {chosen_operation}")

            result = None
            try:
                if chosen_operation == "transform_expression":
                    chosen_operation = "keep_original"
                if chosen_operation == "paraphrase_sentence":
                    result = await self.paraphrase_sentence(sentence, extra_instructions=extra_instructions)
                elif chosen_operation == "modify_sentence_length":
                    result = await self.modify_sentence_length(sentence, extra_instructions=extra_instructions)
                elif chosen_operation.startswith("transform_to_"):
                    target_format = Format(chosen_operation.replace("transform_to_", ""))
                    if current_format and Format(current_format) != target_format:
                        result = await self.transform_question_format(
                            current_question=sentence,
                            current_format=Format(current_format),
                            answer=answer,
                            target_format=target_format,
                            extra_instructions=extra_instructions
                        )
                    else:
                        #print(f"Skipping {chosen_operation} as current format is already {target_format}")
                        attempt -= 1
                        continue
                elif chosen_operation == "keep_original" or chosen_operation == "transform_expression":
                    result = {
                        "sentence": sentence,
                        "enhancement_method": "keep_original"
                    }

                if result:
                    result["enhancement_method"] = chosen_operation
                    return result
            except Exception as e:
                print(f"Attempt failed: {str(e)}. Trying another operation.")

        print("Failed to enhance diversity after multiple attempts. Keep origin")
        result = {
            "sentence": sentence,
            "enhancement_method": "keep_original"
        }
        return result

    async def transform_expression(self, sentence, structure_type='select', example=None, custom_structures=None, extra_instructions=None):
        structure_types = [
            "conditional", "passive_voice", "active_voice", "emphasize", "emotion"
        ]
        if custom_structures:
            available_structures = list(custom_structures.keys())
        else:
            available_structures = structure_types

        # Shuffle the order of available structures
        random.shuffle(available_structures)
        if structure_type == "select":
            # Randomly select half (rounded up) of the available structures
            num_to_select = (len(available_structures) +1) // 2
            available_structures = random.sample(available_structures, num_to_select)
        elif structure_type == "example_type":
            pass
        elif structure_type is None:
            structure_type = random.choice(available_structures)

        try:
            prompt = generate_sentence_structure_prompt(sentence, structure_type, example, available_structures)
            if extra_instructions:
                prompt += f"\n\n{extra_instructions}"
        except ValueError as e:
            print(f"Error: {e}")
            return None

        service = await self._get_model_service()
        response = await service.process_async(prompt=prompt)

        try:
            transformed_sentence = clean_json(response)
            transformed_sentence["structure_type"] = structure_type
        except Exception as e:
            print(f"Error: Unable to parse the response as JSON. {str(e)}")
            return None

        return lowercase_keys(transformed_sentence)

    async def paraphrase_sentence(self, sentence, extra_instructions=None):
        prompt = generate_paraphrase_prompt(sentence)
        if extra_instructions:
            prompt += f"\n\n{extra_instructions}"
        service = await self._get_model_service()
        response = await service.process_async(prompt=prompt)

        try:
            paraphrased_sentence = clean_json(response)
        except Exception as e:
            print(f"Error: Unable to parse the response as JSON. {str(e)}")
            return None

        return lowercase_keys(paraphrased_sentence)

    async def modify_sentence_length(self, sentence, operation=None, extra_instructions=None):
        operations = ["lengthen", "shorten"]

        if operation is None:
            operation = random.choice(operations)
        elif operation.lower() not in operations:
            raise ValueError("Operation must be either 'lengthen' or 'shorten'.")

        prompt = generate_sentence_length_prompt(sentence, operation)
        if extra_instructions:
            prompt += f"\n\n{extra_instructions}"
        service = await self._get_model_service()
        response = await service.process_async(prompt=prompt)

        try:
            modified_sentence = clean_json(response)
            modified_sentence["operation"] = operation
        except Exception as e:
            print(f"Error: Unable to parse the response as JSON. {str(e)}")
            return None

        return lowercase_keys(modified_sentence)

    async def transform_question_format(self, current_question, current_format, answer=None, target_format=None, option_pool=None, extra_instructions=None):
        formats = ["multiple_choice", "true_false", "open_ended"]

        if isinstance(current_format, str):
            current_format = Format(current_format)
        
        if current_format.name not in formats:
            raise ValueError(f"Invalid format: {current_format.name}. Allowed formats are: {formats}")
        formats.remove(current_format.name)

        if target_format is None:
            available_formats = [f for f in formats if Format(f) != current_format]
            target_format = Format(random.choice(available_formats))
        else:
            if isinstance(target_format, str):
                target_format = Format(target_format)

        assert current_format != target_format, "Current format and target format are the same. No transformation needed."

        prompt = generate_transformation_prompt(str(current_format), str(target_format), current_question, option_pool, ground_truth=answer)
        if extra_instructions:
            prompt += f"\n\n{extra_instructions}"
        service = await self._get_model_service()
        response = await service.process_async(prompt=prompt)

        try:
            transformed_question = clean_json(response)
            transformed_question["format"] = target_format.name
        except Exception as e:
            print(f"Error: Unable to parse the response as JSON. {str(e)}")
            return None

        return lowercase_keys(transformed_question)