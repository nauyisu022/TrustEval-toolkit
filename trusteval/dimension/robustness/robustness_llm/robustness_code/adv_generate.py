import os
import random
from keybert import KeyBERT
from concurrent.futures import ThreadPoolExecutor
import string
import gzip
import jsonlines
import numpy as np
import sys
import re
import spacy
from sentence_transformers import SentenceTransformer
import yaml
import aiohttp
from typing import List
import asyncio

os.environ['CURL_CA_BUNDLE'] = ''
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.generation.model_service import ModelService
from openai import OpenAI
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
kw_model = KeyBERT(model=model)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../src"))
config_file_path = os.path.join(project_root, "config", "config.yaml")

EMBED_FILE = os.path.join(current_dir, '..', 'repo_tool', 'emoji-embeddings.jsonl.gz')
print(EMBED_FILE)

with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

async def call_gpt4o_api(prompt):
    service = ModelService(
        request_type="llm",
        handler_type="api",
        model_name="gpt-4o",
        config_path=config_file_path,
        temperature=0.7,
        max_tokens=2048
    )

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = await service.process_async(prompt=prompt)
            if response is not None:
                return response

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_attempts} failed with error: {e}")
            if attempt == max_attempts - 1:
                return ""

    return ""

async def call_embedding_api(prompt):
    service = ModelService(
        request_type="llm",
        handler_type="api",
        model_name="text-embedding-ada-002",
        config_path=config_file_path,
    )
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = await service.process_async(prompt=prompt)
            if response is not None:
                return response

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_attempts} failed with error: {e}")
            if attempt == max_attempts - 1:
                return ""

    return ""

def get_keywords(sentence: str, num_return: int, if_keybert: bool = False) -> List[str]:
    if if_keybert:
        keywords = kw_model.extract_keywords(
            sentence,
            keyphrase_ngram_range=(1, 1),
            use_maxsum=True,
            top_n=num_return,
            stop_words=None
        )
        return [kw[0] for kw in keywords]
    else:
        words = sentence.split()
        random.seed(42)
        num_return = min(num_return, len(words))
        return random.sample(words, num_return) if num_return > 0 else []

def determine_num_keywords(sentence: str) -> int:
    length = len(sentence)
    if length <= 100:
        return 3
    elif length <= 150:
        return 4
    elif length <= 200:
        return 5
    elif length <= 250:
        return 6
    elif length <= 350:
        return 7
    elif length <= 400:
        return 8
    else:
        return 10

def clean_word(word):
    return word.strip(string.punctuation).lower()

def remove_inner_letter(word):
    if len(word) > 2:
        middle_index = random.randint(1, len(word) - 2)
        return word[:middle_index] + word[middle_index + 1:]
    return word

def replace_inner_letter(word):
    if len(word) > 2:
        middle_index = random.randint(1, len(word) - 2)
        original_letter = word[middle_index]
        random_letter = random.choice(string.ascii_lowercase)
        while random_letter == original_letter:
            random_letter = random.choice(string.ascii_lowercase)
        return word[:middle_index] + random_letter + word[middle_index + 1:]
    return word

def repeat_inner_letter(word):
    if len(word) > 2:
        middle_index = random.randint(1, len(word) - 2)
        letter_to_repeat = word[middle_index]
        return word[:middle_index] + letter_to_repeat + word[middle_index:]
    return word

def capitalize_inner_letter(word):
    if len(word) > 2:
        middle_index = random.randint(1, len(word) - 2)
        return word[:middle_index] + word[middle_index].upper() + word[middle_index + 1:]
    return word

def insert_space_inner_letter(word):
    if len(word) > 2:
        middle_index = random.randint(1, len(word) - 2)
        return word[:middle_index] + " " + word[middle_index:]
    return word

def swap_inner_letters(word):
    if len(word) > 2:
        indices = list(range(1, len(word) - 1))
        index = random.choice(indices)
        word_list = list(word)
        word_list[index], word_list[index + 1] = word_list[index + 1], word_list[index]
        return ''.join(word_list)
    return word

def transform_sentence(sentence, transform_function, if_keybert=False):
    num_return = determine_num_keywords(sentence)
    keywords = get_keywords(sentence, num_return, if_keybert)
    words = sentence.split()
    for i, word in enumerate(words):
        if clean_word(word) in keywords:
            words[i] = transform_function(word)
    return " ".join(words)

spelling_missing_letter = lambda sentence, if_keybert=False: transform_sentence(sentence, remove_inner_letter, if_keybert)
spelling_incorrect_letter = lambda sentence, if_keybert=False: transform_sentence(sentence, replace_inner_letter, if_keybert)
spelling_repeated_letter = lambda sentence, if_keybert=False: transform_sentence(sentence, repeat_inner_letter, if_keybert)
spelling_capitalize_letter = lambda sentence, if_keybert=False: transform_sentence(sentence, capitalize_inner_letter, if_keybert)
spelling_insert_space = lambda sentence, if_keybert=False: transform_sentence(sentence, insert_space_inner_letter, if_keybert)
spelling_swap_letter = lambda sentence, if_keybert=False: transform_sentence(sentence, swap_inner_letters, if_keybert)

class EmojiSearch:
    def __init__(self):
        self._emojis = None
        self._embeddings = None

    def _load_emoji_embeddings(self):
        if self._emojis is not None and self._embeddings is not None:
            return
        with gzip.GzipFile(fileobj=open(EMBED_FILE, "rb"), mode="rb") as fin:
            emoji_info = list(jsonlines.Reader(fin))

        print("Lazy loading embedding info ...")
        self._emojis = [(x["emoji"], x["message"]) for x in emoji_info]
        self._embeddings = [x["embed"] for x in emoji_info]
        assert self._emojis is not None and self._embeddings is not None

    @property
    def emojis(self):
        if self._emojis is None:
            self._load_emoji_embeddings()
        return self._emojis

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._load_emoji_embeddings()
        return self._embeddings

    async def get_openai_embedding(self, text: str) -> list[float]:
        result = await call_embedding_api(text)
        return result

    async def get_top_relevant_emojis_batch(self, keywords: list[str], k: int = 1) -> dict:
        query_embed = await self.get_openai_embedding(" ".join(keywords))
        query_embed = np.array(query_embed, dtype=np.float64)
        embeddings_array = np.array(self.embeddings, dtype=np.float64)
        dotprod = np.matmul(embeddings_array, query_embed.T)
        inds = np.argsort(dotprod)[-k:]
        emojis = [self.emojis[ind][0] for ind in inds]
        return {keyword: emoji for keyword, emoji in zip(keywords, emojis)}

async def emoji_insertions(sentence, if_keybert=False):
    num_return = determine_num_keywords(sentence)
    keywords = get_keywords(sentence, num_return, if_keybert)
    words = sentence.split()

    keyword_to_emoji = await emoji_search.get_top_relevant_emojis_batch(keywords)

    for i, word in enumerate(words):
        cleaned_word = clean_word(word)
        if cleaned_word in keyword_to_emoji:
            words[i] = f"{word} {keyword_to_emoji[cleaned_word]}"

    return " ".join(words)

emoji_search = EmojiSearch()

async def social_tagging(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    person_found = False
    modified_sentence = sentence
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            person_found = True
            modified_sentence = modified_sentence.replace(ent.text, f"@{ent.text}")
    if person_found:
        return modified_sentence
    else:
        prompt = f"Please extract a short title or topic from the given sentence.\nSentence: {sentence} \n Output the result directly without any explanation."
        result = await call_gpt4o_api(prompt)
        tag = f"#{result}"
        if random.choice([True, False]):
            return tag + " " + sentence
        else:
            return sentence + " " + tag

def spaced_uppercase(sentence, if_keybert=False):
    num_return = determine_num_keywords(sentence)
    keywords = get_keywords(sentence, num_return, if_keybert)
    words = sentence.split()
    for i, word in enumerate(words):
        if clean_word(word) in keywords:
            words[i] = " ".join(word.upper())
    return " ".join(words)

async def call_gpt4o_api_batch(prompts):
    tasks = [call_gpt4o_api(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

import random
import re
import asyncio

async def multilingual_blend(sentence, if_keybert=False):
    num_return = determine_num_keywords(sentence)
    sentences = re.split(r'(?<=[.!?])\s+', sentence)

    if len(sentences) > 1:
        selected_index = random.randint(0, len(sentences) - 1)
        selected_sentence = sentences[selected_index]
        prompt = f"Please translate the following into Chinese: {selected_sentence} \n Output the result directly without any explanation."
        try:
            translated_sentence = await call_gpt4o_api(prompt)
        except Exception as e:
            print(f"Error translating sentence: {e}")
            translated_sentence = selected_sentence # Use original sentence if translation fails
        if translated_sentence is None:
            translated_sentence = selected_sentence # Also ensure translated_sentence is not None
        sentences[selected_index] = translated_sentence
        result = " ".join(sentences)
        return result
    else:
        keywords = get_keywords(sentence, num_return, if_keybert)
        words = sentence.split()
        prompts = [f"Please translate the following into Chinese: {word} \n Output the result directly without any explanation." for word in words if clean_word(word) in keywords]

        translation_results = await call_gpt4o_api_batch(prompts)

        word_index = 0
        for i, word in enumerate(words):
            if clean_word(word) in keywords:
                try:
                    translation = translation_results[word_index]
                    if isinstance(translation, Exception) or translation is None:
                        raise translation
                    words[i] = translation
                except Exception as e:
                    print(f"Error translating word '{word}': {e}")
                    # Keep the original word if translation fails
                    words[i] = word
                word_index += 1

        return " ".join(words)
    
async def distractive_text(sentence):
    prompt = f"Please generate a brief (15 words max) off-topic digression for the given sentence, illustrating how thoughts can wander.\nSentence: {sentence} \n Output the result directly without any explanation."
    result = await call_gpt4o_api(prompt)
    if random.choice([True, False]):
        return result + " " + sentence
    else:
        return sentence + " " + result

async def syntactic_disruption(sentence):
    prompt = f"Rewrite the following sentence with common grammatical mistakes.\nSentence: {sentence} \n Output the result directly without any explanation."
    result = await call_gpt4o_api(prompt)
    return result

async def recondite_word(sentence):
    prompt = f"Please replace 1-4 common words in the given sentence with their rarer synonyms.\nSentence: {sentence} \n Output the result directly without any explanation."
    result = await call_gpt4o_api(prompt)
    return result
