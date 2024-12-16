import random
from keybert import KeyBERT
import string
import gzip
import jsonlines
import numpy as np
import re
import os
import sys
from src.generation import ModelService
import spacy
from .youdao_api import connect

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(project_root)
#here is the code of different perturbation type of text


service = ModelService(
    request_type='llm',         # The type of request (e.g., 'llm' for language models)
    handler_type='api',         # The type of handler (e.g., 'api' for API-based requests)
    model_name='gpt-4o-mini',        # The name of the model to use
    config_path=os.path.join(project_root, 'src/config/config.yaml')
)

embeding_service = ModelService(
    request_type='llm',         # The type of request (e.g., 'llm' for language models)
    handler_type='api',         # The type of handler (e.g., 'api' for API-based requests)
    model_name='text-embedding-ada-002',        # The name of the model to use
    config_path=os.path.join(project_root, 'src/config/config.yaml')
)

def determine_num_keywords(sentence: str) -> int:
    length = len(sentence)
    
    if length <= 50:
        return 2
    elif length <= 100:
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

# spelling mistake 1 - missing letters
def remove_inner_letter(word):
    if len(word) > 2:  
        middle_index = random.randint(1, len(word) - 2)  
        return word[:middle_index] + word[middle_index + 1:]  
    return word  


def spelling_missing_letter(sentence, if_keybert=False):
    num_return = determine_num_keywords(sentence)
    if if_keybert:
        kw_model = KeyBERT(model='/home/shijiawen/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a')  ##待删除
        keywords = kw_model.extract_keywords(
            sentence, 
            keyphrase_ngram_range=(1, 1),
            use_maxsum=True,
            top_n=num_return,
            stop_words=None
        )
        keywords = [kw[0] for kw in keywords]

    else:
        words = sentence.split()
        random.seed(42)
        keywords = random.sample(words, num_return)
    words = sentence.split()  
    for i, word in enumerate(words):
        if clean_word(word) in keywords:
            words[i] = remove_inner_letter(word)
    return " ".join(words)

# spelling mistake 2 - incorrect letters

def replace_inner_letter(word):
    if len(word) > 2:  
        middle_index = random.randint(1, len(word) - 2)  
        original_letter = word[middle_index]  
        random_letter = random.choice(string.ascii_lowercase)  
        while random_letter == original_letter:
            random_letter = random.choice(string.ascii_lowercase)
        return word[:middle_index] + random_letter + word[middle_index + 1:]  
    return word  

def spelling_incorrect_letter(sentence, if_keybert=False):
    num_return = determine_num_keywords(sentence)
    if if_keybert:
        kw_model = KeyBERT(model='/home/shijiawen/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a')  ##待删除
        keywords = kw_model.extract_keywords(
            sentence, 
            keyphrase_ngram_range=(1, 1),
            use_maxsum=True,
            top_n=num_return,
            stop_words=None
        )
        keywords = [kw[0] for kw in keywords]
    else:
        words = sentence.split()
        random.seed(42)
        keywords = random.sample(words, num_return)
    words = sentence.split()  
    for i, word in enumerate(words):
        if clean_word(word) in keywords:
            words[i] = replace_inner_letter(word)
    return " ".join(words)
# spelling mistake 3 - repeated letters

def repeat_inner_letter(word):
    if len(word) > 2:  
        middle_index = random.randint(1, len(word) - 2)  
        letter_to_repeat = word[middle_index]  
        return word[:middle_index] + letter_to_repeat + word[middle_index:] 
    return word  

def spelling_repeated_letter(sentence, if_keybert=False):
    num_return = determine_num_keywords(sentence)
    if if_keybert:
        kw_model = KeyBERT(model='/home/shijiawen/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a')  ##待删除
        keywords = kw_model.extract_keywords(
            sentence, 
            keyphrase_ngram_range=(1, 1),
            use_maxsum=True,
            top_n=num_return,
            stop_words=None
        )
        keywords = [kw[0] for kw in keywords]
    else:
        words = sentence.split()
        random.seed(42)
        keywords = random.sample(words, num_return)
    words = sentence.split()  
    for i, word in enumerate(words):
        if clean_word(word) in keywords:
            words[i] = repeat_inner_letter(word)
    return " ".join(words)
# spelling mistake 4 - capitalization letters
def capitalize_inner_letter(word):
    if len(word) > 2: 
        middle_index = random.randint(1, len(word) - 2)  
        return word[:middle_index] + word[middle_index].upper() + word[middle_index + 1:]  
    return word  

def spelling_capitalize_letter(sentence, if_keybert=False):
    num_return = determine_num_keywords(sentence)
    if if_keybert:
        kw_model = KeyBERT(model='/home/shijiawen/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a')  ##待删除
        keywords = kw_model.extract_keywords(
            sentence, 
            keyphrase_ngram_range=(1, 1),
            use_maxsum=True,
            top_n=num_return,
            stop_words=None
        )
        keywords = [kw[0] for kw in keywords]

    else:
        words = sentence.split()
        random.seed(42)
        keywords = random.sample(words, num_return)
    words = sentence.split()  
    for i, word in enumerate(words):
        if clean_word(word) in keywords:
            words[i] = capitalize_inner_letter(word)
    return " ".join(words)

# spelling mistake 5 - Insert space

def insert_space_inner_letter(word):
    if len(word) > 2: 
        middle_index = random.randint(1, len(word) - 2)  
        return word[:middle_index] + " " + word[middle_index :]  
    return word  

def spelling_insert_space(sentence, if_keybert=False):
    num_return = determine_num_keywords(sentence)
    if if_keybert:
        kw_model = KeyBERT(model='/home/shijiawen/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a')  ##待删除
        keywords = kw_model.extract_keywords(
            sentence, 
            keyphrase_ngram_range=(1, 1),
            use_maxsum=True,
            top_n=num_return,
            stop_words=None
        )
        keywords = [kw[0] for kw in keywords]

    else:
        words = sentence.split()
        random.seed(42)
        keywords = random.sample(words, num_return)
    words = sentence.split()  
    for i, word in enumerate(words):
        if clean_word(word) in keywords:
            words[i] = insert_space_inner_letter(word)
    return " ".join(words)

# spelling mistake 6 - letters change
def swap_inner_letters(word):
    if len(word) > 2:
        indices = list(range(1, len(word) - 1))
        index = random.choice(indices) 
        word_list = list(word)
        word_list[index], word_list[index + 1] = word_list[index + 1], word_list[index]
        return ''.join(word_list)
    return word


def spelling_swap_letter(sentence, if_keybert=False):
    num_return = determine_num_keywords(sentence)
    if if_keybert:
        kw_model = KeyBERT(model='/home/shijiawen/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a')  ##待删除
        keywords = kw_model.extract_keywords(
            sentence, 
            keyphrase_ngram_range=(1, 1),
            use_maxsum=True,
            top_n=num_return,
            stop_words=None
        )
        keywords = [kw[0] for kw in keywords]

    else:
        words = sentence.split()
        random.seed(42)
        keywords = random.sample(words, num_return)
    words = sentence.split()  
    for i, word in enumerate(words):
        if clean_word(word) in keywords:
            words[i] = swap_inner_letters(word)
    return " ".join(words)


class EmojiSearch:
    def __init__(self):
        self._emojis = None
        self._embeddings = None

    def _load_emoji_embeddings(self):
        if self._emojis is not None and self._embeddings is not None:
            return

        with gzip.GzipFile(fileobj=open(os.path.join(os.path.dirname(__file__), 'emoji-embeddings.jsonl.gz'), "rb"), mode="rb") as fin:
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

    # def get_openai_embedding(self, text: str) -> list[float]:
        
    #     result =  client.embeddings.create(input=text, model ="text-embedding-ada-002")

    #     return result.data[0].embedding


    def get_top_relevant_emojis(self, query: str, k: int = 1) -> str:
        # query_embed = asyncio.run(get_azure_openai_embeding_response(query))
        query_embed = embeding_service.process(query)
        dotprod = np.matmul(self.embeddings, np.array(query_embed).T)
        ind = np.argmax(dotprod)
        return self.emojis[ind][0]

emoji_search = EmojiSearch()


def emoji_insertions(sentence, if_keybert=False):
    num_return = determine_num_keywords(sentence)
    if if_keybert:
        kw_model = KeyBERT(model='/home/shijiawen/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a') ###待删除
        keywords = kw_model.extract_keywords(
            sentence, 
            keyphrase_ngram_range=(1, 1),
            use_maxsum=True,
            top_n=num_return,
            stop_words=None
        )
        keywords = [kw[0] for kw in keywords]

    else:
        words = sentence.split()
        random.seed(42)
        keywords = random.sample(words, num_return)

    words = sentence.split()
    for i, word in enumerate(words):
        if clean_word(word) in keywords:
            emoji = emoji_search.get_top_relevant_emojis(word)
            words[i] = f"{word} {emoji}"
    return " ".join(words)
# social tagging

def social_tagging(sentence):
    from spacy.cli import download
    download("en_core_web_sm")
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
        # result = asyncio.run(get_azure_openai_text_response(prompt=prompt))
        result = service.process(prompt)
        tag = f"# {result}"
        if random.choice([True, False]):
            return tag + " " + sentence
        else:
            return sentence +  " " + tag

# spaced uppercase
def spaced_uppercase(sentence, if_keybert=False):
    num_return = determine_num_keywords(sentence)
    if if_keybert:
        kw_model = KeyBERT(model='/home/shijiawen/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a')  ##待删除
        keywords = kw_model.extract_keywords(
            sentence, 
            keyphrase_ngram_range=(1, 1),
            use_maxsum=True,
            top_n=num_return,
            stop_words=None
        )
        keywords = [kw[0] for kw in keywords]

    else:
        # Split the sentence into words
        words = sentence.split()
        # random.seed(42)
        keywords = random.sample(words, num_return)
    words = sentence.split()
    for i, word in enumerate(words):
        if clean_word(word) in keywords:
            words[i] = " ".join(word.upper())
    return " ".join(words)
# multilingual blend

def multilingual_blend(sentence, if_keybert=False):
    num_return = determine_num_keywords(sentence)
    # print(num_return)
    sentences = re.split(r'(?<=[.!?])\s+', sentence)
    
    if len(sentences) > 1:
        selected_index = random.randint(0, len(sentences) - 1)
        selected_sentence = sentences[selected_index]
        translated_sentence = connect(selected_sentence)

        sentences[selected_index] = translated_sentence
        result = " ".join(sentences)
        return result
    else:
        if if_keybert:
            kw_model = KeyBERT(model='/home/shijiawen/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a')  ##待删除
            keywords = kw_model.extract_keywords(
                sentence, 
                keyphrase_ngram_range=(1, 1),
                use_maxsum=True,
                top_n=num_return,
                stop_words=None
            )
            keywords = [kw[0] for kw in keywords]

        else:
            # Split the sentence into words
            words = sentence.split()
            # print(words)
            random.seed(42)
            keywords = random.sample(words, num_return)
            # print(keywords)
        words = sentence.split()
        for i, word in enumerate(words):
            if clean_word(word) in keywords:
                words[i] = connect(word)
                # words[i] = word
                # print(words[i])
        return " ".join(words)

# distractive Text
def distractive_text(sentence):
    prompt = f"Please generate a brief (15 words max) off-topic digression for the given sentence, illustrating how thoughts can wander.\nSentence: {sentence} \n Output the result directly without any explanation."
    # result = asyncio.run(get_azure_openai_text_response(prompt=prompt))
    result = service.process(prompt)
    if random.choice([True, False]):
        return result + " " + sentence
    else:
        return sentence +  " " + result

# syntactic disruptions
def syntactic_disruption(sentence):
    prompt = f"Rewrite the following sentence with common grammatical mistakes.\nSentence: {sentence} \n Output the result directly without any explanation."
    result = service.process(prompt)
    return result

# recondite words
def recondite_word(sentence):
    prompt = f"Please replace 1-4 common words in the given sentence with their rarer synonyms.\nSentence: {sentence} \n Output the result directly without any explanation."
    result = service.process(prompt)
    return result

 

