import random
from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import asyncio
from openai_handler import get_azure_openai_text_response

def method_5(sentece, num_return=3, if_keybert=False):
    if if_keybert:
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(sentece, keyphrase_ngram_range=(1, 1))
        keywords = [kw[0] for kw in keywords]
        keywords = keywords[:num_return]
    else:
        # Split the sentence into words
        words = sentece.split()
        random.seed(42)
        keywords = random.sample(words, num_return)
    words = sentece.split()
    for i, word in enumerate(words):
        if word in keywords:
            words[i] = " ".join(word.upper())
    print(keywords)
    return " ".join(words)

def method_8(sentence, num):
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to('cuda')
    text =  "paraphrase: " + sentence + " </s>"

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")


    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=5
    )

    final_outputs = []
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if line.lower() != sentence.lower() and line not in final_outputs:
            final_outputs.append(line)
        print(line)

    return final_outputs[:min(num, len(final_outputs))]

def method_9(sentence):
    prompt = f"Please replace 1-4 common words in the given sentence with their rarer synonyms.\nSentence: {sentence} \n Output the result directly without any explanation."
    result = asyncio.run(get_azure_openai_text_response(prompt=prompt))
    return result