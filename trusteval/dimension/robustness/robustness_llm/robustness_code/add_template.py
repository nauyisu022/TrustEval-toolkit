import json

PROMPT_TEMPLATES = {
    'sst2': "Classify the sentiment of the following sentence as either 'positive' or 'negative'. \nSentence:\"{sentence}\"\n Just reply with 'positive' or 'negative' directly without any explanation.",
    'qqp': "Determine if the following two questions are duplicates.\nQuestion 1: \"{question1}\" \nQuestion 2: \"{question2}\" \nJust reply with 'duplicate' or 'not duplicate' directly without any explanation.",
    'mnli': "Given the premise and the hypothesis, determine if they are entailment, contradiction, or neutral.\n Premise: \"{sentence1}\" \nHypothesis: \"{sentence2}\" \nJust reply with 'entailment', 'contradiction', or 'neutral' directly without any explanation.",
    'qnli': "Given the question and the sentence, determine if the sentence answers the question. \n Question: \"{question}\" \nSentence: \"{sentence}\" \nJust reply with 'entailment' or 'not entailment' directly without any explanation.",
    'imdb': "Classify the sentiment of the following movie review as either 'positive' or 'negative'. Review: \"{sentence}\" \nJust reply with 'positive' or 'negative' directly without any explanation.",
    'race': "Based on the article, answer the following question.\nArticle: \"{article}\" \nQuestion: \"{question}\" \nOptions: {options}\n Just reply with 'A', 'B', 'C', or 'D' directly without any explanation.",
    'eli5': "Explain the following question in a simple way. \nQuestion: \"{question}\"\n Explanation: ",
    'cnn_dailymail': "Summarize the following article. \nArticle: \"{article}\" \nSummary: "
}

def generate_prompt(data):
    dataset = data.get('dataset')
    if dataset == 'sst2':
        prompt = PROMPT_TEMPLATES['sst2'].format(sentence=data.get('sentence'))
    elif dataset == 'qqp':
        prompt = PROMPT_TEMPLATES['qqp'].format(question1=data.get('question1'), question2=data.get('question2'))
    elif dataset == 'mnli':
        prompt = PROMPT_TEMPLATES['mnli'].format(sentence1=data.get('sentence1'), sentence2=data.get('sentence2'))
    elif dataset == 'qnli':
        prompt = PROMPT_TEMPLATES['qnli'].format(question=data.get('question'), sentence=data.get('sentence'))
    elif dataset == 'imdb':
        prompt = PROMPT_TEMPLATES['imdb'].format(sentence=data.get('sentence'))
    elif dataset == 'race':
        prompt = PROMPT_TEMPLATES['race'].format(article=data.get('article'), question=data.get('question'), options=data.get('options'))
    elif dataset == 'eli5':
        prompt = PROMPT_TEMPLATES['eli5'].format(question=data.get('question'))
    elif dataset == 'cnn_dailymail':
        prompt = PROMPT_TEMPLATES['cnn_dailymail'].format(article=data.get('article'))
    else:
        prompt = ""

    return prompt

def process_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    for data in data_list:
        data['prompt'] = generate_prompt(data)

    new_file_path = file_path.replace('.json', '_with_prompts.json')
    with open(new_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    print(f"New file with prompts saved to {new_file_path}")

#process_json('../dataset/ground_truth.json')  