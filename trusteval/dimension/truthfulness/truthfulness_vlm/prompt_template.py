prompt = 'Imagine you are an intelligent teacher. Thoroughly read the question, reference answer and the prediction answer to ensure a clear understanding of the information provided. Assess the correctness of the predictions. '
prompt += 'If the prediction answer does not conflict with the reference answer, please generate “correct”. If the prediction answer conflict with the reference answer, please generate “incorrect”. The output should only be “correct” or “incorrect”. \n\n Question:'
prompt += sample['prompt']
prompt += '\nReference answer: '
prompt += sample['ground_truth']
prompt += '\nPrediction answer:'
prompt += sample[output_entry]
prompt += '\nOutput:'

# API call

output_text # either “correct” or “incorrect”

if 'incorrect' in output_text.lower(): 
    correctness = False

elif 'correct' in output_text.lower():
    correctness = True
else:
    correctness = False

