def succinct_prompt(question, choices=[]):
	if len(choices) == 0:
		prompt = question
	else:
		choices = '\n'.join(choices)
		prompt = (f"{question}\n"
				  f"Select from the following choices.\n"
				  f"{choices}")

	return prompt


####################################################################################################
# videoqa
####################################################################################################


def detailed_videoqa_prompt(question, choices=[]):
	if len(choices) == 0:
		prompt = f"Based on the video, answer the question. Question: {question} Answer:"
	else:
		prompt = (f"Based on the video, output the best option for the question.\n"
				  f"You must only output the option.\n"
				  f"Question: {question}\nOptions: {' '.join(choices)}\nBest option:(")
	return prompt


def detailed_video2imageqa_prompt(question, choices=[]):
	if len(choices) == 0:
		prompt = f"This is a series of images sampled at equal intervals from the beginning to the end of a video, based on the series of images, answer the question. Question: {question} Answer:"
	else:
		prompt = (f"This is a series of images sampled at equal intervals from the beginning to the end of a video, based on the series of images, output the best option for the question.\n"
				  f"You must only output the option.\n"
				  f"Question: {question}\nOptions: {' '.join(choices)}\nBest option:(")
	return prompt


####################################################################################################
# imageqa
####################################################################################################

def detailed_imageqa_prompt(question, choices=[]):
	if len(choices) == 0:
		prompt = f"Based on the image, answer the question. Question: {question} Answer:"
	else:
		prompt = (f"Based on the image, output the best option for the question.\n"
				  f"You must only output the option.\n"
				  f"Question: {question}\nOptions: {' '.join(choices)}\nBest option:(")
	return prompt
