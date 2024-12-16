import json
import random
from typing import Callable

import diskcache
import numpy as np
import sentence_transformers
import torch

from .. import Model


def make_options(choices, format='letter'):
	assert format in ['numeric', 'letter']
	if format == 'numeric':
		prefix1 = [str(i + 1) for i in range(len(choices))]
	else:
		prefix1 = [chr(ord("a") + i).upper() for i in range(len(choices))]
	prefix2 = [f"({p})" for p in prefix1]
	return prefix1, prefix2, [f'{p} {c}' for p, c in zip(prefix2, choices)]


def check_contain(answer, options):
	contains = [option in answer for option in options]
	if sum(contains) == 1:
		return contains.index(True)
	else:
		return -1


class QAModelInstance:
	def qa(self, data, prompt):
		"(Abstract method) abstract QA method"


class QAModel(Model):
	def __init__(
			self,
			model_name: str,
			prompt_name: str,
			prompt_func: Callable,
			choice_format='letter',
			enable_choice_search: bool = False,
			cache_path: str = None,
	):
		self.model = None
		self.model_name = f'{model_name} ({prompt_name})'
		self.prompt_func = prompt_func
		self.format = choice_format
		self.cache_path = cache_path

		if self.cache_path is None:
			print("[IMPORTANT] model cache is disabled")
		else:
			print(f"[IMPORTANT] model cache is enabled, cache path: {cache_path}")

		self.enable_choice_search = enable_choice_search
		if enable_choice_search:
			# use SBERT to find the closest choice
			self.sentence_transformer = sentence_transformers.SentenceTransformer("all-mpnet-base-v2", device='cpu')

	@torch.no_grad()
	def choice_search(self, free_form_answer, choices):
		query_embedding = self.sentence_transformer.encode([free_form_answer], normalize_embeddings=True)
		choices_embedding = self.sentence_transformer.encode(choices, normalize_embeddings=True)
		top_choice_index = np.argmax(np.dot(choices_embedding, query_embedding.T))
		return choices[top_choice_index]

	def _data_to_str(self, data):
		""" abstract method """

	@torch.no_grad()
	def _qa(self, data, prompt):
		if self.cache_path is None:
			return self.model.qa(data, prompt)
		else:
			with diskcache.Cache(self.cache_path, size_limit=10 * (2 ** 30)) as cache:
				key = json.dumps([self.model_name, self._data_to_str(data), prompt])
				response = cache.get(key, None)
				if response is None:
					response = self.model.qa(data, prompt)
					cache.set(key, response)
				return response

	@torch.no_grad()
	def qa(self, data, question):
		prompt = self.prompt_func(question)
		return self._qa(data, prompt)

	@torch.no_grad()
	def multiple_choice_qa(self, data, question, choices, answer=None):
		# Get VQA model's answer
		prefix1, prefix2, options = make_options(choices, self.format)
		prompt = self.prompt_func(question, options)
		free_form_answer = self._qa(data, prompt)
		free_form_answer = free_form_answer.strip()

		# Limit the answer to the choices
		if free_form_answer in choices:
			multiple_choice_answer = free_form_answer
		elif free_form_answer in options:
			multiple_choice_answer = choices[options.index(free_form_answer)]
		elif free_form_answer in prefix1:
			multiple_choice_answer = choices[prefix1.index(free_form_answer)]
		elif free_form_answer in prefix2:
			multiple_choice_answer = choices[prefix2.index(free_form_answer)]
		elif self.enable_choice_search:
			multiple_choice_answer = self.choice_search(free_form_answer, choices)
		else:
			multiple_choice_answer = ""
			for to_check in [choices, options, prefix1, prefix2]:
				idx = check_contain(free_form_answer, to_check)
				if idx != -1:
					multiple_choice_answer = choices[idx]
					break

		result = {
			"free_form_answer"      : free_form_answer,
			"multiple_choice_answer": multiple_choice_answer,
			"choices"               : choices.copy(),
		}
		if answer is not None:
			result["accuracy"] = int(answer == multiple_choice_answer)
		return result

	@torch.no_grad()
	def multiple_choice_qa_random_ordering(self, data, question, choices, answer=None, n_trials=3):
		results = {}
		accuracy = 0
		for i in range(n_trials):
			choices_i = choices.copy()
			random.shuffle(choices_i)
			results[i] = self.multiple_choice_qa(data, question, choices_i, answer)
			accuracy += results[i]["accuracy"]
		results["accuracy"] = accuracy / n_trials
		return results
