from typing import Dict

import numpy as np

from .metadata import MetaData


class TaskGenerator:
	schema = {}
	def __init__(self, metadata: MetaData, seed=42):
		self.metadata = metadata
		self.rng = np.random.default_rng(seed=seed)

	def _task_plan_to_str(self, task_plan) -> str:
		"(Abstract method) convert task plan to string for task embedding"

	def generate(self, task_plan, seed=None):
		"(Abstract method) enumerate task"

