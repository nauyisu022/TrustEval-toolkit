import time
from typing import List

import networkx as nx
import numpy as np

from .metadata import Text2ImageMetaData, Text2ThreeDMetaData, Text2VideoMetaData, Text2VisionMetaData
from .scene_graph import get_sg_desc, add_seed_graph_to_template_graph
from .utils import make_and_description, mention_global_attributes, normalized_sentence, capitalize_first_char_if_letter, get_element_num_dict, convert_sg_to_json, convert_json_to_sg
from ..base import TaskGenerator

def get_global_attribute_desc(global_attributes):
	global_attributes = [
		f"{mention_global_attributes(type, v)}"
		for type, v in global_attributes.items()
	]
	global_desc = make_and_description(global_attributes)
	return global_desc

def get_prompt(global_attribute_desc, sg_desc):
	global_attribute_desc = normalized_sentence(global_attribute_desc)
	sg_desc = normalized_sentence(sg_desc)
	if global_attribute_desc == "":
		return f"{capitalize_first_char_if_letter(sg_desc)}."
	else:
		return f"{capitalize_first_char_if_letter(sg_desc)}. {capitalize_first_char_if_letter(global_attribute_desc)}."


class Text2VisionPromptGenerator(TaskGenerator):
	generate_type = "vision"
	metadata: Text2VisionMetaData
	allowed_global_attributes = {
		"style": "all",
		"scene setting": "all",
		"camera setting": "all",
		"video setting": "all",
		"3d setting": "all"
	}
	def __init__(self, metadata: Text2VisionMetaData, seed=42):
		super().__init__(metadata, seed=seed)

	def _task_plan_to_str(self, task_plan):
		return get_sg_desc(task_plan["scene_graph"])

	def _complete_sg(self, scene_graph: nx.DiGraph, allowed_topic: list, consider_node_info: bool = True):
		assert isinstance(scene_graph, nx.DiGraph)
		# first adding data for each object nodes
		for node, data in scene_graph.nodes(data=True):
			if data["type"] == "object_node":
				if "value" not in data:
					data["value"] = self.metadata.sample_metadata(
						self.rng, element_type="object", allowed_topic=allowed_topic, node_info = {}
					)
				for neighbor in scene_graph.neighbors(node):
					if scene_graph.nodes[neighbor]["type"] == "attribute_node":
						k, v = self.metadata.sample_metadata(
							self.rng, element_type="attribute", allowed_topic=allowed_topic, node_info = {'its_object_value': data["value"]}
						)
						scene_graph.nodes[neighbor]["value_type"] = k
						scene_graph.nodes[neighbor]["value"] = v
		for s, t, data in scene_graph.edges(data=True):
			if "value" not in data:
				if data.get("type") == "relation_edge":
					node_info = {'its_source_object_value': scene_graph.nodes[s]["value"], 
								'its_target_object_value': scene_graph.nodes[t]["value"]}
					k, v = self.metadata.sample_metadata(
						self.rng, element_type="relation", allowed_topic=allowed_topic, node_info=node_info, consider_node_info=consider_node_info
					)
					data["value_type"] = k
					data["value"] = v
		return scene_graph

	def _sample_scene_graph(self, complexity, seed_graph, seed_graph_element_num_dict, element_num_dict, allowed_topic, consider_node_info, retry=50):
		sg_templates = self.metadata.query_sg_templates(
			complexity, seed_graph_element_num_dict, element_num_dict
		)
		if len(sg_templates) == 0:
			raise ValueError("No specific template scene graph found")

		conditioned_template = None
		for i in self.rng.permutation(len(sg_templates)):
			template_graph = sg_templates[i]
			conditioned_templates = add_seed_graph_to_template_graph(
				seed_graph, template_graph
			)
			# randomly pick one of the conditioned templates
			if len(conditioned_templates) != 0:
				index = self.rng.integers(len(conditioned_templates))
				conditioned_template = conditioned_templates[index]
				break

		if conditioned_template is None:
			raise ValueError("No template scene graph matches seed graph")
		
		scene_graph = self._complete_sg(conditioned_template, allowed_topic, consider_node_info)
		return scene_graph

	def _sample_global_attributes(self, number_of_global_attributes, allowed_global_attributes):
		return self.metadata.sample_global_attribute(self.rng, number_of_global_attributes, allowed_global_attributes)

	def sample_task_plans(
			self,
			complexity=5,
			number_of_global_attributes=1,
			sample_numbers=100,
			time_limit=60,
			seed_graph: nx.DiGraph = None,
			allowed_topic: list = "all",
			allowed_global_attributes: list = None,
			element_num_dict: dict = None,
			consider_node_info: bool = True,
	) -> List:

		# check whether user input is legal
		if seed_graph is None:
			seed_graph = nx.DiGraph()
		if allowed_global_attributes is None:
			allowed_global_attributes = self.allowed_global_attributes

		seed_graph_element_num_dict = get_element_num_dict(seed_graph)
		assert sum(seed_graph_element_num_dict.values()) <= complexity

		if element_num_dict is None:
			element_num_dict = {
				"object"   : None,
				"attribute": None,
				"relation" : None,
			}
		n_elements = 0
		for k in ['object', 'relation', 'attribute']:
			if element_num_dict[k] is not None:
				assert seed_graph_element_num_dict[k] <= element_num_dict[k]
				n_elements += element_num_dict[k]
		assert n_elements <= complexity

		# sample task plans
		task_plans = []
		start_time = time.time()
		while len(task_plans) < sample_numbers:
			# make sure the time limit is not exceeded
			if time.time() - start_time > time_limit:
				print("Time limit: 60s exceeded. Exiting the sampling process.")
				break
			scene_graph = self._sample_scene_graph(complexity, seed_graph, seed_graph_element_num_dict, element_num_dict, allowed_topic, consider_node_info)
			global_attributes = self._sample_global_attributes(number_of_global_attributes, allowed_global_attributes)
			scene_graph_str = convert_sg_to_json(scene_graph)
			task_plans.append(
				{
					"global_attributes": global_attributes,
					"scene_graph"      : scene_graph_str,
				}
			)
		print(f"sampling {len(task_plans)} task plans.")
		return task_plans

	def _generate_task(self, task_plan):
		scene_graph = convert_json_to_sg(task_plan["scene_graph"])
		sg_desc = get_sg_desc(scene_graph)
		global_attribute_desc = get_global_attribute_desc(task_plan["global_attributes"])
		prompt = get_prompt(global_attribute_desc, sg_desc)
		return prompt

	def generate(self, task_plan, seed=None):
		if seed is not None:
			self.rng = np.random.default_rng(seed=seed)
		prompt = self._generate_task(task_plan)

		task = {
			"prompt"           : prompt,
			"global_attributes": task_plan["global_attributes"],
			"scene_graph"      : task_plan["scene_graph"],
		}
		return task


class Text2ImagePromptGenerator(Text2VisionPromptGenerator):
	generate_type = "image"
	allowed_global_attributes = {
		"style": "all",
		"scene setting": "all",
		"camera setting": "all",
	}

	def __init__(self, metadata: Text2ImageMetaData, seed=42):
		super().__init__(metadata, seed=seed)


class Text2VideoPromptGenerator(Text2VisionPromptGenerator):
	generate_type = "video"
	allowed_global_attributes = {
		"style": "all",
		"scene setting": "all",
		"camera setting": "all",
		"video setting": "all",
	}
	def __init__(self, metadata: Text2VideoMetaData, seed=42):
		super().__init__(metadata, seed=seed)


class Text2ThreeDPromptGenerator(Text2VisionPromptGenerator):
	generate_type = "3D scene"
	allowed_global_attributes = {
		"style": "all",
	}

	def __init__(self, metadata: Text2ThreeDMetaData, seed=42):
		super().__init__(metadata, seed=seed)


