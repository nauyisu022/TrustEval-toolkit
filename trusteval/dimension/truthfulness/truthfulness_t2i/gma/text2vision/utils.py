import networkx as nx

def make_and_description(names):
	names = [name for name in names]
	if len(names) == 0:
		return ""
	elif len(names) == 1:
		return names[0]
	elif len(names) == 2:
		return ' and '.join(names)
	else:
		names = names[:-1] + [f'and {names[-1]}']
		return ', '.join(names)

def normalized_sentence(sentence):
	return sentence.replace('_', ' ')

# add proposition
def mention_global_attributes(type, global_attribute):
    if type == "genre":
        return f"in the {global_attribute} genre" 
    elif type == "artist":
        return f"in the style of the artist {global_attribute}" 
    elif type == "painting style":
        return f"with the {global_attribute} painting style" 
    elif type == "technique":
        return f"using the {global_attribute} technique"
    elif type == "weather":
        return f"in {global_attribute} weather"
    elif type == "location":
        return f"in {global_attribute} scene"
    elif type == "lighting":
        return f"illuminated by {global_attribute}"
    elif type == "size":
        return f"with {global_attribute}"
    elif type == "view": 
        return f"viewed from {global_attribute}"
    elif type == "depth of focus":
        return f"with {global_attribute}"
    elif type == "focal length":
        return f"shot at {global_attribute}"
    elif type == "camera model":
        return f"filmed with a {global_attribute}"
    elif type == "camera movement":
        return f"filmed with {global_attribute}"
    elif type == "camera gear":
        return f"using a {global_attribute}"
    elif type == "video editting style":
        return f"edited in the {global_attribute} style"
    elif type == "time span":
        return f"spanning {global_attribute}"
    elif type == "ISO":
        return f"ISO {global_attribute}"
    elif type == "aperture":
        return f"at {global_attribute} aperture"
    else:
        return global_attribute


def capitalize_first_char_if_letter(s):
	if len(s) == 0:
		return s
	if s[0].isalpha():
		return s[0].upper() + s[1:]
	return s


def get_element_num_dict(graph):
	object_nodes = [
		(n, d) for n, d in graph.nodes(data=True) if d["type"] == "object_node"
	]
	attribute_nodes = [
		(n, d) for n, d in graph.nodes(data=True) if d["type"] == "attribute_node"
	]
	relation_edges = [
		(n1, n2, d)
		for n1, n2, d in graph.edges(data=True)
		if d.get("type") == "relation_edge"
	]
	return {
		"object"   : len(object_nodes),
		"attribute": len(attribute_nodes),
		"relation" : len(relation_edges),
	}


def convert_sg_to_json(graph: nx.DiGraph):
	nodes = list(graph.nodes(data=True))
	edges = list(graph.edges(data=True))
	graph = {
		"nodes": nodes,
		"edges": edges,
	}
	return graph


def convert_json_to_sg(graph_json: dict):
	graph = nx.DiGraph()
	graph.add_nodes_from(graph_json["nodes"])
	graph.add_edges_from(graph_json["edges"])
	return graph
