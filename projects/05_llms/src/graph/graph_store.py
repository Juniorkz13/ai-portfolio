import networkx as nx


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity: str, entity_type: str):
        self.graph.add_node(entity, type=entity_type)

    def add_relation(self, source: str, relation: str, target: str):
        self.graph.add_edge(source, target, relation=relation)

    def get_subgraph(self, entity: str, depth: int = 1):
        nodes = nx.single_source_shortest_path_length(
            self.graph, entity, cutoff=depth
        ).keys()
        return self.graph.subgraph(nodes)
