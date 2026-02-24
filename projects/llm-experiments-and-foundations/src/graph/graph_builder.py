from src.graph.graph_store import KnowledgeGraph
from src.graph.entity_extractor import extract_entities


def build_graph_from_docs(docs, root_entity: str | None = None):
    """
    Build a knowledge graph from retrieved documents.
    Always guarantees the root entity exists.
    """

    graph = KnowledgeGraph()

    if root_entity:
        graph.add_entity(root_entity, "ENTITY")

    for doc in docs:
        text = doc["metadata"]["content"]
        entities = extract_entities(text)

        subject = root_entity
        if not subject and entities["PERSON"]:
            subject = entities["PERSON"][0]

        if not subject:
            continue

        graph.add_entity(subject, "ENTITY")

        for role in entities.get("ROLE", []):
            graph.add_entity(role, "ROLE")
            graph.add_relation(subject, "HAS_ROLE", role)

        for tech in entities.get("TECH", []):
            graph.add_entity(tech, "TECH")
            graph.add_relation(subject, "WORKED_WITH", tech)

    return graph
