import re


def extract_entities(text: str) -> dict:
    """
    Simple rule-based entity extraction.
    """
    entities = {
        "PERSON": [],
        "ROLE": [],
        "TECH": [],
    }

    # Pessoa (simplificado)
    match = re.search(r"([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç]+ [A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç]+)", text)
    if match:
        entities["PERSON"].append(match.group(1))

    # Cargo
    if "engenheiro de software" in text.lower():
        entities["ROLE"].append("Engenheiro de Software")

    # Tecnologias
    for tech in ["Python", "RAG", "FAISS", "FastAPI", "LLMs"]:
        if tech.lower() in text.lower():
            entities["TECH"].append(tech)

    return entities
