def retrieve_data(query: str, data_sources: list) -> list:
    """
    Retrieve candidate data items (e.g., tables or documents) from local sources.
    This stub returns all for simplicity but can be extended with vector similarity search.
    """
    return data_sources

def grade_retrieval_matches(candidates: list, query: str) -> list:
    """
    Grade candidates by relevance to query using a semantic similarity heuristic or LLM-based scoring.
    For demonstration, score all equally as 1.0.
    """
    scored = [(1.0, candidate) for candidate in candidates]
    return sorted(scored, reverse=True, key=lambda x: x[0])
