def rewrite_query(original_query: str, previous_response: str = None) -> str:
    """
    Simple hard-coded rewriter or placeholder for integration with an LLM-based rewrite.
    """
    if previous_response:
        # Real implementation would call an LLM to improve query clarity/usefulness.
        return original_query + " " + "Please clarify or be more specific."
    return original_query
