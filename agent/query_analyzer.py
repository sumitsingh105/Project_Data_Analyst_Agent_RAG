def analyze_query(task_description: str) -> str:
    """
    Simple heuristic-based query analyzer to classify queries as 
    'local' (related to known data) or 'external' (requiring web search).
    """
    keywords = ['movie', 'film', 'gross', 'table', 'dataset', 'finance', 'revenue']
    if any(k in task_description.lower() for k in keywords):
        return 'local'
    return 'external'
