def analyze_query(task_description: str) -> str:
    """Enhanced query analysis that's more adaptive"""
    # Extract keywords from the actual task instead of hardcoded list
    task_lower = task_description.lower()
    
    # Look for data analysis indicators
    analysis_indicators = ['analyze', 'calculate', 'correlation', 'plot', 'chart', 'compare']
    web_indicators = ['http', 'www', 'url', 'website', 'scrape']
    duckdb_indicators = [
        'duckdb', 'parquet', 's3://', 'court', 'judgment', 'regression slope', 
        'high court', 'disposed', 'cases', 'ecourts'
    ]
    
    if any(indicator in task_lower for indicator in web_indicators):
        return 'external'
    elif any(indicator in task_lower for indicator in analysis_indicators):
        return 'local'
    elif any(indicator in task_lower for indicator in duckdb_indicators):
        return 'duckdb'
    else:
        return 'external'  # Default to external for unknown patterns
