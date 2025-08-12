import asyncio
from .query_analyzer import analyze_query
from .retriever import retrieve_data, grade_retrieval_matches
from .query_rewriter import rewrite_query
from .grader import grade_answer

from .agent import (
    run_agent_loop,
    detect_analysis_type,
    handle_duckdb_analysis,
    validate_generic_results
)

MAX_RETRIES = 3

async def agentic_rag_agent(task_description: str, workspace_dir: str, data_sources: list):
    print("Starting Agentic RAG flow")
    print(f"Task preview: {task_description[:150]}...")
    
    # BULLETPROOF ROUTING - Force DuckDB for court analysis
    task_lower = task_description.lower()
    
    # Primary check: Force DuckDB for court-related content
    court_keywords = ['high court', 'court', 'judgment', 'disposed', 'cases', 'ecourts', 'regression slope']
    is_court_analysis = any(keyword in task_lower for keyword in court_keywords)
    
    if is_court_analysis:
        print("🎯 FORCED ROUTING: Court analysis detected - using DuckDB")
        try:
            return await handle_duckdb_analysis(task_description, workspace_dir)
        except Exception as e:
            print(f"DuckDB analysis failed: {e}")
            return {"error": f"DuckDB analysis failed: {e}"}
    
    # Secondary check: Use your existing detection logic
    try:
        print("→ Running secondary detection...")
        analysis_type = detect_analysis_type(task_description)
        print(f"Detection result: {analysis_type}")
        
        if analysis_type == 'duckdb':
            print("→ Using DuckDB analysis (secondary detection)")
            return await handle_duckdb_analysis(task_description, workspace_dir)
    except Exception as e:
        print(f"Secondary detection failed: {e}")
    
    # Continue with your existing RAG logic for Wikipedia/general queries
    print("→ Using existing RAG analysis")
    
    query_type = analyze_query(task_description)
    print(f"Query type from analyzer: {query_type}")
    
    if query_type == 'local':
        candidates = retrieve_data(task_description, data_sources)
        graded_candidates = grade_retrieval_matches(candidates, task_description)
        if not graded_candidates or graded_candidates[0][0] < 0.5:
            rewritten_query = rewrite_query(task_description)
        else:
            rewritten_query = task_description
    else:
        rewritten_query = task_description

    last_error = None
    for attempt in range(MAX_RETRIES):
        print(f"Attempt {attempt+1}")
        result = await run_agent_loop(rewritten_query, workspace_dir)
        
        if 'error' in result:
            last_error = result['error']
            print(f"Execution error: {last_error}")
            rewritten_query = rewrite_query(rewritten_query, last_error)
            continue
        
        rubric = {'required_keys': []}
        if grade_answer(result):
            return result
        else:
            print("Answer failed grading, retrying...")
            last_error = "Answer failed grading criteria"
            rewritten_query = rewrite_query(rewritten_query, last_error)
            
    return {"error": f"Agent failed after {MAX_RETRIES} attempts. Last error: {last_error}"}
