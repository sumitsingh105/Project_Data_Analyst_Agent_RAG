import asyncio
from .query_analyzer import analyze_query
from .retriever import retrieve_data, grade_retrieval_matches
from .query_rewriter import rewrite_query
from .grader import grade_answer

from .agent import (
    run_agent_loop,
    validate_generic_results   # ← add this line
)

MAX_RETRIES = 3

async def agentic_rag_agent(task_description: str, workspace_dir: str, data_sources: list):
    print("Starting Agentic RAG flow")
    query_type = analyze_query(task_description)
    
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
        
        rubric = {'required_keys': []}  # Define expected keys in output if applicable
        if grade_answer(result):
            return result
        else:
            print("Answer failed grading, retrying...")
            last_error = "Answer failed grading criteria"
            rewritten_query = rewrite_query(rewritten_query, last_error)
            
    return {"error": f"Agent failed after {MAX_RETRIES} attempts. Last error: {last_error}"}
