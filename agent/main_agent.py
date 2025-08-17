import asyncio
import os
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
    print(f"🔍 DEBUG - Task: {task_description[:100]}...")
    print(f"🔍 DEBUG - Data sources: {data_sources}")
    print(f"🔍 DEBUG - Workspace: {workspace_dir}")
    
    # Check CSV detection
    csv_files = [f for f in data_sources if f.endswith('.csv')]
    print(f"🔍 DEBUG - CSV files found: {csv_files}")
    # ✅ FIX: Handle type issues that cause 'list' object has no attribute 'lower'
    if isinstance(task_description, list):
        task_description = " ".join(str(item) for item in task_description)
    if not isinstance(data_sources, list):
        data_sources = []
    else:
        data_sources = [str(ds) for ds in data_sources if ds is not None]
    
    print("Starting Enhanced Agentic RAG flow")
    print(f"Task preview: {task_description[:150]}...")
    print(f"Data sources: {data_sources}")
    
    task_lower = task_description.lower()
    
    # 🎯 PRIORITY 1: CSV FILES - Route through main agent pipeline
    csv_files = [f for f in data_sources if f.endswith('.csv')]
    if csv_files:
        csv_file = csv_files[0]
        print(f"📊 CSV file detected: {csv_file}")
        print("→ Routing CSV analysis through main agent pipeline")
        
        # Create enhanced task description for the agent
        csv_analysis_task = f"""
You are analyzing a CSV file named '{csv_file}' located in the workspace directory '/workspace/'.

Original task: {task_description}

Instructions:
1. Load the CSV file using pandas: df = pd.read_csv('/workspace/{csv_file}')
2. Analyze the data according to the task requirements
3. If charts are requested, generate them using matplotlib and encode as base64
4. Return results in the exact JSON format requested in the original task
5. Handle any missing or malformed data gracefully

Generate Python code to complete this analysis.
        """
        
        # Use the main agent pipeline (same as Wikipedia)
        try:
            result = await run_agent_loop(csv_analysis_task, workspace_dir)
            print("✅ CSV analysis completed through main agent")
            return result
        except Exception as e:
            print(f"❌ CSV agent analysis failed: {e}")
            # Fall through to other analysis methods
    
    # 🎯 PRIORITY 2: DuckDB analysis for court data
    court_keywords = ['high court', 'court', 'judgment', 'disposed', 'cases', 'ecourts', 'regression slope']
    is_court_analysis = any(keyword in task_lower for keyword in court_keywords)
    
    if is_court_analysis:
        print("⚖️ Court analysis detected - using DuckDB")
        try:
            return await handle_duckdb_analysis(task_description, workspace_dir)
        except Exception as e:
            print(f"DuckDB analysis failed: {e}")
            return {"error": f"DuckDB analysis failed: {e}"}
    
    # 🎯 PRIORITY 3: Wikipedia/film analysis  
    if 'wikipedia.org' in task_lower or 'film' in task_lower or 'movie' in task_lower:
        print("🌐 Wikipedia/film analysis detected")
        try:
            return await run_agent_loop(task_description, workspace_dir)
        except Exception as e:
            print(f"Wikipedia analysis failed: {e}")
    
    # 🎯 PRIORITY 4: Secondary detection logic
    try:
        print("→ Running secondary detection...")
        analysis_type = detect_analysis_type(task_description)
        print(f"Detection result: {analysis_type}")
        
        if analysis_type == 'duckdb':
            print("→ Using DuckDB analysis (secondary detection)")
            return await handle_duckdb_analysis(task_description, workspace_dir)
    except Exception as e:
        print(f"Secondary detection failed: {e}")
    
    # 🎯 PRIORITY 5: General RAG analysis
    print("→ Using general RAG analysis")
    
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
