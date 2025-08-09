from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
import os
import shutil
import uuid

# Import the Agentic RAG orchestrator instead of just run_agent_loop
from agent.main_agent import agentic_rag_agent

# --- Configuration ---
TEMP_DIR = "temp_workspaces"

# --- API Router ---
router = APIRouter()

@router.post("/api/", tags=["Analysis"])
async def analyze_data(
    questions: UploadFile = File(...),
    attachments: list[UploadFile] = File([])
):
    """
    API endpoint to handle data analysis tasks.
    Accepts a 'questions.txt' file and optional attachments,
    then runs the modular RAG data analyst agent to produce a result.
    """
    # Create a unique workspace for this request
    request_id = str(uuid.uuid4())
    workspace_dir = os.path.join(TEMP_DIR, request_id)
    os.makedirs(workspace_dir, exist_ok=True)

    try:
        # Save the main questions file
        question_content = await questions.read()
        question_text = question_content.decode('utf-8')
        question_path = os.path.join(workspace_dir, "questions.txt")
        async with aiofiles.open(question_path, 'wb') as out_file:
            await out_file.write(question_content)

        # Save any additional attachments
        for attachment in attachments:
            if attachment.filename:
                attachment_path = os.path.join(workspace_dir, attachment.filename)
                async with aiofiles.open(attachment_path, 'wb') as out_file:
                    content = await attachment.read()
                    await out_file.write(content)

        # Prepare your data sources list. For now, you can start with empty or a list that your retriever module expects.
        # You may want to add logic to load or extract data sources based on attachments or saved files.
        data_sources = []  # Example placeholder; update as per your implementation

        # Run the modular Agentic RAG orchestrator for the given question
        final_answer = await agentic_rag_agent(question_text, workspace_dir, data_sources)

        if final_answer:
            return JSONResponse(content=final_answer)
        else:
            raise HTTPException(status_code=500, detail="Agent could not complete the task.")

    except Exception as e:
        # Generic error handling
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup workspace directory - consider async deletion if you want non-blocking cleanup
        if os.path.exists(workspace_dir):
            shutil.rmtree(workspace_dir)
