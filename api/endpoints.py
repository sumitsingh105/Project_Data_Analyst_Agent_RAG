from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
import logging

# Import the main agent function
from agent.main_agent import agentic_rag_agent

TEMP_DIR = "temp_workspaces"
router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/api/", tags=["Analysis"])
async def analyze_data(
    questions_txt: UploadFile = File(..., alias="questions.txt"),
    sample_weather_csv: UploadFile = File(None, alias="sample-weather.csv"),
    sample_sales_csv: UploadFile = File(None, alias="sample-sales.csv"),
    edges_csv: UploadFile = File(None, alias="edges.csv"),
    attachments: list[UploadFile] = File(default=[])
):
    """Enhanced endpoint - routes all analysis through main agent"""
    
    request_id = str(uuid.uuid4())
    workspace_dir = os.path.join(TEMP_DIR, request_id)
    os.makedirs(workspace_dir, exist_ok=True)

    try:
        # Read questions file
        question_content = await questions_txt.read()
        question_text = question_content.decode('utf-8')
        logger.info(f"📋 Processing: {question_text[:100]}...")

        # Save questions file
        questions_path = os.path.join(workspace_dir, "questions.txt")
        with open(questions_path, 'w', encoding='utf-8') as f:
            f.write(question_text)

        # Save all uploaded files
        dataset_files = []
        
        file_mapping = [
            ("sample-weather.csv", sample_weather_csv),
            ("sample-sales.csv", sample_sales_csv),
            ("edges.csv", edges_csv)
        ]
        
        for field_name, file_obj in file_mapping:
            if file_obj:
                file_path = os.path.join(workspace_dir, field_name)
                content = await file_obj.read()
                with open(file_path, 'wb') as f:
                    f.write(content)
                dataset_files.append(field_name)
                logger.info(f"💾 Saved: {field_name}")

        # Save additional attachments
        for attachment in attachments:
            if attachment and attachment.filename:
                file_path = os.path.join(workspace_dir, attachment.filename)
                content = await attachment.read()
                with open(file_path, 'wb') as f:
                    f.write(content)
                dataset_files.append(attachment.filename)

        # 🚀 Route everything through main agent (CSV + Wikipedia + DuckDB)
        logger.info("🧠 Routing analysis through main agent pipeline")
        result = await agentic_rag_agent(question_text, workspace_dir, dataset_files)
        
        # Ensure JSON-safe response
        return sanitize_json_response(result)
        
    except Exception as e:
        logger.error(f"❌ Processing error: {str(e)}")
        return {"error": f"Processing failed: {str(e)}"}
    
    finally:
        # Cleanup workspace
        if os.path.exists(workspace_dir):
            try:
                shutil.rmtree(workspace_dir)
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")

def sanitize_json_response(response_data):
    """Ensure response data is JSON-safe"""
    if isinstance(response_data, dict):
        sanitized = {}
        for key, value in response_data.items():
            if isinstance(value, str) and value.startswith("data:image"):
                # Ensure base64 strings are properly escaped
                sanitized[key] = value.replace('\\', '\\\\').replace('"', '\\"')
            elif isinstance(value, dict):
                sanitized[key] = sanitize_json_response(value)
            elif isinstance(value, list):
                sanitized[key] = [sanitize_json_response(item) if isinstance(item, (dict, list)) else item for item in value]
            else:
                sanitized[key] = value
        return sanitized
    
    elif isinstance(response_data, list):
        return [sanitize_json_response(item) if isinstance(item, (dict, list)) else item for item in response_data]
    
    return response_data
