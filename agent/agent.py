import json
import os
import re
import asyncio
from pathlib import Path
from urllib.parse import urlparse
import aiohttp
import pandas as pd
from io import StringIO
from playwright.async_api import async_playwright
import google.generativeai as genai

# Import these from your repo:
from .prompts import get_plan_and_execute_prompt
from .sandbox import run_in_sandbox

# Configure Gemini LLM
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except Exception as e:
    print(f"FATAL: Error configuring Gemini: {e}. Ensure GOOGLE_API_KEY is set.")
    raise

generation_config = {"response_mime_type": "application/json"}

model = genai.GenerativeModel(
    'gemini-1.5-flash-latest',
    system_instruction=get_plan_and_execute_prompt()
)

def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False

async def is_url_reachable(url: str) -> bool:
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.head(url) as resp:
                return resp.status == 200
        except:
            return False

async def fetch_html_with_playwright(url: str, workspace_dir: str) -> str:
    save_path = os.path.join(workspace_dir, "page.html")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=60000, wait_until="networkidle")
            previous_height = None
            for _ in range(5):
                current_height = await page.evaluate("document.body.scrollHeight")
                if current_height == previous_height:
                    break
                previous_height = current_height
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(2000)
            html = await page.content()
            Path(save_path).write_text(html, encoding="utf-8")
        finally:
            await browser.close()
    return html

def normalize_column(col_name) -> str:
    col = str(col_name).lower().strip()
    col = re.sub(r'\(.*?\)', '', col)  # Remove parentheses content
    col = re.sub(r'\[.*?\]', '', col)  # Remove brackets content
    return col.strip()

def find_best_table(tables: list[pd.DataFrame], required_column_groups: list[list[str]]) -> pd.DataFrame | None:
    for df in tables:
        norm_cols = [normalize_column(c) for c in df.columns]
        if all(
            any(any(req_col == col or req_col in col for col in norm_cols) for req_col in group)
            for group in required_column_groups
        ):
            return df
    return None

def infer_schema(df: pd.DataFrame) -> dict:
    schema = {}
    for col in df.columns:
        norm_col = normalize_column(col)
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().unique()[:3].tolist() if not df[col].dropna().empty else []
        schema[col] = {
            'normalized': norm_col,
            'dtype': dtype,
            'sample_values': sample_vals
        }
    return schema

def format_schema_for_prompt(schema: dict) -> str:
    lines = []
    for col, info in schema.items():
        samples = ', '.join(repr(str(s)) for s in info['sample_values'])
        lines.append(f"- '{col}' (normalized: '{info['normalized']}', dtype: {info['dtype']}, examples: [{samples}])")
    return '\n'.join(lines)

async def scrape_and_find_table(url: str, question: str, workspace_dir: str) -> tuple[pd.DataFrame, str]:
    print(f"🌐 Fetching URL: {url}")
    retries = 3
    for attempt in range(1, retries + 1):
        try:
            html = await fetch_html_with_playwright(url, workspace_dir)
            break
        except Exception as e:
            print(f"Scrape attempt {attempt} failed: {e}")
            if attempt == retries:
                raise RuntimeError(f"Failed to fetch URL after {retries} attempts: {e}")
            await asyncio.sleep(2 ** attempt)

    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        raise RuntimeError(f"Error reading tables from HTML: {e}")

    if not tables:
        raise RuntimeError("No tables found on the page.")

    print(f"Found {len(tables)} tables.")
    for i, table in enumerate(tables):
        print(f"Table {i} columns: {list(table.columns)}")

    required_column_groups = [
        ['rank'],
        ['title', 'film', 'movie'],
        ['gross', 'worldwide gross', 'worldwide gross (2024 $)', 'gross worldwide'],
        ['year', 'release year']
    ]

    best_df = find_best_table(tables, required_column_groups)
    if best_df is None:
        all_cols = [list(df.columns) for df in tables]
        raise RuntimeError(f"No suitable table found. Available table columns: {all_cols}")

    print(f"🎯 Selected best table with shape {best_df.shape} and columns: {list(best_df.columns)}")

    schema = infer_schema(best_df)
    schema_description = format_schema_for_prompt(schema)
    print(f"Schema description:\n{schema_description}")

    return best_df, schema_description

def clean_dataframe_general(df: pd.DataFrame) -> pd.DataFrame:
    print("--- Cleaning DataFrame ---")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(r'\[.*?\]', '', regex=True).str.strip()
        if any(keyword in col.lower() for keyword in ['gross', 'rank', 'year', 'peak']):
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[₹$,]', '', regex=True), errors='coerce')

    df = df.dropna(how='all')
    print(df.info())
    return df

def validate_generated_code(code: str) -> list[str]:
    """Pre-run validation of generated Python code."""
    validation_errors = []
    
    # Check for required imports
    required_imports = ['pandas', 'numpy', 'matplotlib', 'base64', 'io', 'json']
    for imp in required_imports:
        if f"import {imp}" not in code:
            validation_errors.append(f"Missing import: {imp}")
    
    # Check for img_b64 assignment
    if "img_b64 = base64.b64encode" not in code:
        validation_errors.append("Missing img_b64 assignment for plot encoding")
    
    # Check for JSON output
    if "print(json.dumps(" not in code:
        validation_errors.append("Missing JSON output print statement")
    
    # Check for CSV loading
    if "pd.read_csv('/workspace/cleaned_data.csv')" not in code:
        validation_errors.append("Missing CSV file loading")
    
    # Check for forbidden variables
    if "print(prompt)" in code or "prompt" in code.split('=')[0] if '=' in code else False:
        validation_errors.append("Code uses forbidden 'prompt' variable")
    
    return validation_errors

def create_retry_prompt(task_description: str, schema_description: str, error_type: str, error_details: str = "") -> str:
    """Create context-aware retry prompts based on error type."""
    
    base_context = f"""
CSV schema at '/workspace/cleaned_data.csv':
{schema_description}

Original task: {task_description}

"""
    
    if error_type == "img_b64_missing":
        return base_context + """
Your previous script failed because 'img_b64' was not properly defined. Please include this exact code block after creating your plot:

buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight')
buf.seek(0)
img_b64 = base64.b64encode(buf.getvalue()).decode()
plt.close()



Return your complete corrected script as a JSON object with a 'code' key.
"""
    
    elif error_type == "validation_errors":
        return base_context + f"""
Your previous script had validation issues: {error_details}

Please ensure your script includes:
1. All required imports: pandas, numpy, matplotlib, base64, io, json
2. Loads CSV from '/workspace/cleaned_data.csv'
3. Defines img_b64 after plotting with the exact base64 encoding block
4. Only prints json.dumps([answers]) containing the final results
5. Does not use 'prompt' as a variable name

Return your complete corrected script as a JSON object with a 'code' key.
"""
    
    elif error_type == "module_error":
        return base_context + f"""
Your script failed due to missing or incorrect imports: {error_details}

Use only these libraries: pandas, numpy, matplotlib, base64, io, json
Make sure all imports are at the top of your script.

Return your complete corrected script as a JSON object with a 'code' key.
"""
    
    elif error_type == "column_error":
        return base_context + f"""
Your script failed due to incorrect column names: {error_details}

Use the exact column names from the schema above with correct casing and spacing.
Handle missing values safely with .dropna() or pd.to_numeric(..., errors='coerce').

Return your complete corrected script as a JSON object with a 'code' key.
"""
    
    elif error_type == "json_error":
        return base_context + f"""
Your response was not valid JSON: {error_details}

Please respond with a valid JSON object containing a single 'code' key with your Python script as a properly escaped string.

Example format:
{{"code": "import pandas as pd\\nimport json\\n# your script here\\nprint(json.dumps([result]))"}}
"""
    
    else:  # generic error
        return base_context + f"""
Your script encountered an error: {error_details}

Please fix the issue and return your complete corrected script as a JSON object with a 'code' key.
Ensure you follow all system instructions for proper Python script generation.
"""

async def run_agent_loop(task_description: str, workspace_dir: str):
    print("--- AGENT STARTED ---")
    
    # Extract and validate URL
    url_match = re.search(r'https?://\S+', task_description)
    if not url_match:
        return {"error": "No URL found in task."}
    url = url_match.group(0)

    if not is_valid_url(url):
        return {"error": f"URL '{url}' is invalid."}

    if not await is_url_reachable(url):
        return {"error": f"URL '{url}' is unreachable."}

    # Scrape and process data
    try:
        scraped_df, schema_description = await scrape_and_find_table(url, task_description, workspace_dir)
    except Exception as e:
        print(f"Error scraping: {e}")
        return {"error": f"Failed scraping: {e}"}

    try:
        cleaned_df = clean_dataframe_general(scraped_df)
    except Exception as e:
        print(f"Error cleaning: {e}")
        return {"error": f"Failed cleaning: {e}"}

    # Save cleaned data
    csv_path = os.path.join(workspace_dir, "cleaned_data.csv")
    cleaned_df.to_csv(csv_path, index=False)
    print(f"✅ Saved cleaned CSV to {csv_path}")

    # Initial prompt with schema context
    current_prompt = f"""
CSV schema at '/workspace/cleaned_data.csv':
{schema_description}

Task: {task_description}

Generate a complete Python script following all system instructions. Use exact column names from the schema above.
"""

    # Code generation and execution loop
    for attempt in range(1, 5):  # Increased to 5 attempts for better reliability
        print(f"\n--- ATTEMPT {attempt} ---")
        
        try:
            # Generate code with LLM
            response = await model.generate_content_async(
                current_prompt,
                generation_config=generation_config
            )
            assistant_message = response.text
            print("--- Model response received ---")

            # Parse LLM response
            try:
                action = json.loads(assistant_message)
                code_to_run = action.get("code")
                if not code_to_run:
                    current_prompt = create_retry_prompt(
                        task_description, schema_description, 
                        "json_error", "No 'code' key found in response"
                    )
                    continue
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                current_prompt = create_retry_prompt(
                    task_description, schema_description, 
                    "json_error", str(e)
                )
                continue

            # Pre-run validation
            validation_errors = validate_generated_code(code_to_run)
            if validation_errors:
                print(f"Code validation failed: {', '.join(validation_errors)}")
                current_prompt = create_retry_prompt(
                    task_description, schema_description, 
                    "validation_errors", ', '.join(validation_errors)
                )
                continue

            # Execute code in sandbox
            print("--- Executing code in sandbox ---")
            stdout, stderr = run_in_sandbox(workspace_dir, code_to_run)
            print(f"--- Script STDOUT ---\n{stdout}")
            print(f"--- Script STDERR ---\n{stderr}")

            # Handle execution errors
            if stderr:
                if "name 'img_b64' is not defined" in stderr:
                    print("img_b64 error detected")
                    current_prompt = create_retry_prompt(
                        task_description, schema_description, 
                        "img_b64_missing", stderr
                    )
                    continue
                elif "ModuleNotFoundError" in stderr or "ImportError" in stderr:
                    print("Import error detected")
                    current_prompt = create_retry_prompt(
                        task_description, schema_description, 
                        "module_error", stderr
                    )
                    continue
                elif "KeyError" in stderr or "AttributeError" in stderr:
                    print("Column/attribute error detected")
                    current_prompt = create_retry_prompt(
                        task_description, schema_description, 
                        "column_error", stderr
                    )
                    continue
                else:
                    # Generic execution error - return with details
                    return {
                        "error": f"Script execution failed: {stderr}", 
                        "code": code_to_run,
                        "attempt": attempt
                    }

            # Parse and validate output
            if not stdout.strip():
                current_prompt = create_retry_prompt(
                    task_description, schema_description, 
                    "generic", "Script produced no output"
                )
                continue

            try:
                final_answer = json.loads(stdout.strip())
                print("--- JSON output parsed successfully ---")
                
                # Additional validation of output format
                if not isinstance(final_answer, list):
                    current_prompt = create_retry_prompt(
                        task_description, schema_description, 
                        "generic", "Output must be a JSON array"
                    )
                    continue
                
                return final_answer
                
            except json.JSONDecodeError as e:
                print(f"Output JSON decode error: {e}")
                current_prompt = create_retry_prompt(
                    task_description, schema_description, 
                    "generic", f"Script output was not valid JSON: {e}. Output: {stdout[:200]}..."
                )
                continue

        except Exception as e:
            print(f"Fatal error in attempt {attempt}: {e}")
            current_prompt = create_retry_prompt(
                task_description, schema_description, 
                "generic", f"Fatal error: {e}"
            )
            continue

    # All attempts failed
    print("--- Agent failed after all attempts ---")
    return {
        "error": "Agent failed to produce a valid working script after maximum attempts.",
        "total_attempts": attempt
    }
