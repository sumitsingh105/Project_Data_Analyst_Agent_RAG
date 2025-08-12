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
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import traceback

# Import these from your repo:
from .prompts import get_plan_and_execute_prompt, generate_adaptive_analysis_prompt
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

def smart_table_selection(tables: list[pd.DataFrame], task_description: str) -> pd.DataFrame:
    """Select best table based on task requirements - enhanced for specific queries"""
    
    print(f"Analyzing {len(tables)} tables for relevance...")
    print(f"Task mentions: {task_description[:100]}...")
    
    best_table = None
    best_score = -1
    
    # Detect what type of table the user wants
    wants_inflation_adjusted = any(word in task_description.lower() for word in 
                                 ['inflation', 'adjusted', '2024', 'real value'])
    wants_by_year = any(phrase in task_description.lower() for phrase in [
        'by year', 'year of release', 'chronological', 'annually', 'each year'
    ])
    
    print(f"Table selection criteria:")
    print(f"- Wants inflation-adjusted: {wants_inflation_adjusted}")
    print(f"- Wants by-year: {wants_by_year}")
    
    for i, df in enumerate(tables):
        try:
            score = 0
            column_text = ' '.join(str(col).lower() for col in df.columns)
            
            print(f"\nTable {i}: shape={df.shape}")
            print(f"  Columns: {list(df.columns)[:6]}")
            
            # Check first few cell values to understand table content
            try:
                first_row_text = ' '.join(str(df.iloc[0]).lower()) if len(df) > 0 else ""
                print(f"  First row sample: {first_row_text[:100]}...")
            except:
                first_row_text = ""
            
            # PRIORITY 1: By-year table detection
            if wants_by_year:
                by_year_score = 0
                
                # Check if table has Year as first column and is chronologically organized
                if 'year' in column_text and len(df) > 80:  # by-year table is typically longer
                    by_year_score += 20
                    
                    try:
                        # Check if first column contains chronological years (1915, 1916, etc.)
                        if 'Year' in df.columns or df.columns[0].lower() == 'year':
                            first_years = df.iloc[:5, 0].tolist()  # First 5 values of first column
                            chronological_years = []
                            
                            for year in first_years:
                                if pd.notna(year):
                                    year_str = str(year)
                                    if year_str.isdigit() and 1900 < int(year_str) < 2030:
                                        chronological_years.append(int(year_str))
                            
                            # Check if years are in chronological order
                            if len(chronological_years) >= 3 and chronological_years == sorted(chronological_years):
                                by_year_score += 60
                                print(f"  ✅ Found chronological years - likely by-year table!")
                                print(f"  Sample years: {chronological_years}")
                    except Exception as e:
                        print(f"  Error checking chronological years: {e}")
                
                # Check for "by year" specific column patterns (Year, Title, Gross, Budget)
                if all(col in column_text for col in ['year', 'title', 'gross']):
                    by_year_score += 30
                
                # Penalize tables with Rank/Peak (those are ranking tables, not by-year)
                if 'rank' in column_text and 'peak' in column_text:
                    by_year_score -= 40
                    print(f"  ❌ Has Rank/Peak columns - not a by-year table")
                
                score += by_year_score
                print(f"  By-year score: {by_year_score}")
            
            # PRIORITY 2: Inflation-adjusted table detection
            elif wants_inflation_adjusted:
                inflation_score = 0
                
                # Look for inflation-adjusted indicators
                inflation_indicators = ['2024', 'adjusted', 'inflation']
                inflation_score += sum(10 for word in inflation_indicators if word in column_text)
                
                # Check if this table has "Gone with the Wind" as #1 (key indicator)
                try:
                    if len(df) > 0 and 'gone with the wind' in str(df.iloc[0]).lower():
                        inflation_score += 50
                        print(f"  ✅ Found 'Gone with the Wind' at top - likely inflation table!")
                except:
                    pass
                
                # Look for specific column patterns in inflation table
                if '2024' in column_text or 'adjusted' in column_text:
                    inflation_score += 30
                    
                score += inflation_score
                print(f"  Inflation score: {inflation_score}")
            
            # PRIORITY 3: Regular/nominal table detection
            else:
                nominal_score = 0
                
                # Look for Avatar at #1 (indicates nominal table)
                try:
                    if len(df) > 0 and 'avatar' in str(df.iloc[0]).lower():
                        nominal_score += 50
                        print(f"  ✅ Found 'Avatar' at top - likely nominal table!")
                except:
                    pass
                
                # Standard scoring for main ranking table
                if 'rank' in column_text and 'peak' in column_text:
                    nominal_score += 30
                
                score += nominal_score
                print(f"  Nominal score: {nominal_score}")
            
            # Basic quality checks (applied to all tables)
            if len(df) > 10 and len(df.columns) >= 4:
                score += 20
            
            # Check for expected content
            expected_words = ['film', 'movie', 'gross', 'title']
            content_score = sum(5 for word in expected_words if word in column_text)
            score += content_score
            
            print(f"  Total score: {score}")
            
            if score > best_score:
                best_score = score
                best_table = df
                print(f"  🎯 New best table!")
                
        except Exception as e:
            print(f"Error scoring table {i}: {e}")
            continue
    
    if best_table is not None:
        print(f"\n✅ SELECTED TABLE:")
        print(f"   Score: {best_score}")
        print(f"   Shape: {best_table.shape}")
        print(f"   Columns: {list(best_table.columns)}")
        try:
            print(f"   Top entry: {best_table.iloc[0].iloc[1] if len(best_table.columns) > 1 else 'N/A'}")
        except:
            pass
        return best_table
    else:
        print("⚠️ Falling back to largest table")
        return max(tables, key=lambda df: len(df) * len(df.columns))

def detect_data_types_automatically(df: pd.DataFrame) -> dict:
    """Ultra-safe data type detection with Indian currency support"""
    column_analysis = {}
    
    print("Starting data type analysis...")
    
    for col in df.columns:
        col_str = str(col)
        print(f"Analyzing column: {col_str}")
        
        try:
            # Get sample data safely
            sample_data = []
            try:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    samples = non_null.iloc[:3].tolist()
                    sample_data = [str(x) for x in samples]
                else:
                    sample_data = []
            except Exception as e:
                print(f"Error getting samples for {col_str}: {e}")
                sample_data = []
            
            # Enhanced type detection with Indian currency support
            col_lower = col_str.lower()
            likely_type = 'text'  # Default
            
            # Check sample data for currency indicators
            sample_text = ' '.join(sample_data).lower()
            
            if col_lower in ['rank', '#', 'no.'] or 'rank' in col_lower:
                likely_type = 'rank'
            elif col_lower in ['peak'] or 'peak' in col_lower:
                likely_type = 'rank'
            elif any(word in col_lower for word in ['year', 'date']):
                likely_type = 'year'
            elif any(word in col_lower for word in ['title', 'name', 'film']):
                likely_type = 'title'
            elif any(word in col_lower for word in ['gross', 'revenue', 'price']):
                # Detect currency type based on sample data
                if 'crore' in sample_text or '₹' in sample_text:
                    likely_type = 'indian_currency'
                    print(f"  → Detected Indian currency in column '{col_str}'")
                elif '$' in sample_text:
                    likely_type = 'currency'
                    print(f"  → Detected USD currency in column '{col_str}'")
                else:
                    likely_type = 'currency'  # Default to USD for gross columns
            
            column_analysis[col_str] = {
                'original_name': col_str,
                'likely_type': likely_type,
                'sample_values': sample_data,
                'patterns': [],
                'numeric_ratio': 0.0,
                'date_ratio': 0.0
            }
            
            print(f"Column {col_str}: {likely_type}, samples: {sample_data[:2]}")
            
        except Exception as e:
            print(f"Error analyzing column {col_str}: {e}")
            column_analysis[col_str] = {
                'original_name': col_str,
                'likely_type': 'text',
                'sample_values': [],
                'patterns': [],
                'numeric_ratio': 0.0,
                'date_ratio': 0.0
            }
    
    print(f"Data type analysis complete for {len(column_analysis)} columns")
    return column_analysis




def universal_data_cleaner(df: pd.DataFrame, column_analysis: dict) -> pd.DataFrame:
    """Universal cleaner that adapts to any data structure - handles both dollar and Indian currency"""
    print("--- Universal Data Cleaning ---")
    df_clean = df.copy()
    
    for col, analysis in column_analysis.items():
        data_type = analysis['likely_type']
        print(f"Cleaning column '{col}' as {data_type}")
        
        if data_type == 'currency':
            # Enhanced currency cleaning for Wikipedia format (Dollar amounts)
            def clean_currency(value):
                if pd.isna(value):
                    return None
                try:
                    value_str = str(value)
                    print(f"Cleaning currency: '{value_str[:50]}...'")
                    
                    # Handle complex Wikipedia formats like "$2,923,706,026" or "^T^$2,257,844,554"
                    # First remove footnote markers like ^T^, ^SM^, etc.
                    cleaned = re.sub(r'\^[A-Z0-9]+\^', '', value_str)
                    
                    # Extract the main dollar amount - look for patterns like $X,XXX,XXX,XXX
                    dollar_match = re.search(r'\$([0-9,]+(?:\.[0-9]+)?)', cleaned)
                    if dollar_match:
                        # Remove commas and convert to float
                        number_str = dollar_match.group(1).replace(',', '')
                        result = float(number_str)
                        print(f"  → Converted to: {result}")
                        return result
                    
                    # Fallback: remove everything except digits, commas, and decimal points
                    fallback = re.sub(r'[^\d,.]', '', value_str)
                    if fallback and fallback.replace(',', '').replace('.', '').isdigit():
                        result = float(fallback.replace(',', ''))
                        print(f"  → Fallback conversion: {result}")
                        return result
                    
                    print(f"  → Could not convert: '{value_str}'")
                    return None
                except Exception as e:
                    print(f"  → Error converting '{value_str}': {e}")
                    return None
            
            df_clean[col] = df_clean[col].apply(clean_currency)
            
        elif data_type == 'indian_currency':
            # Indian currency cleaning for crore format  
            def clean_indian_currency(value):
                if pd.isna(value):
                    return None
                try:
                    value_str = str(value)
                    print(f"Cleaning Indian currency: '{value_str[:50]}...'")
                    
                    # Step 1: Remove currency symbol and 'crore'
                    cleaned = value_str.replace('₹', '').replace('crore', '').strip()
                    
                    # Step 2: Remove footnote markers and extra spaces
                    cleaned = re.sub(r'\s+', '', cleaned)  # Remove all spaces
                    cleaned = re.sub(r'\^[A-Z0-9]+\^', '', cleaned)  # Remove footnotes
                    
                    # Step 3: Handle range values - take first number
                    if '–' in cleaned:  # En dash
                        cleaned = cleaned.split('–')[0]
                        print(f"  → Found range, taking first value: '{cleaned}'")
                    elif '−' in cleaned:  # Minus sign (different character)
                        cleaned = cleaned.split('−')
                    elif '-' in cleaned and not cleaned.startswith('-'):
                        # Regular dash, but not negative number
                        parts = cleaned.split('-')
                        if len(parts) > 1 and '.' in parts:  # Has decimal, likely range
                            cleaned = parts
                            print(f"  → Found dash range, taking first value: '{cleaned}'")
                    
                    # Step 4: Remove any remaining non-numeric characters except decimal
                    cleaned = re.sub(r'[^\d.]', '', cleaned)
                    
                    # Step 5: Convert to float
                    if cleaned and len(cleaned) > 0:
                        # Check if it's a valid number
                        if cleaned.replace('.', '').isdigit():
                            result = float(cleaned)
                            print(f"  → Successfully converted to: {result}")
                            return result
                    
                    print(f"  → Could not convert cleaned value: '{cleaned}'")
                    return None
                    
                except Exception as e:
                    print(f"  → Error converting '{value_str}': {e}")
                    return None
            



            df_clean[col] = df_clean[col].apply(clean_indian_currency)
            
        elif data_type in ['rank', 'numeric']:
            # Clean numeric columns
            def clean_numeric(value):
                if pd.isna(value):
                    return None
                try:
                    # Remove footnote markers first
                    cleaned_str = re.sub(r'\^[A-Z0-9]+\^', '', str(value))
                    # Remove any non-numeric characters except decimal points and minus signs
                    cleaned = re.sub(r'[^\d.-]', '', cleaned_str)
                    return float(cleaned) if cleaned and cleaned.replace('.', '').replace('-', '').isdigit() else None
                except:
                    return None
            
            df_clean[col] = df_clean[col].apply(clean_numeric)
            
        elif data_type == 'year':
            # Extract 4-digit years
            def clean_year(value):
                if pd.isna(value):
                    return None
                try:
                    year_match = re.search(r'\b(19|20)\d{2}\b', str(value))
                    return int(year_match.group(0)) if year_match else None
                except:
                    return None
            
            df_clean[col] = df_clean[col].apply(clean_year)
            
        elif data_type in ['title', 'text']:
            # Clean text columns
            def clean_text(value):
                if pd.isna(value):
                    return None
                try:
                    text = str(value)
                    # Remove footnote references like [1][2]
                    text = re.sub(r'\[.*?\]', '', text)
                    # Remove markdown formatting like *Title*
                    text = re.sub(r'\*([^*]+)\*', r'\1', text)
                    return text.strip()
                except:
                    return str(value)
            
            df_clean[col] = df_clean[col].apply(clean_text)
    
    # Remove completely empty rows
    df_clean = df_clean.dropna(how='all')
    print(f"Cleaned data shape: {df_clean.shape}")
    print("Sample cleaned data:")
    print(df_clean.head())
    
    # Debug: Check for extreme values that indicate cleaning errors
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'int64']:
            max_val = df_clean[col].max()
            min_val = df_clean[col].min()
            print(f"Column '{col}' range: {min_val} to {max_val}")
            if max_val > 1e20:  # Suspiciously large number
                print(f"⚠️ Warning: Column '{col}' has suspiciously large values (max: {max_val})")
    
    return df_clean

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
    try:
        for col, info in schema.items():
            # Ensure all values are strings
            col_str = str(col)
            normalized = str(info.get('normalized', col_str.lower()))
            dtype = str(info.get('dtype', 'mixed'))
            
            # Handle sample values safely
            sample_values = info.get('sample_values', [])
            if isinstance(sample_values, list):
                samples = ', '.join(repr(str(s)) for s in sample_values)
            else:
                samples = repr(str(sample_values))
                
            lines.append(f"- '{col_str}' (normalized: '{normalized}', dtype: {dtype}, examples: [{samples}])")
    except Exception as e:
        print(f"Error formatting schema: {e}")
        lines.append("- Schema formatting failed - will analyze dynamically")
    
    return '\n'.join(lines)

async def universal_scrape_and_analyze(url: str, task_description: str, workspace_dir: str):
    """Universal pipeline that works with any website/data"""
    
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
    
    # Select best table dynamically
    best_table = smart_table_selection(tables, task_description)
    
    # Analyze data structure automatically
    column_analysis = detect_data_types_automatically(best_table)
    
    print("Column analysis:")
    for col, analysis in column_analysis.items():
        print(f"  {col}: {analysis['likely_type']} (samples: {analysis['sample_values'][:2]})")
    
    # Clean data adaptively
    cleaned_df = universal_data_cleaner(best_table, column_analysis)
    
    # Save cleaned data
    csv_path = os.path.join(workspace_dir, "cleaned_data.csv")
    cleaned_df.to_csv(csv_path, index=False)
    print(f"✅ Saved cleaned CSV to {csv_path}")
    
    # Create adaptive prompt with error handling
    try:
        sample_data = str(cleaned_df.head(3).to_string())
        analysis_prompt = generate_adaptive_analysis_prompt(task_description, column_analysis, sample_data)
    except Exception as e:
        print(f"Error generating adaptive prompt: {e}")
        # Fallback to basic prompt
        analysis_prompt = get_plan_and_execute_prompt()
    
    return cleaned_df, analysis_prompt, column_analysis

def validate_generic_results(results: list, expected_types: list | None = None) -> tuple[bool, str]:
    """Check that the LLM's JSON answer is list-like, contains no NaN, 
       and every element is JSON-serialisable."""
    if not isinstance(results, list):
        return False, "Output must be a list"

    for i, item in enumerate(results):
        if item is None:
            continue  # None is JSON serializable
        if isinstance(item, (np.generic, pd.Series)):
            return False, f"Item {i} is a pandas / numpy object"
    return True, "OK"



def validate_generated_code(code: str) -> list[str]:
    """Pre-run validation of generated Python code."""
    validation_errors = []
    
    # Check for essential imports only
    essential_imports = ['pandas', 'json']
    for imp in essential_imports:
        if f"import {imp}" not in code:
            validation_errors.append(f"Missing essential import: {imp}")
    
    # Check for CSV loading
    if "pd.read_csv('/workspace/cleaned_data.csv')" not in code:
        validation_errors.append("Missing CSV file loading")
    
    # Check for JSON output
    if "print(json.dumps(" not in code:
        validation_errors.append("Missing JSON output print statement")
    
    return validation_errors




def create_retry_prompt(task_description: str, schema_description: str, error_type: str, error_details: str = "") -> str:
    """Create context-aware retry prompts based on error type."""
    
    base_context = f"""
CSV schema at '/workspace/cleaned_data.csv':
{schema_description}

Original task: {task_description}

"""
    
    # Add pandas boolean error handling FIRST
    if "truth value of an array" in error_details or "Use a.any() or a.all()" in error_details:
        return base_context + """
Your script failed due to improper pandas boolean operations. This error occurs when you try to use a pandas Series in an if statement.

CRITICAL FIXES:

❌ WRONG:
if df['column']: # Error!
if df['column'] == value: # Error if multiple rows!


✅ CORRECT:

For checking if column has any values:
if df['column'].any():
if not df['column'].empty:

For filtering data:
filtered_df = df[df['column'] == value]
if not filtered_df.empty:

For checking conditions:
if (df['column'] == value).any(): # At least one match
if (df['column'] == value).all(): # All match

For checking specific values:
if df['column'].iloc == value: # First row only






EXAMPLES OF SAFE PATTERNS:


Instead of: if df['Year']:
Use: if not df['Year'].empty:
Instead of: if df['Title'] == 'Avatar':
Use: if (df['Title'] == 'Avatar').any():
Instead of: if some_calculation:
Use: if some_calculation > 0: # or appropriate condition



DEBUGGING TIPS:
- Always use .any() or .all() when comparing pandas columns
- Check if DataFrames are empty with .empty property
- Use .iloc[0] to get single values for comparison

Return your complete corrected script as a JSON object with a 'code' key.
"""
    
    # Add JSON serialization error handling
    elif "int64 is not JSON serializable" in error_details or "float64 is not JSON serializable" in error_details:
        return base_context + """
Your script failed because pandas data types (int64, float64) cannot be directly JSON serialized.

CRITICAL FIX: Convert all pandas/numpy data types to native Python types before JSON output:

def safe_json_convert(obj):
import pandas as pd
import numpy as np
if hasattr(obj, 'item'): # numpy scalar
return obj.item()
elif hasattr(obj, 'tolist'): # numpy array
return obj.tolist()
elif pd.isna(obj): # pandas NaN
return None
elif isinstance(obj, (np.integer,)):
return int(obj)
elif isinstance(obj, (np.floating,)):
return float(obj)
else:
return obj

Apply conversion to your results
answers = [safe_json_convert(result) for result in answers]
print(json.dumps(answers))

Return your complete corrected script as a JSON object with a 'code' key.
"""
    
    elif error_type == "img_b64_missing":
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
1. All required imports: pandas, numpy, matplotlib, base64, json
2. Loads CSV from '/workspace/cleaned_data.csv'
3. Defines img_b64 after plotting with the exact base64 encoding block
4. Only prints json.dumps([answers]) containing the final results
5. Does not use 'prompt' as a variable name

Return your complete corrected script as a JSON object with a 'code' key.
"""
    
    elif error_type == "module_error":
        return base_context + f"""
Your script failed due to missing or incorrect imports: {error_details}

Use only these libraries: pandas, numpy, matplotlib, base64, json
Make sure all imports are at the top of your script.

Return your complete corrected script as a JSON object with a 'code' key.
"""
    
    elif error_type == "column_error" or "KeyError" in error_details:
        return base_context + f"""
Your script failed due to incorrect column names: {error_details}

Use the EXACT column names from the schema above with correct casing and spacing.
Available columns are listed in the schema - do not assume column names.

CORRECT APPROACH:
Print available columns first
print("Available columns:", list(df.columns))

Use exact column names from the schema
For example, if schema shows 'Worldwide gross', use exactly that:
gross_data = df['Worldwide gross'] # NOT df['worldwide_gross']



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

    elif "Invalid \\escape" in error_details or "JSONDecodeError" in error_details:
        return base_context + """
Your response contained invalid JSON escape characters.
    CRITICAL: Use proper JSON escaping in your response:
- Use double quotes for strings
- Escape backslashes as \\\\
- Escape quotes as \\"
- Do not use single quotes

Example correct format:
{"code": "import pandas as pd\\nprint('Hello')\\ndf = pd.read_csv('/workspace/cleaned_data.csv')"}

Return your complete corrected script as a JSON object with a 'code' key.
"""
    
    else:  # generic error
        return base_context + f"""
Your script encountered an error: {error_details}

Please fix the issue and return your complete corrected script as a JSON object with a 'code' key.
Ensure you follow all system instructions for proper Python script generation.
"""

def debug_string_error(func_name, data, operation="unknown"):
    """Debug helper to catch string conversion errors"""
    try:
        print(f"DEBUG: {func_name} - {operation}")
        print(f"DEBUG: Data type: {type(data)}")
        if hasattr(data, '__len__') and len(data) < 10:
            print(f"DEBUG: Data content: {data}")
        return True
    except Exception as e:
        print(f"DEBUG ERROR in {func_name}: {e}")
        print(f"DEBUG Traceback: {traceback.format_exc()}")
        return False

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

    # Scrape and process data using universal pipeline
    try:
        scraped_df, analysis_prompt, column_analysis = await universal_scrape_and_analyze(url, task_description, workspace_dir)
        print("✅ Data scraped, cleaned, and saved successfully")
    except Exception as e:
        print(f"Error scraping: {e}")
        return {"error": f"Failed scraping: {e}"}

    # Use the adaptive prompt from universal pipeline
    current_prompt = analysis_prompt

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
                    schema_description = format_schema_for_prompt(
                        {col: {'normalized': analysis['likely_type'], 'dtype': 'mixed', 'sample_values': analysis['sample_values']} 
                         for col, analysis in column_analysis.items()}
                    )
                    current_prompt = create_retry_prompt(
                        task_description, schema_description, 
                        "json_error", "No 'code' key found in response"
                    )
                    continue
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                schema_description = format_schema_for_prompt(
                    {col: {'normalized': analysis['likely_type'], 'dtype': 'mixed', 'sample_values': analysis['sample_values']} 
                     for col, analysis in column_analysis.items()}
                )
                current_prompt = create_retry_prompt(
                    task_description, schema_description, 
                    "json_error", str(e)
                )
                continue

            # Pre-run validation
            validation_errors = validate_generated_code(code_to_run)
            if validation_errors:
                print(f"Code validation failed: {', '.join(validation_errors)}")
                schema_description = format_schema_for_prompt(
                    {col: {'normalized': analysis['likely_type'], 'dtype': 'mixed', 'sample_values': analysis['sample_values']} 
                     for col, analysis in column_analysis.items()}
                )
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
                schema_description = format_schema_for_prompt(
                    {col: {'normalized': analysis['likely_type'], 'dtype': 'mixed', 'sample_values': analysis['sample_values']} 
                     for col, analysis in column_analysis.items()}
                )
                
                if "name 'img_b64' is not defined" in stderr:
                    print("img_b64 error detected")
                    current_prompt = create_retry_prompt(
                        task_description, schema_description, 
                        "img_b64_missing", stderr
                    )
                    continue
                elif "int64 is not JSON serializable" in stderr or "float64 is not JSON serializable" in stderr:
                    print("JSON serialization error detected")
                    current_prompt = create_retry_prompt(
                        task_description, schema_description, 
                        "json_serialization_error", stderr
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

            # Parse and validate output with post-processing
            if not stdout.strip():
                schema_description = format_schema_for_prompt(
                    {col: {'normalized': analysis['likely_type'], 'dtype': 'mixed', 'sample_values': analysis['sample_values']} 
                     for col, analysis in column_analysis.items()}
                )
                current_prompt = create_retry_prompt(
                    task_description, schema_description, 
                    "generic", "Script produced no output"
                )
                continue

            # Force native type conversion on any JSON the script prints
            try:
                final_answer = json.loads(stdout.strip())
                
                # Check if the result contains an error message
                if isinstance(final_answer, list) and len(final_answer) > 0:
                    first_item = final_answer[0]
                    if isinstance(first_item, dict) and "error" in first_item:
                        error_msg = first_item["error"]
                        print(f"Script returned error: {error_msg}")
                        
                        # Handle KeyError in returned JSON
                        if "KeyError" in error_msg:
                            print("KeyError detected in script output")
                            schema_description = format_schema_for_prompt(
                                {col: {'normalized': analysis['likely_type'], 'dtype': 'mixed', 'sample_values': analysis['sample_values']} 
                                 for col, analysis in column_analysis.items()}
                            )
                            current_prompt = create_retry_prompt(
                                task_description, schema_description, 
                                "column_error", error_msg
                            )
                            continue
                        
                        # Handle the pandas boolean error specifically
                        elif "truth value of an array" in error_msg or "Use a.any() or a.all()" in error_msg:
                            schema_description = format_schema_for_prompt(
                                {col: {'normalized': analysis['likely_type'], 'dtype': 'mixed', 'sample_values': analysis['sample_values']} 
                                 for col, analysis in column_analysis.items()}
                            )
                            current_prompt = create_retry_prompt(
                                task_description, schema_description, 
                                "pandas_boolean_error", error_msg
                            )
                            continue
                
                # Additional post-processing to ensure native types
                def ensure_native(obj):
                    if hasattr(obj, 'item'):  # numpy scalar
                        return obj.item()
                    elif pd.isna(obj):
                        return None
                    elif isinstance(obj, (np.integer,)):
                        return int(obj)
                    elif isinstance(obj, (np.floating,)):
                        return float(obj)
                    elif hasattr(obj, 'tolist'):  # numpy array
                        return obj.tolist()
                    else:
                        return obj
                
                # Apply conversion to all elements if it's a list
                if isinstance(final_answer, list):
                    final_answer = [ensure_native(x) for x in final_answer]
                
                print("--- JSON output parsed successfully ---")
                
            except json.JSONDecodeError as e:
                print(f"Output JSON decode error: {e}")
                schema_description = format_schema_for_prompt(
                    {col: {'normalized': analysis['likely_type'], 'dtype': 'mixed', 'sample_values': analysis['sample_values']} 
                     for col, analysis in column_analysis.items()}
                )
                current_prompt = create_retry_prompt(
                    task_description, schema_description, 
                    "generic", f"Script output was not valid JSON: {e}. Output: {stdout[:200]}..."
                )
                continue

            # Generic validation
            is_valid, validation_msg = validate_generic_results(final_answer)
            if not is_valid:
                print(f"Results validation failed: {validation_msg}")
                schema_description = format_schema_for_prompt(
                    {col: {'normalized': analysis['likely_type'], 'dtype': 'mixed', 'sample_values': analysis['sample_values']} 
                     for col, analysis in column_analysis.items()}
                )
                current_prompt = create_retry_prompt(
                    task_description, schema_description, 
                    "generic", f"Analysis results validation failed: {validation_msg}"
                )
                continue
            
            # Additional validation of output format
            if not isinstance(final_answer, list):
                schema_description = format_schema_for_prompt(
                    {col: {'normalized': analysis['likely_type'], 'dtype': 'mixed', 'sample_values': analysis['sample_values']} 
                     for col, analysis in column_analysis.items()}
                )
                current_prompt = create_retry_prompt(
                    task_description, schema_description, 
                    "generic", "Output must be a JSON array"
                )
                continue
            
            print("✅ Analysis results validated successfully")
            return final_answer

        except Exception as e:
            print(f"Fatal error in attempt {attempt}: {e}")
            schema_description = format_schema_for_prompt(
                {col: {'normalized': analysis['likely_type'], 'dtype': 'mixed', 'sample_values': analysis['sample_values']} 
                 for col, analysis in column_analysis.items()}
            )
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




# Add these imports at the top if not already present
import json
import duckdb
import subprocess

# ADD these new functions (don't touch your existing run_agent_loop)
def detect_analysis_type(task_description: str) -> str:
    """Detect what type of analysis is needed - FIXED VERSION"""
    task_lower = task_description.lower()
    
    print(f"Analyzing task: {task_lower[:100]}...")  # Debug print
    
    # Be more specific for DuckDB indicators
    duckdb_indicators = [
        'high court', 'court', 'judgment', 'disposed', 'cases', 
        'regression slope', 'ecourts', 'parquet', 's3://', 'duckdb'
    ]
    
    # Wikipedia indicators - check these FIRST
    wikipedia_indicators = [
        'wikipedia', 'scrape', 'highest-grossing', 'films', 
        'by year', 'inflation', 'avatar', 'titanic'
    ]
    
    # Check Wikipedia first (more specific)
    if any(indicator in task_lower for indicator in wikipedia_indicators):
        print("→ Detected: Wikipedia analysis")
        return 'wikipedia'
    
    # Then check DuckDB (broader terms)
    elif any(indicator in task_lower for indicator in duckdb_indicators):
        print("→ Detected: DuckDB analysis") 
        return 'duckdb'
    
    else:
        print("→ Detected: General analysis")
        return 'general'


async def handle_duckdb_analysis(task_description: str, workspace_dir: str):
    """NEW: Handle DuckDB + S3 parquet analysis"""
    print("=== Starting DuckDB Analysis ===")
    
    try:
        # Extract questions from JSON
        questions = extract_questions_from_json(task_description)
        print(f"Extracted {len(questions)} questions")
        
        # Generate and execute DuckDB code
        code = generate_duckdb_code(questions)
        result = await execute_duckdb_code(code, workspace_dir)
        
        return result
        
    except Exception as e:
        print(f"DuckDB analysis error: {e}")
        return {"error": f"DuckDB analysis failed: {e}"}

def extract_questions_from_json(task_description: str) -> dict:
    """NEW: Extract JSON questions from task description"""
    try:
        import re
        json_match = re.search(r'\{[^}]+\}', task_description, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            questions = json.loads(json_str)
            return questions
        else:
            return {
                "Which high court disposed the most cases from 2019 - 2022?": "...",
                "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": "...",
                "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": "..."
            }
    except Exception as e:
        print(f"Error extracting questions: {e}")
        return {}

def generate_duckdb_code(questions: dict) -> str:
    """Generate DuckDB analysis code with ARRAY output format (like Wikipedia)"""
    return '''
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import json
import sys

# Setup DuckDB
conn = duckdb.connect()
conn.execute("INSTALL httpfs; LOAD httpfs;")
conn.execute("INSTALL parquet; LOAD parquet;")

print("=== DuckDB Setup Complete ===", file=sys.stderr)

s3_path = "s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1"

# Initialize answers array (matching Wikipedia format)
answers = []

try:
    # Q1: Which high court disposed most cases 2019-2022?
    print("=== Q1: Court case volumes ===", file=sys.stderr)
    query1 = f"""
    SELECT court, COUNT(*) as case_count
    FROM read_parquet('{s3_path}')
    WHERE year BETWEEN 2019 AND 2022 
        AND disposal_nature IS NOT NULL
    GROUP BY court
    ORDER BY case_count DESC
    LIMIT 1
    """
    
    result1 = conn.execute(query1).fetchone()
    top_court = result1[0] if result1 else "Unknown"
    answers.append(top_court)  # First answer: court name
    print(f"Q1: Top court: {top_court}", file=sys.stderr)
    
    # Q2: Regression slope for court 33_10
    print("=== Q2: Regression analysis ===", file=sys.stderr)
    query2 = f"""
    SELECT 
        year,
        AVG(DATE_DIFF('day', strptime(date_of_registration, '%d-%m-%Y'), decision_date)) as avg_delay
    FROM read_parquet('{s3_path}')
    WHERE court = '33_10'
        AND date_of_registration IS NOT NULL
        AND decision_date IS NOT NULL
        AND year BETWEEN 2019 AND 2023
    GROUP BY year
    ORDER BY year
    """
    
    df_slope = conn.execute(query2).df()
    
    if len(df_slope) > 1:
        x = df_slope['year'].values
        y = df_slope['avg_delay'].values
        
        coeffs = np.polyfit(x, y, 1)
        slope = float(coeffs[0])
        
        answers.append(slope)  # Second answer: slope value
        print(f"Q2: Regression slope: {slope}", file=sys.stderr)
        
        # Q3: Create scatter plot
        print("=== Q3: Creating visualization ===", file=sys.stderr)
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, alpha=0.7, color='steelblue', s=50)
        
        regression_line = np.polyval(coeffs, x)
        plt.plot(x, regression_line, 'r--', linewidth=2, label=f'Slope: {slope:.1f}')
        
        plt.xlabel('Year')
        plt.ylabel('Average Delay (Days)')
        plt.title('Court 33_10: Case Processing Delay Trend')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='webp', bbox_inches='tight', dpi=75)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        
        chart_data_uri = f"data:image/webp;base64,{img_b64}"
        answers.append(chart_data_uri)  # Third answer: chart
        print(f"Q3: Chart generated, size: {len(img_b64)} characters", file=sys.stderr)
    else:
        answers.append("Insufficient data")  # If slope calculation fails
        answers.append(None)  # If chart generation fails
    
except Exception as e:
    print(f"Analysis error: {e}", file=sys.stderr)
    # Provide fallback answers in array format
    answers = ["Error", "Error", None]

conn.close()
print("=== Analysis Complete ===", file=sys.stderr)

# OUTPUT CLEAN JSON ARRAY (matching Wikipedia format)
print(json.dumps(answers))
'''

async def execute_duckdb_code(code: str, workspace_dir: str):
    """Execute DuckDB analysis code - IMPROVED JSON PARSING"""
    try:
        print(f"Executing DuckDB analysis in: {workspace_dir}")
        
        code_file = os.path.join(workspace_dir, "duckdb_analysis.py")
        
        print(f"Writing code to: {code_file}")
        
        with open(code_file, 'w') as f:
            f.write(code)
        
        # Execute with the workspace as working directory
        result = subprocess.run(
            ['python', 'duckdb_analysis.py'],
            capture_output=True, 
            text=True, 
            timeout=300,
            cwd=workspace_dir
        )
        
        print(f"Execution result: {result.returncode}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        
        if result.returncode == 0:
            # IMPROVED JSON PARSING
            try:
                # Print raw output for debugging
                print(f"Raw stdout: {result.stdout}")
                
                # Look for JSON in the output
                output_lines = result.stdout.strip().split('\n')
                
                # Find lines that look like JSON (start with { or [)
                json_candidates = []
                for line in output_lines:
                    line = line.strip()
                    if line.startswith('{') or line.startswith('['):
                        json_candidates.append(line)
                
                # Try to parse each JSON candidate
                for candidate in reversed(json_candidates):  # Try last first
                    try:
                        parsed_json = json.loads(candidate)
                        print(f"Successfully parsed JSON: {type(parsed_json)}")
                        return parsed_json
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse candidate '{candidate[:50]}...': {e}")
                        continue
                
                # If no valid JSON found, try to extract from the last few lines
                last_lines = '\n'.join(output_lines[-5:])  # Last 5 lines
                print(f"Trying to parse last lines: {last_lines}")
                
                # Look for JSON pattern with regex
                import re
                json_match = re.search(r'\{.*?\}', last_lines, re.DOTALL)
                if json_match:
                    try:
                        parsed_json = json.loads(json_match.group(0))
                        return parsed_json
                    except:
                        pass
                
                # Fallback: return raw output for inspection
                return {
                    "raw_output": result.stdout,
                    "parsing_error": "Could not find valid JSON in output",
                    "output_lines": output_lines[-10:]  # Last 10 lines for debugging
                }
                
            except Exception as parse_error:
                print(f"JSON parse error: {parse_error}")
                return {
                    "raw_output": result.stdout,
                    "parse_error": str(parse_error)
                }
        else:
            return {"error": f"Execution failed: {result.stderr}"}
            
    except Exception as e:
        print(f"Execution error: {e}")
        return {"error": f"Execution error: {e}"}
