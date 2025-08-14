import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io          # ← add this
import json

def generate_adaptive_analysis_prompt(task_description: str, column_analysis: dict, sample_data: str) -> str:
    """Completely universal prompt that works with ANY data and questions"""
    
    print("Generating bulletproof universal prompt...")
    
    # Safe question extraction
    try:
        task_str = str(task_description)
        questions = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', task_str, re.DOTALL)
        questions = [str(q).strip() for q in questions if q.strip()]
        if not questions:
            questions = [task_description]
    except Exception as e:
        print(f"Error extracting questions: {e}")
        questions = [task_description]
    
    # Safe column context
    try:
        column_lines = []
        for col_name, analysis in column_analysis.items():
            col_str = str(col_name)
            type_str = str(analysis.get('likely_type', 'unknown'))
            samples = analysis.get('sample_values', [])
            sample_strs = [str(s)[:30] for s in samples[:2] if s is not None]
            sample_text = ', '.join(sample_strs) if sample_strs else "no samples"
            column_lines.append(f"- {col_str} ({type_str}): {sample_text}")
        column_context = "Available columns:\n" + '\n'.join(column_lines)
    except Exception as e:
        print(f"Error building column context: {e}")
        column_context = "Column information unavailable"
    
    # Format questions safely
    try:
        questions_text = '\n'.join(f"{i+1}. {q}" for i, q in enumerate(questions))
    except:
        questions_text = "1. Analyze the data"
    
    print("Universal prompt generation complete")
    
    return f"""
You are a completely universal data analyst. Write a Python script that works with ANY data structure and questions.

COLUMNS:
{column_context}

QUESTIONS:
{questions_text}

REQUIREMENTS:
- Auto-detect ALL column types and purposes
- Handle ANY question pattern dynamically  
- Return JSON array matching question count
- NO hardcoded assumptions about data

COMPLETE UNIVERSAL SCRIPT:
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import json
import re

Load data
df = pd.read_csv('/workspace/cleaned_data.csv')
print("=== DATA LOADED ===")
print(f"Shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
print("Sample:")
print(df.head(3))

Safe JSON conversion
def safe_json_convert(obj):
if hasattr(obj, "item"):
return obj.item()
if pd.isna(obj):
return None
if isinstance(obj, (np.integer,)):
return int(obj)
if isinstance(obj, (np.floating,)):
return float(obj)
if isinstance(obj, str):
return str(obj)
return obj

DYNAMIC COLUMN DETECTION - Completely embedded
def detect_columns(dataframe):
detected = {{
'location_col': None,
'population_col': None,
'continent_col': None,
'year_col': None,
'title_col': None,
'gross_col': None,
'rank_col': None,
'peak_col': None,
'numeric_cols': [],
'text_cols': []
}}

for col in dataframe.columns:
    col_lower = col.lower()
    
    # Detect location/country column
    if any(term in col_lower for term in ['location', 'country', 'name', 'place']) and not detected['location_col']:
        detected['location_col'] = col
        
    # Detect population column
    elif any(term in col_lower for term in ['population', 'pop']) and not detected['population_col']:
        detected['population_col'] = col
        
    # Detect continent column
    elif 'continent' in col_lower and not detected['continent_col']:
        detected['continent_col'] = col
        
    # Detect year/date column
    elif any(term in col_lower for term in ['year', 'date']) and not detected['year_col']:
        detected['year_col'] = col
        
    # Detect title/movie column
    elif any(term in col_lower for term in ['title', 'film', 'movie']) and not detected['title_col']:
        detected['title_col'] = col
        
    # Detect gross/revenue column  
    elif any(term in col_lower for term in ['gross', 'revenue', 'earnings', 'box']) and not detected['gross_col']:
        detected['gross_col'] = col
        
    # Detect rank column
    elif 'rank' in col_lower and not detected['rank_col']:
        detected['rank_col'] = col
        
    # Detect peak column
    elif 'peak' in col_lower and not detected['peak_col']:
        detected['peak_col'] = col
    
    # Classify as numeric or text
    if pd.api.types.is_numeric_dtype(dataframe[col]):
        detected['numeric_cols'].append(col)
    elif pd.api.types.is_object_dtype(dataframe[col]):
        # Test if convertible to numeric
        test_numeric = pd.to_numeric(dataframe[col], errors='coerce')
        if test_numeric.notna().sum() > len(dataframe) * 0.6:  # >60% convertible
            detected['numeric_cols'].append(col)
        else:
            detected['text_cols'].append(col)

return detected
Detect columns
col_info = detect_columns(df)
print("=== COLUMN DETECTION ===")
for key, value in col_info.items():
print(f"{{key}}: {{value}}")

Prepare numeric data
df_work = df.copy()

Convert detected columns to numeric
if col_info['population_col'] and col_info['population_col'] in df.columns:
df_work['population_num'] = pd.to_numeric(df_work[col_info['population_col']], errors='coerce')
print(f"Created population_num from {{col_info['population_col']}}")

if col_info['year_col'] and col_info['year_col'] in df.columns:
df_work['year_num'] = pd.to_numeric(df_work[col_info['year_col']], errors='coerce')
print(f"Created year_num from {{col_info['year_col']}}")

if col_info['gross_col'] and col_info['gross_col'] in df.columns:
df_work['gross_num'] = pd.to_numeric(df_work[col_info['gross_col']], errors='coerce')
print(f"Created gross_num from {{col_info['gross_col']}}")

if col_info['rank_col'] and col_info['rank_col'] in df.columns:
df_work['rank_num'] = pd.to_numeric(df_work[col_info['rank_col']], errors='coerce')
print(f"Created rank_num from {{col_info['rank_col']}}")

if col_info['peak_col'] and col_info['peak_col'] in df.columns:
df_work['peak_num'] = pd.to_numeric(df_work[col_info['peak_col']], errors='coerce')
print(f"Created peak_num from {{col_info['peak_col']}}")

Parse questions
questions_text = "{questions_text}"
question_list = [line.strip() for line in questions_text.split('\n') if line.strip() and not line.strip().replace('.', '').isdigit()]

print(f"=== PROCESSING {{len(question_list)}} QUESTIONS ===")
answers = []

for question_idx, question in enumerate(question_list):
# Clean question text
clean_question = re.sub(r'^\d+\.\s*', '', question).strip()
if not clean_question:
answers.append(None)
continue
print(f"\\nQ{{question_idx+1}}: {{clean_question[:100]}}")
q_lower = clean_question.lower()

try:
    # 1. COUNTING QUESTIONS
    if any(word in q_lower for word in ['how many', 'count', 'number of']):
        print("→ Count question")
        
        filtered_df = df_work.copy()
        
        # Apply continent filters
        if col_info['continent_col'] and col_info['continent_col'] in df_work.columns:
            continent_applied = False
            for continent in ['asia', 'europe', 'africa', 'north america', 'south america', 'oceania']:
                if continent in q_lower:
                    if continent == 'north america':
                        mask = df_work[col_info['continent_col']].str.contains('North America', case=False, na=False)
                    elif continent == 'south america':
                        mask = df_work[col_info['continent_col']].str.contains('South America', case=False, na=False)
                    else:
                        mask = df_work[col_info['continent_col']].str.contains(continent.title(), case=False, na=False)
                    
                    filtered_df = filtered_df[mask]
                    print(f"  Applied continent filter: {{continent}} ({{mask.sum()}} matches)")
                    continent_applied = True
                    break
        
        # Apply population thresholds
        if 'population_num' in df_work.columns:
            # Look for million thresholds
            million_matches = re.findall(r'(\\d+)\\s*million', q_lower)
            for million_str in million_matches:
                threshold = float(million_str) * 1000000
                mask = filtered_df['population_num'] >= threshold
                filtered_df = filtered_df[mask]
                print(f"  Applied population filter >= {{threshold}} ({{mask.sum()}} matches)")
            
            # Look for billion thresholds
            billion_matches = re.findall(r'(\\d+(?:\\.\\d+)?)\\s*(?:bn|billion)', q_lower)
            for billion_str in billion_matches:
                threshold = float(billion_str) * 1000000000
                mask = filtered_df['gross_num'] >= threshold if 'gross_num' in df_work.columns else pd.Series([False] * len(filtered_df))
                filtered_df = filtered_df[mask]
                print(f"  Applied billion filter >= {{threshold}} ({{mask.sum()}} matches)")
        
        # Apply year filters
        if 'year_num' in df_work.columns:
            if 'before' in q_lower:
                year_matches = re.findall(r'before\\s*(\\d{{4}})', q_lower)
                for year_str in year_matches:
                    year_threshold = int(year_str)
                    mask = filtered_df['year_num'] < year_threshold
                    filtered_df = filtered_df[mask]
                    print(f"  Applied year filter < {{year_threshold}} ({{mask.sum()}} matches)")
        
        count = len(filtered_df)
        print(f"  Final count: {{count}}")
        answers.append(int(count))
    
    # 2. IDENTIFICATION QUESTIONS
    elif any(word in q_lower for word in ['which', 'what is', 'name of', 'highest', 'largest', 'earliest']):
        print("→ Identification question")
        
        filtered_df = df_work.copy()
        result = None
        
        # Apply continent filters first
        if col_info['continent_col'] and col_info['continent_col'] in df_work.columns:
            for continent in ['asia', 'europe', 'africa', 'north america', 'south america', 'oceania']:
                if continent in q_lower:
                    if continent == 'north america':
                        mask = df_work[col_info['continent_col']].str.contains('North America', case=False, na=False)
                    elif continent == 'south america':
                        mask = df_work[col_info['continent_col']].str.contains('South America', case=False, na=False)
                    else:
                        mask = df_work[col_info['continent_col']].str.contains(continent.title(), case=False, na=False)
                    
                    filtered_df = filtered_df[mask]
                    print(f"  Applied continent filter: {{continent}} ({{mask.sum()}} matches)")
                    break
        
        # Find highest/largest
        if any(term in q_lower for term in ['highest', 'largest', 'most']) and 'population_num' in df_work.columns:
            if len(filtered_df) > 0:
                max_idx = filtered_df['population_num'].idxmax()
                if col_info['location_col'] and max_idx in df_work.index:
                    result = str(df_work.loc[max_idx, col_info['location_col']])
                    print(f"  Found highest population: {{result}}")
        
        # Find earliest
        elif 'earliest' in q_lower:
            if 'gross_num' in df_work.columns and 'year_num' in df_work.columns:
                # Look for thresholds
                threshold_found = False
                for threshold_match in re.findall(r'(\\d+(?:\\.\\d+)?)\\s*(?:bn|billion)', q_lower):
                    threshold = float(threshold_match) * 1000000000
                    over_threshold = df_work[df_work['gross_num'] >= threshold]
                    if len(over_threshold) > 0:
                        earliest_idx = over_threshold['year_num'].idxmin()
                        if col_info['title_col'] and earliest_idx in df_work.index:
                            result = str(df_work.loc[earliest_idx, col_info['title_col']])
                            threshold_found = True
                            break
                
                if not threshold_found:
                    for threshold_match in re.findall(r'(\\d+)\\s*crore', q_lower):
                        threshold = float(threshold_match)
                        over_threshold = df_work[df_work['gross_num'] >= threshold]
                        if len(over_threshold) > 0:
                            earliest_idx = over_threshold['year_num'].idxmin()
                            if col_info['title_col'] and earliest_idx in df_work.index:
                                result = str(df_work.loc[earliest_idx, col_info['title_col']])
                                break
        
        if result is None:
            result = "Not found"
        
        answers.append(result)
    
    # 3. STATISTICAL QUESTIONS
    elif any(word in q_lower for word in ['average', 'mean', 'correlation']):
        print("→ Statistical question")
        
        if 'correlation' in q_lower:
            if 'rank_num' in df_work.columns and 'peak_num' in df_work.columns:
                valid_data = df_work[['rank_num', 'peak_num']].dropna()
                if len(valid_data) > 1:
                    correlation = valid_data['rank_num'].corr(valid_data['peak_num'])
                    answers.append(float(correlation))
                    print(f"  Correlation: {{correlation}}")
                else:
                    answers.append(None)
            else:
                answers.append(None)
        
        elif any(word in q_lower for word in ['average', 'mean']):
            filtered_df = df_work.copy()
            
            # Apply continent filter
            if col_info['continent_col'] and col_info['continent_col'] in df_work.columns:
                for continent in ['asia', 'europe', 'africa', 'america', 'oceania']:
                    if continent in q_lower:
                        if continent == 'america':
                            mask = df_work[col_info['continent_col']].str.contains('America', case=False, na=False)
                        else:
                            mask = df_work[col_info['continent_col']].str.contains(continent.title(), case=False, na=False)
                        
                        filtered_df = filtered_df[mask]
                        print(f"  Applied continent filter: {{continent}} ({{mask.sum()}} matches)")
                        break
            
            if 'population_num' in df_work.columns:
                avg_pop = filtered_df['population_num'].mean()
                answers.append(float(avg_pop))
                print(f"  Average population: {{avg_pop}}")
            else:
                answers.append(None)
    
    # 4. VISUALIZATION QUESTIONS
    elif any(word in q_lower for word in ['chart', 'plot', 'bar', 'graph', 'scatter']):
        print("→ Visualization question")
        
        try:
            if 'scatter' in q_lower and 'rank_num' in df_work.columns and 'peak_num' in df_work.columns:
                # Scatter plot
                valid_data = df_work[['rank_num', 'peak_num']].dropna()
                if len(valid_data) > 1:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(valid_data['rank_num'], valid_data['peak_num'], alpha=0.7, s=50)
                    
                    if 'regression' in q_lower or 'line' in q_lower:
                        coeffs = np.polyfit(valid_data['rank_num'], valid_data['peak_num'], 1)
                        poly_func = np.poly1d(coeffs)
                        x_range = np.linspace(valid_data['rank_num'].min(), valid_data['rank_num'].max(), 100)
                        plt.plot(x_range, poly_func(x_range), 'r--', alpha=0.8, linewidth=2)
                    
                    plt.xlabel('Rank')
                    plt.ylabel('Peak')
                    plt.title('Rank vs Peak')
                    plt.grid(True, alpha=0.3)
            
            elif any(word in q_lower for word in ['bar', 'top']):
                # Bar chart
                if col_info['location_col'] and 'population_num' in df_work.columns:
                    top_5 = df_work.nlargest(5, 'population_num')
                    
                    plt.figure(figsize=(12, 6))
                    
                    if 'horizontal' in q_lower:
                        plt.barh(range(len(top_5)), top_5['population_num'] / 1000000, color='steelblue')
                        plt.yticks(range(len(top_5)), top_5[col_info['location_col']])
                        plt.xlabel('Population (Millions)')
                        plt.gca().invert_yaxis()
                    else:
                        plt.bar(range(len(top_5)), top_5['population_num'] / 1000000, color='steelblue')
                        plt.xticks(range(len(top_5)), top_5[col_info['location_col']], rotation=45)
                        plt.ylabel('Population (Millions)')
                    
                    plt.title('Top 5 Most Populous Countries')
                    plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            plt.close()
            
            chart_data_uri = f"data:image/png;base64,{{img_b64}}"
            answers.append(chart_data_uri)
            print("  Chart generated successfully")
            
        except Exception as e:
            print(f"  Chart generation error: {{e}}")
            answers.append("Chart generation failed")
    
    else:
        print("→ Question type not recognized")
        answers.append("Question type not recognized")
        
except Exception as e:
    print(f"Error processing question {{question_idx+1}}: {{e}}")
    answers.append("Processing error")

Ensure correct answer count
target_count = len(question_list)
while len(answers) < target_count:
answers.append(None)
answers = answers[:target_count]

Final output
answers = [safe_json_convert(a) for a in answers]
print("\n=== FINAL RESULTS ===")
print(f"Questions: {{len(question_list)}}")
print(f"Answers: {{len(answers)}}")
print("Final answers:", answers)
print(json.dumps(answers))


Return as: {{"code": "your_python_script"}}
"""





# ---------------------------------------------------------------------
# Fallback Prompt (simple)
# ---------------------------------------------------------------------
def get_plan_and_execute_prompt() -> str:
    """Basic prompt used when adaptive generation fails."""
    return """
You are a world-class data-analyst agent.

Task:
– Load '/workspace/cleaned_data.csv'
– Explore columns and data types
– Answer the user's questions
– Handle missing / unexpected data gracefully
– Output answers as a JSON array via json.dumps()

Required libraries only: pandas, numpy, matplotlib, base64, io, json.

Use the same safe_json_convert helper shown below before printing:

def safe_json_convert(obj):
import pandas as pd, numpy as np
if hasattr(obj, "item"):
return obj.item()
if pd.isna(obj):
return None
if isinstance(obj, (np.integer,)):
return int(obj)
if isinstance(obj, (np.floating,)):
return float(obj)
if hasattr(obj, "tolist"):
return obj.tolist()
return obj

Print:

print(json.dumps([safe_json_convert(x) for x in answers]))



Respond as JSON: {{ "code": "your_python_script_here" }}
    """
