from agent.data_type_registry import registry
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)
from agent.analyzers.weather_analyzer import WeatherAnalyzer

# Add this after the registry import
registry.register("weather", WeatherAnalyzer())

def enhanced_universal_analysis(question_text: str, workspace_dir: str):
    try:
        csv_files = [f for f in os.listdir(workspace_dir) if f.endswith('.csv')]
        if not csv_files:
            return {"error": "No CSV file found"}
        
        csv_path = os.path.join(workspace_dir, csv_files[0])
        df = pd.read_csv(csv_path)
        
        data_type = registry.detect_data_type(df, question_text, csv_files)
        logger.info(f"🎯 Detected data type: {data_type}")
        
        result = registry.analyze(data_type, df, question_text)
        result["detected_type"] = data_type
        result["filename"] = csv_files
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Enhanced analysis error: {e}")
        return {"error": f"Analysis failed: {str(e)}"}
