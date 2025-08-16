from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataAnalyzer(ABC):
    @abstractmethod
    def detect(self, df: pd.DataFrame, question: str, filename: str) -> float:
        pass
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_metrics(self) -> List[str]:
        pass

class DataTypeRegistry:
    def __init__(self):
        self.analyzers: Dict[str, DataAnalyzer] = {}
    
    def register(self, name: str, analyzer: DataAnalyzer):
        self.analyzers[name] = analyzer
        logger.info(f"📝 Registered analyzer: {name}")
    
    def detect_data_type(self, df: pd.DataFrame, question: str, filename: str) -> str:
        best_type = "generic"
        best_score = 0.0
        
        for name, analyzer in self.analyzers.items():
            score = analyzer.detect(df, question, filename)
            if score > best_score:
                best_score = score
                best_type = name
        
        return best_type if best_score > 0.5 else "generic"
    
    def analyze(self, data_type: str, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        if data_type in self.analyzers:
            return self.analyzers[data_type].analyze(df, question)
        else:
            return self.generic_analysis(df, question)
    
    def generic_analysis(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return {
            "data_type": "generic",
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": len(numeric_cols)
        }

registry = DataTypeRegistry()
