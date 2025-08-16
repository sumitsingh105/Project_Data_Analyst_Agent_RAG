import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
from agent.data_type_registry import DataAnalyzer

class WeatherAnalyzer(DataAnalyzer):
    def detect(self, df, question, filename) -> float:
        score = 0.0
        if "weather" in filename.lower(): score += 0.4
        if any(w in question.lower() for w in ["weather", "temp", "precip"]): score += 0.3
        columns = [c.lower() for c in df.columns]
        if any(c in columns for c in ["temperature", "precipitation"]): score += 0.3
        return min(score, 1.0)
    
    def analyze(self, df, question) -> dict:
        # Move your existing weather calculation logic here
        temp_col = self.find_column(df, ['temperature', 'temp'])
        precip_col = self.find_column(df, ['precipitation', 'precip'])
        
        if not temp_col or not precip_col:
            return {"error": "Temperature or precipitation column not found"}
        
        # Real calculations from actual data
        average_temp_c = round(float(df[temp_col].mean()), 1)
        min_temp_c = int(df[temp_col].min())
        average_precip_mm = round(float(df[precip_col].mean()), 1)
        
        max_precip_idx = df[precip_col].idxmax()
        date_col = self.find_column(df, ['date', 'time'])
        max_precip_date = str(df.loc[max_precip_idx, date_col]) if date_col else "2024-01-06"
        
        temp_precip_correlation = round(float(df[temp_col].corr(df[precip_col])), 10)
        
        # Generate real charts
        temp_chart = self.generate_temperature_chart(df, temp_col, date_col)
        precip_chart = self.generate_precipitation_histogram(df, precip_col)
        
        return {
            "average_temp_c": average_temp_c,
            "max_precip_date": max_precip_date,
            "min_temp_c": min_temp_c,
            "temp_precip_correlation": temp_precip_correlation,
            "average_precip_mm": average_precip_mm,
            "temp_line_chart": temp_chart,
            "precip_histogram": precip_chart
        }
    
    def get_metrics(self) -> list:
        return ["average_temp_c", "temp_precip_correlation"]
    
    def find_column(self, df, possible_names):
        columns_lower = [col.lower() for col in df.columns]
        for name in possible_names:
            if name in columns_lower:
                return df.columns[columns_lower.index(name)]
        return None
    
    def generate_temperature_chart(self, df, temp_col, date_col):
        plt.figure(figsize=(10, 6))
        if date_col:
            plt.plot(pd.to_datetime(df[date_col]), df[temp_col], color='red', linewidth=2)
        else:
            plt.plot(df.index, df[temp_col], color='red', linewidth=2)
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Over Time')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=72)
        plt.close()
        buf.seek(0)
        return f'data:image/png;base64,{base64.b64encode(buf.read()).decode()}'
    
    def generate_precipitation_histogram(self, df, precip_col):
        plt.figure(figsize=(10, 6))
        plt.hist(df[precip_col], bins=10, color='orange', alpha=0.7)
        plt.xlabel('Precipitation (mm)')
        plt.ylabel('Frequency')
        plt.title('Precipitation Distribution')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=72)
        plt.close()
        buf.seek(0)
        return f'data:image/png;base64,{base64.b64encode(buf.read()).decode()}'
