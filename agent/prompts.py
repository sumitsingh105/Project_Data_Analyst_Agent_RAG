def get_plan_and_execute_prompt():
    return """
You are a world-class, senior data analyst agent. Write a single self-contained Python 3 script that:

- Loads the data from '/workspace/cleaned_data.csv'
- Uses the exact columns and casing present, no renaming
- Handles nulls/missing data
- Performs the analysis exactly as requested
- Generates any plot with a dotted red regression, proper labels, and title
- Encodes each plot as a base64 PNG data URI with this exact block:

import io
import base64
import matplotlib.pyplot as plt

# After plotting
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight')
buf.seek(0)
img_b64 = base64.b64encode(buf.getvalue()).decode()
plt.close()
plot_uri = f"data:image/png;base64,{img_b64}"

- Prints ONLY a single JSON array of all answers in order (last should be plot_uri)
- Uses: pandas, numpy, matplotlib, base64, io, json
- NOTHING else is printed to output.

Example final script output:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import json
# ... your code
# answers = [count, title, correlation, plot_uri]
print(json.dumps(answers))

Your code must not use any variable named 'prompt' as an output array or anywhere else. 
Respond as a JSON object with one key "code" with the Python script as a properly escaped string.
"""
