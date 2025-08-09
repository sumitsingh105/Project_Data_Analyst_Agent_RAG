import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import json

# Read the data
df = pd.read_csv('/workspace/cleaned_data.csv')

# Question 1: How many $2 bn movies were released before 2000?
movies_before_2000 = df[df['Year'] < 2000]
billion_movies = len(movies_before_2000[movies_before_2000['Worldwide gross'] > 2000000000])

# Question 2: Which is the earliest film that grossed over $1.5 bn?
early_movies = df[df['Worldwide gross'] > 1500000000].sort_values('Year')
earliest_movie = early_movies.iloc[0]['Year'] if not early_movies.empty else "No movies found"

# Question 3: What's the correlation between Rank and Peak?
correlation = df[['Rank', 'Peak']].corr().iloc[0,1]

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['Rank'], df['Peak'])
plt.xlabel('Rank')
plt.ylabel('Peak')
plt.title('Rank vs Peak')
plt.grid(True, linestyle='--', alpha=0.7)

# Save plot as base64
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight')
buf.seek(0)
plot_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
plt.close()

# Prepare results array
results = [billion_movies, int(earliest_movie), round(correlation, 4), plot_uri]

# Print results as JSON
print(json.dumps(results))
