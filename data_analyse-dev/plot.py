import pandas as pd
import plotly.express as px

df = pd.read_clipboard() # Assuming the dataframe is copied to clipboard
fig = px.line(df, x='TIMESTAMP', y='process_cpu')
fig.show()