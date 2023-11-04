import pandas as pd

df = pd.read_csv('analyze2s.csv')
print(df['time'].sum())
