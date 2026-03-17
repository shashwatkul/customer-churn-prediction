import pandas as pd

df = pd.read_csv("data/processed/segmented_data.csv")

df.shape
df.columns
df.isnull().sum()
df["Churned"].value_counts(normalize=True)
df.head()
