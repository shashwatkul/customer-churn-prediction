import pandas as pd

df = pd.read_csv("data/raw/churn_data.csv")
df.to_csv("data/processed/ingested_data.csv", index=False)

print("Data ingestion completed.")
