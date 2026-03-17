import pandas as pd

df = pd.read_csv("data/processed/ingested_data.csv")

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())


df.to_csv("data/processed/clean_data.csv", index=False)
print("Data cleaning completed.")
