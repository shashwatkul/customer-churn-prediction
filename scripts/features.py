import pandas as pd

df = pd.read_csv("data/processed/clean_data.csv")

# Engagement Index
df["engagement_index"] = (
    df["Login_Frequency"]*0.2 +
    df["Session_Duration_Avg"]*0.2 +
    df["Pages_Per_Session"]*0.2 +
    df["Email_Open_Rate"]*0.2 +
    df["Social_Media_Engagement_Score"]*0.2
)

# Risk Score
df["risk_score"] = (
    df["Days_Since_Last_Purchase"]*0.3 +
    df["Cart_Abandonment_Rate"]*0.3 +
    df["Returns_Rate"]*0.2 +
    df["Customer_Service_Calls"]*0.2
)

# Value Score
df["value_score"] = (
    df["Lifetime_Value"]*0.5 +
    df["Total_Purchases"]*0.3 +
    df["Average_Order_Value"]*0.2
)

df.to_csv("data/processed/featured_data.csv", index=False)
print("Feature engineering completed.")
