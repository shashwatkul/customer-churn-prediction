import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

# Load data
df = pd.read_csv("data/processed/featured_data.csv")

# Select features
features = df[["engagement_index", "risk_score", "value_score"]]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(features)

# --- SAMPLE FOR SILHOUETTE (performance fix) ---
X_sample = resample(X, n_samples=min(2000, len(X)), random_state=42)

silhouette_scores = {}
bic_scores = {}

for k in range(2, 8):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X)

    labels_sample = gmm.predict(X_sample)

    silhouette_scores[k] = silhouette_score(X_sample, labels_sample)
    bic_scores[k] = gmm.bic(X)

    print(
        f"Clusters: {k} | "
        f"Silhouette: {silhouette_scores[k]:.3f} | "
        f"BIC: {bic_scores[k]:.0f}"
    )

# Normalize scores
sil_vals = silhouette_scores
bic_vals = bic_scores

max_sil = max(sil_vals.values())

# candidate ks with silhouette close to best
candidates = [
    k for k, v in sil_vals.items()
    if v >= max_sil - 0.02   # tolerance
]

# among candidates, choose lowest BIC
best_k = min(candidates, key=lambda k: bic_vals[k])

print(f"\nBest number of segments selected (hybrid): {best_k}")


# --- Final model ---
final_gmm = GaussianMixture(n_components=best_k, random_state=42)
df["segment"] = final_gmm.fit_predict(X)

# Segment confidence
df["segment_confidence"] = final_gmm.predict_proba(X).max(axis=1)

# Create segment profile
segment_profile = (
    df.groupby("segment")[["engagement_index", "value_score", "risk_score"]]
    .mean()
)

# Rank segments (relative, data-driven)
segment_profile["engagement_rank"] = segment_profile["engagement_index"].rank(ascending=False)
segment_profile["value_rank"] = segment_profile["value_score"].rank(ascending=False)
segment_profile["risk_rank"] = segment_profile["risk_score"].rank(ascending=True)  # low risk = good

# Assign labels
def assign_label(row):
    if row["risk_rank"] == 1 and row["value_rank"] == 1:
        return "Loyal"
    elif row["risk_rank"] == segment_profile["risk_rank"].max():
        return "At Risk"
    else:
        return "Needs Attention"

segment_profile["segment_label"] = segment_profile.apply(assign_label, axis=1)

# Map labels back to main dataframe
segment_label_map = segment_profile["segment_label"].to_dict()
df["segment_label"] = df["segment"].map(segment_label_map)

# Save output
df.to_csv("data/processed/segmented_data.csv", index=False)
print("Segmentation completed with auto-selected segments.")


print(
    df.groupby("segment_label")[[
        "Churned",
        "engagement_index",
        "value_score",
        "risk_score"
    ]].mean()
)
