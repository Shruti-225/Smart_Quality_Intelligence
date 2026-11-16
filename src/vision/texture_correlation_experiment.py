
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "outputs"
    / "reports"
    / "vision"
    / "local_texture_feature_values.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "reports" / "vision"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "lbp_entropy",
    "lbp_uniformity",
    "gradient_mean",
    "gradient_std",
    "gradient_p90",
    "local_contrast_mean",
    "local_contrast_std",
    "local_contrast_p90",
    "edge_density",
    "edge_count",
]


# ============================================================
# LOAD DATA
# ============================================================

print("\n========== TEXTURE CORRELATION ANALYSIS ==========\n")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input report not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)
print("Available columns:", list(df.columns))

group_column = next(
    (col for col in df.columns if col.lower() == "group"),
    None
)

if group_column is None:
    raise ValueError(
        "Could not find the Group column. "
        "Check the printed column names."
    )

missing = [f for f in FEATURES if f not in df.columns]

if missing:
    raise ValueError(f"Missing feature columns: {missing}")

# Use only Training Fit Good for the primary
# feature-redundancy analysis.
training = df[
    df[group_column].astype(str).str.strip().str.lower()
    == "training fit good"
].copy()

if len(training) == 0:
    raise ValueError(
        "No Training Fit Good rows found. "
        "Check the group names in the CSV."
    )

training[FEATURES] = training[FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
)

training = training.dropna(subset=FEATURES)

print("\nTraining Fit Good images:", len(training))


# ============================================================
# FEATURE VARIANCE
# ============================================================

print("\n========== FEATURE VARIANCE ==========\n")

variance = training[FEATURES].var().sort_values(
    ascending=False
)

variance_df = variance.reset_index()
variance_df.columns = ["feature", "variance"]

print(variance_df.to_string(index=False))

variance_df.to_csv(
    OUTPUT_DIR / "texture_feature_variance.csv",
    index=False
)


# ============================================================
# CORRELATION MATRIX
# ============================================================

correlation = training[FEATURES].corr(
    method="spearman"
)

print("\n========== SPEARMAN CORRELATION MATRIX ==========\n")
print(correlation.round(3).to_string())

correlation.to_csv(
    OUTPUT_DIR / "texture_feature_correlation.csv"
)


# ============================================================
# CORRELATED FEATURE PAIRS
# ============================================================

pairs = []

for i in range(len(FEATURES)):
    for j in range(i + 1, len(FEATURES)):

        feature_a = FEATURES[i]
        feature_b = FEATURES[j]

        value = correlation.loc[feature_a, feature_b]

        pairs.append({
            "feature_1": feature_a,
            "feature_2": feature_b,
            "correlation": value,
            "absolute_correlation": abs(value),
        })

pairs_df = pd.DataFrame(pairs)

pairs_df = pairs_df.sort_values(
    "absolute_correlation",
    ascending=False
)

print("\n========== TOP CORRELATED PAIRS ==========\n")
print(pairs_df.head(15).round(4).to_string(index=False))

high_pairs = pairs_df[
    pairs_df["absolute_correlation"] >= 0.85
]

print("\n========== HIGH REDUNDANCY PAIRS (|r| >= 0.85) ==========\n")

if len(high_pairs):
    print(high_pairs.round(4).to_string(index=False))
else:
    print("No feature pairs crossed the 0.85 threshold.")

pairs_df.to_csv(
    OUTPUT_DIR / "texture_feature_correlation_pairs.csv",
    index=False
)


# ============================================================
# CORRELATION HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(12, 9))

image = ax.imshow(
    correlation.values,
    aspect="auto",
    vmin=-1,
    vmax=1,
    cmap="coolwarm"
)

ax.set_xticks(range(len(FEATURES)))
ax.set_yticks(range(len(FEATURES)))

ax.set_xticklabels(
    FEATURES,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(FEATURES)

for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        ax.text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=7
        )

fig.colorbar(image, ax=ax, label="Spearman correlation")

ax.set_title(
    "Local Texture Feature Correlation\nTraining Fit Good Images"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "texture_feature_correlation_heatmap.png",
    dpi=200
)

plt.close(fig)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========== SUMMARY ==========\n")

print("Features analysed:", len(FEATURES))
print("Training images analysed:", len(training))
print("High-redundancy pairs:", len(high_pairs))

print("\nReports saved:")

print(OUTPUT_DIR / "texture_feature_variance.csv")
print(OUTPUT_DIR / "texture_feature_correlation.csv")
print(OUTPUT_DIR / "texture_feature_correlation_pairs.csv")
print(OUTPUT_DIR / "texture_feature_correlation_heatmap.png")

print("\nEXPERIMENT COMPLETE")
print("No production model or saved model artifacts were changed.")
