
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = ROOT / "outputs/reports/vision"
REPORT_PATH = REPORT_DIR / "VISION_EXPERIMENT_FINAL_SUMMARY.md"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(relative_path):
    path = REPORT_DIR / relative_path

    if not path.exists():
        print(f"Not found, skipping: {path}")
        return None

    print(f"Loaded: {path.name}")
    return pd.read_csv(path)


def markdown_table(df, max_rows=30):
    if df is None or df.empty:
        return "_No data available._\n"

    display = df.head(max_rows).copy()

    def format_value(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")

    columns = list(display.columns)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |"
    ]

    for _, row in display.iterrows():
        values = [format_value(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")

    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")

    return "\n".join(lines) + "\n"


def add_section(parts, title, df, max_rows=30):
    parts.append(f"\n## {title}\n")
    parts.append(markdown_table(df, max_rows))


parts = []

parts.append("# Smart Quality Intelligence")
parts.append("# Final Vision Experiment Report")

parts.append("""
## 1. Objective

The vision experiments investigated whether image-based foreground
segmentation and local contrast features can help identify defective
screw regions.

The experiments used the screw image dataset and its available
ground-truth defect masks. All experiments were conducted separately
from the production model.

## 2. Dataset

- Image resolution used: 128 × 128
- Defect categories:
  - manipulated_front
  - scratch_head
  - scratch_neck
  - thread_side
  - thread_top
- Defective images evaluated: 119
- Manipulated-front images evaluated in the focused analysis: 24

## 3. Foreground Segmentation Method Comparison
""")

foreground_overall = load_csv(
    "foreground_method_comparison/foreground_method_overall_summary.csv"
)
foreground_category = load_csv(
    "foreground_method_comparison/foreground_method_category_summary.csv"
)

add_section(parts, "3.1 Overall Method Comparison", foreground_overall)
add_section(parts, "3.2 Category-wise Comparison", foreground_category)

parts.append("""
## 4. Numerical Foreground Diagnostics
""")

diagnostics_overall = load_csv(
    "foreground_method_comparison/numerical_diagnostics/"
    "foreground_diagnostics_overall.csv"
)
diagnostics_category = load_csv(
    "foreground_method_comparison/numerical_diagnostics/"
    "foreground_diagnostics_category.csv"
)

add_section(parts, "4.1 Overall Diagnostics", diagnostics_overall)
add_section(parts, "4.2 Category-wise Diagnostics", diagnostics_category)

parts.append("""
## 5. Manipulated-front Intensity Analysis

The intensity analysis compared defect and non-defect pixel brightness
against each image's Otsu threshold.
""")

intensity = load_csv(
    "foreground_method_comparison/intensity_analysis/"
    "manipulated_front_intensity.csv"
)

if intensity is not None:
    columns = [
        col for col in [
            "Otsu_Threshold",
            "Defect_Mean_Intensity",
            "Normal_Mean_Intensity",
            "Defect_Darker_Percent",
            "Defect_Brighter_Percent",
            "Normal_Darker_Percent",
            "Normal_Brighter_Percent",
        ] if col in intensity.columns
    ]

    if columns:
        intensity_summary = intensity[columns].mean().to_frame(
            "Mean Across Images"
        ).T
        add_section(parts, "5.1 Average Intensity Findings", intensity_summary)

parts.append("""
## 6. Local Contrast Inside Predicted Foreground
""")

inside_foreground = load_csv(
    "foreground_method_comparison/inside_foreground_analysis/"
    "manipulated_front_inside_foreground_summary.csv"
)

add_section(parts, "6.1 Inside-foreground Summary", inside_foreground)

parts.append("""
## 7. Pixel-level Defect Detection Evaluation
""")

pixel_summary = load_csv(
    "foreground_method_comparison/pixel_auc_analysis/"
    "manipulated_front_pixel_auc_summary.csv"
)

pixel_details = load_csv(
    "foreground_method_comparison/pixel_auc_analysis/"
    "manipulated_front_pixel_auc_detailed.csv"
)

add_section(parts, "7.1 Overall Pixel-level Results", pixel_summary)
add_section(parts, "7.2 Per-image Pixel-level Results", pixel_details)

parts.append("""
## 8. Main Experimental Findings

1. The Otsu no-opening variant achieved the highest average defect-pixel
   retention among the four compared foreground methods, at approximately
   87.36% across the 119 defective images.

2. Foreground segmentation remains category-dependent. The
   manipulated_front category achieved only approximately 56.74%
   defect-pixel retention with the no-opening method.

3. Defect pixels inside the predicted manipulated_front foreground had
   higher mean local contrast than non-defect foreground pixels:
   approximately 38.57 compared with 24.81.

4. In the manipulated_front pixel-level experiment, local contrast
   achieved a macro ROC-AUC of 0.9348 and pooled ROC-AUC of 0.8855.
   Raw intensity achieved 0.4095 and 0.3575 respectively.

5. Selecting the top 10% highest-contrast foreground pixels achieved
   macro recall of approximately 84.35%, but macro precision was only
   approximately 10.46%. This is useful for exploratory highlighting,
   not proof of reliable standalone defect segmentation.

## 9. Limitations

- The focused pixel-level experiment covers manipulated_front only.
- Foreground masks are predicted using Otsu thresholding; independent
  ground-truth screw masks were not available to validate segmentation.
- The defect masks identify defective regions, not necessarily the
  complete physical screw boundary.
- The exploratory pixel-level results use labeled test images and must
  not be presented as independent model generalization performance.
- Pixel-level localization metrics do not establish production-level
  image classification performance.
- These experiments do not validate a replacement for the existing
  Isolation Forest production model.

## 10. Final Recommendation

Retain the existing production model and its saved artifacts unchanged.

Keep Otsu no-opening and local contrast as experimental analysis
methods. Local contrast demonstrates promising defect-pixel ranking
performance in manipulated_front, but further independent validation
would be required before deployment.

## 11. Production Model Status

No production model retraining, threshold modification, or dashboard
modification was performed as part of this experiment consolidation.

---

**Project:** Smart Quality Intelligence  
**Report type:** Final Vision Experiment Summary  
**Resolution:** 128 × 128
""")

REPORT_PATH.write_text("\n".join(parts), encoding="utf-8")

print("\n========== REPORT GENERATION COMPLETE ==========")
print(f"Saved report: {REPORT_PATH}")
print(f"Report size: {REPORT_PATH.stat().st_size:,} bytes")
