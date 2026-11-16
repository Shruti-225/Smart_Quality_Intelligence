
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# PATH CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data/raw/images/screw/test/manipulated_front"
MASK_DIR = ROOT / "data/raw/images/screw/ground_truth/manipulated_front"

OUTPUT_DIR = (
    ROOT / "outputs/reports/vision/screw_foreground/"
    "manipulated_front_inspection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
MORPH_KERNEL_SIZE = 5
IMAGES_PER_PAGE = 8


# ==========================================
# SCREW FOREGROUND SEGMENTATION
# ==========================================

def segment_screw(image):

    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    foreground = np.zeros_like(binary)

    if count > 1:
        largest_label = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        foreground[labels == largest_label] = 255

    return foreground > 0


# ==========================================
# IMAGE LOADING
# ==========================================

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )


def load_mask(path):

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError(f"Could not read mask: {path}")

    mask = cv2.resize(
        mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    return mask > 0


# ==========================================
# MAIN INSPECTION
# ==========================================

def main():

    print("\n========== MANIPULATED FRONT INSPECTION ==========\n")

    image_paths = sorted(IMAGE_DIR.glob("*.png"))

    if not image_paths:
        image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    if not image_paths:
        print("ERROR: No test images found.")
        print("Checked:", IMAGE_DIR)
        return

    print("Images found:", len(image_paths))

    records = []
    errors = []

    for image_path in image_paths:

        mask_path = MASK_DIR / f"{image_path.stem}_mask.png"

        if not mask_path.exists():
            errors.append(
                f"Missing mask: {image_path.name}"
            )
            continue

        try:

            image = load_image(image_path)
            defect_mask = load_mask(mask_path)

            foreground = segment_screw(image)

            outside_defect = (
                defect_mask & (~foreground)
            )

            defect_total = np.sum(defect_mask)

            inside_percent = (
                np.sum(defect_mask & foreground)
                / defect_total * 100
                if defect_total else 0
            )

            outside_percent = (
                np.sum(outside_defect)
                / defect_total * 100
                if defect_total else 0
            )

            # ==================================
            # CREATE COLOR OVERLAY
            # ==================================

            overlay = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

            # Yellow = ground-truth defect pixels
            overlay[defect_mask] = (
                overlay[defect_mask] * 0.45
                + np.array([255, 190, 0]) * 0.55
            ).astype(np.uint8)

            # Red = defect pixels outside foreground
            overlay[outside_defect] = [255, 0, 0]

            records.append({
                "name": image_path.name,
                "image": image,
                "mask": defect_mask,
                "foreground": foreground,
                "overlay": overlay,
                "inside": inside_percent,
                "outside": outside_percent
            })

            print(
                f"{image_path.name}: "
                f"Inside={inside_percent:.2f}% | "
                f"Outside={outside_percent:.2f}%"
            )

        except Exception as error:
            errors.append(
                f"{image_path.name}: {error}"
            )

    print("\nSuccessfully loaded:", len(records))

    if not records:
        print("No images could be processed.")
        return

    # ==========================================
    # GENERATE VISUALIZATION PAGES
    # ==========================================

    total_pages = math.ceil(
        len(records) / IMAGES_PER_PAGE
    )

    for page_number in range(total_pages):

        page_records = records[
            page_number * IMAGES_PER_PAGE:
            (page_number + 1) * IMAGES_PER_PAGE
        ]

        fig, axes = plt.subplots(
            len(page_records),
            4,
            figsize=(14, 3.5 * len(page_records))
        )

        if len(page_records) == 1:
            axes = np.array([axes])

        for row, record in enumerate(page_records):

            axes[row, 0].imshow(
                record["image"],
                cmap="gray"
            )
            axes[row, 0].set_title("Original")

            axes[row, 1].imshow(
                record["mask"],
                cmap="gray"
            )
            axes[row, 1].set_title("Ground-truth defect")

            axes[row, 2].imshow(
                record["foreground"],
                cmap="gray"
            )
            axes[row, 2].set_title("Detected foreground")

            axes[row, 3].imshow(
                record["overlay"]
            )
            axes[row, 3].set_title(
                f"Outside defect: {record['outside']:.1f}%"
            )

            for axis in axes[row]:
                axis.axis("off")

            axes[row, 0].set_ylabel(
                record["name"],
                rotation=0,
                labelpad=65,
                va="center"
            )

        fig.suptitle(
            "Manipulated Front - Foreground Inspection "
            f"(Page {page_number + 1})",
            fontsize=15
        )

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR /
            f"inspection_page_{page_number + 1}.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        print("\nSaved:", output_path)

    # ==========================================
    # SAVE SUMMARY
    # ==========================================

    import pandas as pd

    summary = pd.DataFrame([
        {
            "Image": record["name"],
            "Defect_Inside_Foreground_Percent": record["inside"],
            "Defect_Outside_Foreground_Percent": record["outside"]
        }
        for record in records
    ])

    summary_path = OUTPUT_DIR / "inspection_summary.csv"

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\nSummary saved:", summary_path)

    if errors:
        error_path = OUTPUT_DIR / "inspection_errors.txt"

        error_path.write_text(
            "\n".join(errors),
            encoding="utf-8"
        )

        print("Errors saved:", error_path)

    print("\n========== INSPECTION COMPLETE ==========")
    print("Total images visualized:", len(records))
    print("Total pages:", total_pages)
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
