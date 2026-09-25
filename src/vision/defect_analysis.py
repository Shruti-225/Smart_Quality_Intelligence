from pathlib import Path

import pandas as pd


def get_project_root():
    """Return the project root directory."""

    return Path(__file__).resolve().parents[2]


def analyze_defect_dataset():
    """Analyze test images by defect category."""

    project_root = get_project_root()

    dataset_directory = (
        project_root
        / "data"
        / "raw"
        / "images"
        / "screw"
        / "test"
    )

    defect_categories = {
        "Good": "good",
        "Manipulated Front": "manipulated_front",
        "Scratch Head": "scratch_head",
        "Scratch Neck": "scratch_neck",
        "Thread Side": "thread_side",
        "Thread Top": "thread_top",
    }

    results = []

    print(
        "\n========== DEFECT TYPE ANALYSIS =========="
    )

    print(
        f"\nDataset:\n{dataset_directory}"
    )

    for defect_name, folder_name in defect_categories.items():

        folder_path = (
            dataset_directory
            / folder_name
        )

        if not folder_path.exists():

            print(
                f"\nWARNING: Folder not found:"
                f"\n{folder_path}"
            )

            continue

        image_files = [
            file
            for file in folder_path.iterdir()
            if file.suffix.lower()
            in {".png", ".jpg", ".jpeg"}
        ]

        image_count = len(image_files)

        results.append(
            {
                "defect_type": defect_name,
                "folder": folder_name,
                "image_count": image_count,
            }
        )

        print(
            f"{defect_name:<20}: "
            f"{image_count}"
        )

    results_df = pd.DataFrame(
        results
    )

    output_directory = (
        project_root
        / "outputs"
        / "reports"
        / "vision"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_directory
        / "defect_type_summary.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nReport saved to:\n{output_file}"
    )

    print(
        "\n========== TOTAL IMAGES =========="
    )

    if not results_df.empty:

        total_images = (
            results_df["image_count"].sum()
        )

        print(
            f"Total: {total_images}"
        )

    else:

        print(
            "No valid image folders were found."
        )


if __name__ == "__main__":

    analyze_defect_dataset()