from pathlib import Path


def get_dataset_paths(project_root: Path) -> dict:
    """Return all important screw dataset paths."""

    screw_root = (
        project_root
        / "data"
        / "raw"
        / "images"
        / "screw"
    )

    return {
        "train_good": screw_root / "train" / "good",
        "test_good": screw_root / "test" / "good",
        "test_manipulated": (
            screw_root / "test" / "manipulated_front"
        ),
        "test_scratch_head": (
            screw_root / "test" / "scratch_head"
        ),
        "test_scratch_neck": (
            screw_root / "test" / "scratch_neck"
        ),
        "test_thread_side": (
            screw_root / "test" / "thread_side"
        ),
        "test_thread_top": (
            screw_root / "test" / "thread_top"
        ),
    }


def count_images(folder: Path) -> int:
    """Count PNG images inside a folder."""

    return len(
        list(folder.glob("*.png"))
    )


def inspect_dataset(project_root: Path) -> None:
    """Print image counts for every category."""

    paths = get_dataset_paths(project_root)

    print("\n========== SCREW DATASET ==========\n")

    total = 0

    for name, path in paths.items():

        image_count = count_images(path)

        total += image_count

        print(
            f"{name:<25} : {image_count}"
        )

    print("\n------------------------------")

    print(
        f"Total images : {total}"
    )


if __name__ == "__main__":

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    inspect_dataset(project_root)