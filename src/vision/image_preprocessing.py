from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SIZE = (224, 224)


def load_image(
    image_path: Path
) -> np.ndarray:
    """Load an image and convert it to RGB."""

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    return np.array(image)


def resize_image(
    image: np.ndarray
) -> np.ndarray:
    """Resize an image to the configured size."""

    pil_image = Image.fromarray(
        image
    )

    resized_image = pil_image.resize(
        IMAGE_SIZE
    )

    return np.array(
        resized_image
    )


def normalize_image(
    image: np.ndarray
) -> np.ndarray:
    """Normalize pixel values from 0-255 to 0-1."""

    return image.astype(
        np.float32
    ) / 255.0


def preprocess_image(
    image_path: Path
) -> np.ndarray:
    """Complete image preprocessing pipeline."""

    image = load_image(
        image_path
    )

    image = resize_image(
        image
    )

    image = normalize_image(
        image
    )

    return image


def inspect_preprocessing(
    image_path: Path
) -> None:
    """Display preprocessing information for one image."""

    original = load_image(
        image_path
    )

    processed = preprocess_image(
        image_path
    )

    print(
        "\n========== IMAGE PREPROCESSING =========="
    )

    print(
        f"Image: {image_path.name}"
    )

    print(
        f"Original shape: {original.shape}"
    )

    print(
        f"Processed shape: {processed.shape}"
    )

    print(
        f"Processed data type: {processed.dtype}"
    )

    print(
        f"Minimum pixel value: {processed.min():.4f}"
    )

    print(
        f"Maximum pixel value: {processed.max():.4f}"
    )


if __name__ == "__main__":

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    train_good_dir = (
        project_root
        / "data"
        / "raw"
        / "images"
        / "screw"
        / "train"
        / "good"
    )

    image_files = sorted(
        train_good_dir.glob("*.png")
    )

    if not image_files:
        raise FileNotFoundError(
            "No training images found."
        )

    sample_image = image_files[0]

    inspect_preprocessing(
        sample_image
    )