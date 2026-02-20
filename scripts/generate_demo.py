"""Generate synthetic demo data for testing the LEXAI pipeline."""

import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_blood_smear(
    size: int = 224,
    class_name: str = "Normal",
) -> Image.Image:
    """
    Generate a synthetic blood smear image.

    - Background: light pink (plasma)
    - Red circles: RBCs
    - Larger purple circles: WBCs (more in leukemia classes)
    """
    rng = random.Random()

    # Background: light pink plasma
    bg_r = rng.randint(230, 245)
    bg_g = rng.randint(200, 220)
    bg_b = rng.randint(200, 220)
    img = Image.new("RGB", (size, size), (bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)

    # Number of cells based on class
    cell_counts = {
        "Normal": {"rbc": (20, 35), "wbc": (2, 5)},
        "ALL": {"rbc": (15, 25), "wbc": (8, 15)},
        "AML": {"rbc": (12, 22), "wbc": (10, 18)},
        "CML": {"rbc": (18, 28), "wbc": (6, 12)},
    }

    counts = cell_counts.get(class_name, cell_counts["Normal"])
    n_rbc = rng.randint(*counts["rbc"])
    n_wbc = rng.randint(*counts["wbc"])

    # Draw RBCs (red, small)
    for _ in range(n_rbc):
        x = rng.randint(10, size - 10)
        y = rng.randint(10, size - 10)
        r = rng.randint(5, 9)
        fill = (
            rng.randint(180, 220),
            rng.randint(50, 90),
            rng.randint(50, 80),
        )
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)
        # Light center (biconcave appearance)
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(
            min(fill[0] + 40, 255),
            min(fill[1] + 40, 255),
            min(fill[2] + 40, 255),
        ))

    # WBC nucleus colors by class
    wbc_colors = {
        "Normal": (100, 50, 150),
        "ALL": (80, 30, 160),
        "AML": (70, 40, 140),
        "CML": (90, 45, 155),
    }
    base_color = wbc_colors.get(class_name, (100, 50, 150))

    # Draw WBCs (larger, purple nuclei)
    for _ in range(n_wbc):
        x = rng.randint(15, size - 15)
        y = rng.randint(15, size - 15)

        # Blast cells are larger in leukemia
        if class_name != "Normal" and rng.random() > 0.3:
            r = rng.randint(10, 18)  # Blast cells
        else:
            r = rng.randint(7, 12)   # Normal WBCs

        # Cytoplasm
        cyto_r = r + rng.randint(2, 5)
        draw.ellipse(
            [x - cyto_r, y - cyto_r, x + cyto_r, y + cyto_r],
            fill=(180, 200, 220),
        )

        # Nucleus
        variation = tuple(
            max(0, min(255, c + rng.randint(-20, 20)))
            for c in base_color
        )
        draw.ellipse([x - r, y - r, x + r, y + r], fill=variation)

    # Add slight noise
    arr = np.array(img)
    noise = np.random.normal(0, 3, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def generate_dataset(
    output_dir: str,
    samples_per_class: int = 50,
):
    """Generate a full synthetic dataset."""
    classes = ["ALL", "AML", "CML", "Normal"]
    output = Path(output_dir)

    for class_name in classes:
        class_dir = output / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            img = generate_blood_smear(class_name=class_name)
            img.save(class_dir / f"{class_name.lower()}_{i:04d}.png")

        print(f"Generated {samples_per_class} images for {class_name}")

    print(f"\nDataset saved to: {output_dir}")
    print(f"Total images: {samples_per_class * len(classes)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic leukemia demo dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="demo_data",
        help="Output directory for synthetic images"
    )
    parser.add_argument(
        "--samples", type=int, default=50,
        help="Number of images per class"
    )
    args = parser.parse_args()

    generate_dataset(args.output_dir, args.samples)
