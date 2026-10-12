"""Script to generate a synthetic sample dataset for testing the Image Provenance Graph.

Creates:
1. 3 synthetic original images with distinctive features and mock EXIF data.
2. 4 transformation variants per original (resized, cropped, compressed, brightened).
3. 1 unrelated distractor image.
4. A corrupted test file to verify invalid-file handling.
5. data/manifest.csv documenting ground-truth parent-child relationships.
"""

from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

DATA_DIR = Path("data")
ORIGINALS_DIR = DATA_DIR / "originals"
VARIANTS_DIR = DATA_DIR / "variants"
MANIFEST_FILE = DATA_DIR / "manifest.csv"


def create_synthetic_original(
    filename: str,
    pattern_type: str,
    camera_make: str = "Sony",
    camera_model: str = "ILCE-7M4",
    capture_time: str = "2026:04:10 09:15:00",
) -> Path:
    """Generate a rich, multi-feature synthetic image suitable for SIFT and hashing."""
    width, height = 800, 600
    img = Image.new("RGB", (width, height), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)

    if pattern_type == "shapes":
        # Draw colorful nested geometric shapes
        for r in range(20, 280, 25):
            draw.ellipse(
                [(width // 2 - r, height // 2 - r), (width // 2 + r, height // 2 + r)],
                outline=(r % 255, (r * 2) % 255, (255 - r) % 255),
                width=4,
            )
        draw.rectangle([(50, 50), (250, 250)], fill=(40, 120, 220), outline=(0, 0, 0), width=3)
        draw.polygon([(600, 100), (750, 250), (520, 300)], fill=(220, 80, 50))
        draw.line([(50, 550), (750, 550)], fill=(30, 30, 30), width=6)

    elif pattern_type == "grid":
        # Checkered texture and diagonal bands
        step = 40
        for x in range(0, width, step):
            for y in range(0, height, step):
                if (x // step + y // step) % 2 == 0:
                    draw.rectangle([(x, y), (x + step, y + step)], fill=(60, 160, 100))
                else:
                    draw.rectangle([(x, y), (x + step, y + step)], fill=(230, 230, 230))
        draw.ellipse([(200, 150), (600, 450)], fill=(255, 140, 0), outline=(20, 20, 20), width=5)

    elif pattern_type == "landscape":
        # Simulated landscape: sky, mountains, sun
        draw.rectangle([(0, 0), (width, 350)], fill=(135, 206, 235))  # sky
        draw.ellipse([(620, 60), (720, 160)], fill=(255, 215, 0))  # sun
        # Mountains
        draw.polygon([(0, 350), (220, 160), (450, 350)], fill=(105, 105, 105))
        draw.polygon([(260, 350), (520, 130), (780, 350)], fill=(130, 130, 130))
        # Ground
        draw.rectangle([(0, 350), (width, height)], fill=(34, 139, 34))

    # Add EXIF metadata
    exif = img.getexif()
    exif[0x010F] = camera_make
    exif[0x0110] = camera_model
    exif[0x0131] = "ImageProvenance Synthetic 1.0"
    exif[0x0132] = capture_time

    out_path = ORIGINALS_DIR / filename
    img.save(out_path, format="JPEG", quality=95, exif=exif)
    return out_path


def generate_variants(original_path: Path) -> list[tuple[str, str, str]]:
    """Generate variants for a given original image and return manifest entries."""
    base_name = original_path.stem
    manifest_entries = []

    with Image.open(original_path) as img:
        # 1. Resized variant (50% scale)
        resized_name = f"{base_name}_resized.jpg"
        resized_path = VARIANTS_DIR / resized_name
        w, h = img.size
        resized_img = img.resize((w // 2, h // 2), Image.Resampling.LANCZOS)
        resized_img.save(resized_path, quality=90)
        manifest_entries.append((resized_name, base_name + ".jpg", "resize_50%"))

        # 2. Cropped variant (central 60% crop)
        cropped_name = f"{base_name}_cropped.jpg"
        cropped_path = VARIANTS_DIR / cropped_name
        left, top = int(w * 0.2), int(h * 0.2)
        right, bottom = int(w * 0.8), int(h * 0.8)
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(cropped_path, quality=90)
        manifest_entries.append((cropped_name, base_name + ".jpg", "crop_center_60%"))

        # 3. Heavy JPEG compression variant
        compressed_name = f"{base_name}_compressed.jpg"
        compressed_path = VARIANTS_DIR / compressed_name
        img.save(compressed_path, format="JPEG", quality=20)
        manifest_entries.append((compressed_name, base_name + ".jpg", "jpeg_compression_q20"))

        # 4. Brightness adjusted variant
        bright_name = f"{base_name}_bright.jpg"
        bright_path = VARIANTS_DIR / bright_name
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(1.35)
        bright_img.save(bright_path, quality=90)
        manifest_entries.append((bright_name, base_name + ".jpg", "brightness_plus_35%"))

    return manifest_entries


def main():
    ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = [("filename", "parent", "transformation")]

    # 1. Generate 3 originals
    originals = [
        ("orig_shapes.jpg", "shapes", "Nikon", "Z7 II", "2026:03:01 14:10:00"),
        ("orig_grid.jpg", "grid", "Sony", "A7 IV", "2026:03:02 11:25:30"),
        ("orig_landscape.jpg", "landscape", "Fujifilm", "X-T5", "2026:03:03 16:45:12"),
    ]

    for fname, ptype, make, model, dt in originals:
        orig_path = create_synthetic_original(fname, ptype, make, model, dt)
        manifest_rows.append((fname, "None", "original"))
        # Generate variants
        variant_entries = generate_variants(orig_path)
        manifest_rows.extend(variant_entries)

    # 2. Add an unrelated distractor image
    unrelated_path = ORIGINALS_DIR / "unrelated_noise.jpg"
    noise_arr = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    noise_img = Image.fromarray(noise_arr)
    noise_img.save(unrelated_path, format="JPEG", quality=90)
    manifest_rows.append(("unrelated_noise.jpg", "None", "unrelated_original"))

    # 3. Add a corrupted file in variants to test invalid handling
    corrupt_path = VARIANTS_DIR / "corrupted_file.jpg"
    corrupt_path.write_bytes(b"\xFF\xD8\xFF\xE0TruncatedCorruptDataNotAnImage")
    manifest_rows.append(("corrupted_file.jpg", "None", "corrupted_test_file"))

    # Write manifest.csv
    with open(MANIFEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(manifest_rows)

    print(f"Sample dataset generated successfully:")
    print(f"  - Originals: {ORIGINALS_DIR} ({len(list(ORIGINALS_DIR.glob('*')))} files)")
    print(f"  - Variants: {VARIANTS_DIR} ({len(list(VARIANTS_DIR.glob('*')))} files)")
    print(f"  - Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
