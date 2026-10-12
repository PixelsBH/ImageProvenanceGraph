"""Tests for the image metadata extraction and validation module."""

from pathlib import Path

import pytest
from PIL import ExifTags, Image

from src.metadata import (
    ImageMetadata,
    display_metadata_table,
    extract_metadata,
    format_bytes,
    load_image,
    metadata_to_dataframe,
    scan_directory,
    summarize_exif,
)


@pytest.fixture
def sample_png(tmp_path: Path) -> Path:
    """Create a valid synthetic PNG image."""
    img_path = tmp_path / "test_image.png"
    img = Image.new("RGBA", (150, 100), color=(255, 0, 0, 255))
    img.save(img_path, format="PNG")
    return img_path


@pytest.fixture
def sample_jpeg_with_exif(tmp_path: Path) -> Path:
    """Create a valid synthetic JPEG image with mock EXIF tags."""
    img_path = tmp_path / "test_with_exif.jpg"
    img = Image.new("RGB", (200, 150), color=(0, 128, 255))

    # Build EXIF using native Pillow Exif
    exif = img.getexif()
    exif[0x010F] = "Canon"  # Make
    exif[0x0110] = "Canon EOS 80D"  # Model
    exif[0x0131] = "Test Suite 1.0"  # Software
    exif[0x0132] = "2026:05:12 10:30:00"  # DateTime
    img.save(img_path, format="JPEG", exif=exif)
    return img_path


@pytest.fixture
def non_image_file(tmp_path: Path) -> Path:
    """Create a text file masquerading with an image extension or text extension."""
    txt_path = tmp_path / "not_an_image.txt"
    txt_path.write_text("Hello, this is just plain text content.")
    return txt_path


@pytest.fixture
def zero_byte_file(tmp_path: Path) -> Path:
    """Create an empty zero-byte file."""
    empty_path = tmp_path / "empty_file.jpg"
    empty_path.touch()
    return empty_path


@pytest.fixture
def corrupt_image_file(tmp_path: Path) -> Path:
    """Create a file with corrupted image bytes."""
    corrupt_path = tmp_path / "corrupted.png"
    corrupt_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRcorruptedcontentgarbage")
    return corrupt_path


def test_format_bytes():
    assert format_bytes(0) == "0 B"
    assert format_bytes(512) == "512 B"
    assert format_bytes(1024) == "1.0 KB"
    assert format_bytes(1536) == "1.5 KB"
    assert format_bytes(1048576) == "1.0 MB"


def test_valid_png_metadata(sample_png: Path):
    meta = extract_metadata(sample_png)

    assert meta.is_valid is True
    assert meta.filename == "test_image.png"
    assert meta.width == 150
    assert meta.height == 100
    assert meta.aspect_ratio == 1.5
    assert meta.format == "PNG"
    assert meta.mode == "RGBA"
    assert meta.error_message is None
    assert meta.file_size_bytes > 0
    assert meta.sha256 is not None
    assert len(meta.sha256) == 64


def test_valid_jpeg_with_exif_metadata(sample_jpeg_with_exif: Path):
    meta = extract_metadata(sample_jpeg_with_exif)

    assert meta.is_valid is True
    assert meta.filename == "test_with_exif.jpg"
    assert meta.width == 200
    assert meta.height == 150
    assert meta.format == "JPEG"
    assert meta.mode == "RGB"
    assert "Canon" in meta.exif_summary
    assert "EOS 80D" in meta.exif_summary
    assert "2026:05:12" in meta.exif_summary


def test_nonexistent_file(tmp_path: Path):
    ghost_path = tmp_path / "does_not_exist.png"
    meta = extract_metadata(ghost_path)

    assert meta.is_valid is False
    assert "does not exist" in meta.error_message.lower()


def test_non_image_file(non_image_file: Path):
    meta = extract_metadata(non_image_file)

    assert meta.is_valid is False
    assert meta.error_message is not None


def test_zero_byte_file(zero_byte_file: Path):
    meta = extract_metadata(zero_byte_file)

    assert meta.is_valid is False
    assert "0 bytes" in meta.error_message.lower()


def test_corrupt_image_file(corrupt_image_file: Path):
    meta = extract_metadata(corrupt_image_file)

    assert meta.is_valid is False
    assert meta.error_message is not None


def test_load_image(sample_png: Path, non_image_file: Path):
    img = load_image(sample_png, to_rgb=True)
    assert img is not None
    assert img.mode == "RGB"
    assert img.size == (150, 100)

    # Invalid file returns None
    invalid = load_image(non_image_file)
    assert invalid is None


def test_scan_directory(
    tmp_path: Path,
    sample_png: Path,
    sample_jpeg_with_exif: Path,
    non_image_file: Path,
):
    results = scan_directory(tmp_path)
    # Should find all created files
    assert len(results) >= 3

    valid_results = [r for r in results if r.is_valid]
    invalid_results = [r for r in results if not r.is_valid]

    assert len(valid_results) == 2
    assert len(invalid_results) >= 1


def test_metadata_to_dataframe_and_table(sample_png: Path, non_image_file: Path):
    records = [extract_metadata(sample_png), extract_metadata(non_image_file)]
    df = metadata_to_dataframe(records)

    assert len(df) == 2
    assert "Filename" in df.columns
    assert "Dimensions" in df.columns
    assert "Format" in df.columns
    assert "Status / Error" in df.columns

    table_str = display_metadata_table(df)
    assert "test_image.png" in table_str
    assert "not_an_image.txt" in table_str
