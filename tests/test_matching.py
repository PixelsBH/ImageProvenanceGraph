"""Unit tests for the SIFT matching and similarity scoring module."""

from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import pytest

from src.matching import (
    FeatureExtractor,
    calculate_combined_similarity,
    compare_image_collection,
    compare_image_pair,
    match_sift_features,
)


@pytest.fixture
def patterned_image(tmp_path: Path) -> Path:
    """Create a high-contrast patterned image with rich SIFT features."""
    img_path = tmp_path / "pattern.png"
    im = Image.new("RGB", (300, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(im)

    # Draw geometric features for SIFT
    for r in range(10, 140, 15):
        draw.ellipse([(150 - r, 150 - r), (150 + r, 150 + r)], outline=(0, 0, 0), width=3)
    draw.rectangle([(30, 30), (100, 100)], fill=(0, 100, 255))
    draw.polygon([(200, 30), (280, 100), (220, 150)], fill=(255, 50, 0))

    im.save(img_path)
    return img_path


@pytest.fixture
def scaled_image(patterned_image: Path, tmp_path: Path) -> Path:
    """Create a 50% resized variant of the patterned image."""
    out_path = tmp_path / "pattern_scaled.png"
    with Image.open(patterned_image) as im:
        resized = im.resize((150, 150), Image.Resampling.LANCZOS)
        resized.save(out_path)
    return out_path


@pytest.fixture
def cropped_image(patterned_image: Path, tmp_path: Path) -> Path:
    """Create a central crop of the patterned image."""
    out_path = tmp_path / "pattern_cropped.png"
    with Image.open(patterned_image) as im:
        crop = im.crop((50, 50, 250, 250))
        crop.save(out_path)
    return out_path


def test_feature_extractor_caching(patterned_image: Path):
    extractor = FeatureExtractor()
    feat1 = extractor.extract_from_path(patterned_image)
    assert feat1 is not None
    assert feat1.num_keypoints > 20
    assert feat1.has_descriptors

    # Second extraction should return cached object
    feat2 = extractor.extract_from_path(patterned_image)
    assert feat1 is feat2


def test_sift_matching_scaled(patterned_image: Path, scaled_image: Path):
    extractor = FeatureExtractor()
    feat1 = extractor.extract_from_path(patterned_image)
    feat2 = extractor.extract_from_path(scaled_image)

    match = match_sift_features(feat1, feat2)
    assert match["num_inliers"] >= 10
    assert match["sift_similarity"] > 0.40


def test_sift_matching_cropped(patterned_image: Path, cropped_image: Path):
    extractor = FeatureExtractor()
    feat1 = extractor.extract_from_path(patterned_image)
    feat2 = extractor.extract_from_path(cropped_image)

    match = match_sift_features(feat1, feat2)
    assert match["num_inliers"] >= 10
    assert match["sift_similarity"] > 0.40


def test_combined_similarity_crop_boost():
    # In a crop case: pHash similarity might be weak (e.g. 0.15) but SIFT inliers high
    score = calculate_combined_similarity(
        hash_sim=0.15,
        sift_sim=0.85,
        num_inliers=25,
        phash_dist=28,
    )
    # The combination should boost to recognize the crop relationship
    assert score >= 0.70


def test_compare_image_collection(patterned_image: Path, scaled_image: Path, cropped_image: Path):
    paths = [patterned_image, scaled_image, cropped_image]
    results, df = compare_image_collection(paths, threshold=0.3)

    assert len(results) >= 2
    assert "Combined Score" in df.columns
    assert df["Combined Score"].iloc[0] > 0.5
