"""Unit tests for the perceptual hashing module."""

from pathlib import Path
from PIL import Image
import pytest

from src.hashing import (
    ImageHashes,
    compare_image_hashes,
    compute_hashes,
    hamming_distance,
    hash_distance_to_similarity,
    pairwise_hash_matrix,
)


@pytest.fixture
def image_pair(tmp_path: Path):
    """Create two test images: one original and one slightly edited variant."""
    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"

    # Image 1
    im1 = Image.new("RGB", (120, 120), color=(100, 150, 200))
    im1.save(img1_path)

    # Image 2 (identical content, saved with slight color tweak)
    im2 = Image.new("RGB", (120, 120), color=(102, 148, 200))
    im2.save(img2_path)

    return img1_path, img2_path


@pytest.fixture
def unrelated_image(tmp_path: Path):
    """Create an unrelated checkerboard pattern image."""
    img_path = tmp_path / "unrelated.png"
    im = Image.new("L", (120, 120))
    pixels = im.load()
    for x in range(120):
        for y in range(120):
            pixels[x, y] = 255 if (x // 10 + y // 10) % 2 == 0 else 0
    im.save(img_path)
    return img_path


def test_compute_hashes_valid(image_pair):
    img1, _ = image_pair
    hashes = compute_hashes(img1)
    assert hashes is not None
    assert isinstance(hashes, ImageHashes)
    assert len(hashes.phash) > 0
    assert len(hashes.dhash) > 0
    assert len(hashes.whash) > 0


def test_compute_hashes_invalid(tmp_path: Path):
    corrupt = tmp_path / "bad.png"
    corrupt.write_text("not an image")
    assert compute_hashes(corrupt) is None


def test_hamming_distance_identical():
    h = "0123456789abcdef"
    assert hamming_distance(h, h) == 0


def test_hash_distance_to_similarity():
    assert hash_distance_to_similarity(0) == 1.0
    assert hash_distance_to_similarity(16, max_expected_dist=32) == 0.5
    assert hash_distance_to_similarity(35, max_expected_dist=32) == 0.0


def test_compare_similar_images(image_pair):
    img1, img2 = image_pair
    h1 = compute_hashes(img1)
    h2 = compute_hashes(img2)
    comp = compare_image_hashes(h1, h2)

    assert comp["phash_dist"] <= 2
    assert comp["phash_similarity"] >= 0.90
    assert comp["composite_similarity"] >= 0.90


def test_compare_unrelated_images(image_pair, unrelated_image):
    img1, _ = image_pair
    h1 = compute_hashes(img1)
    h_unrelated = compute_hashes(unrelated_image)
    comp = compare_image_hashes(h1, h_unrelated)

    assert comp["phash_dist"] > 15
    assert comp["phash_similarity"] < 0.60


def test_pairwise_hash_matrix(image_pair, unrelated_image):
    img1, img2 = image_pair
    h_dict = {
        "img1": compute_hashes(img1),
        "img2": compute_hashes(img2),
        "unrelated": compute_hashes(unrelated_image),
    }
    df = pairwise_hash_matrix(h_dict)
    assert len(df) == 3
    assert "Hash Similarity" in df.columns
    # Top pair should be img1 and img2
    top = df.iloc[0]
    assert set([top["Image 1"], top["Image 2"]]) == {"img1", "img2"}
