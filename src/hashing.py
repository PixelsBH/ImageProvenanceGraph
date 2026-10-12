"""Perceptual hashing module for Image Provenance Graph.

Provides perceptual hash calculation (pHash, dHash, aHash, wHash) and pairwise
similarity computation to identify visually similar images across transformations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import imagehash
import pandas as pd
from PIL import Image

from src.metadata import load_image, scan_directory


@dataclass
class ImageHashes:
    """Stores multiple perceptual hash values for an image."""

    filename: str
    filepath: str
    phash: str
    dhash: str
    ahash: str
    whash: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def compute_hashes(
    image_or_path: Union[str, Path, Image.Image],
    hash_size: int = 8,
    filename: Optional[str] = None,
) -> Optional[ImageHashes]:
    """Compute perceptual hashes for an image.

    Args:
        image_or_path: Path to image file or PIL Image object.
        hash_size: Size of the hash matrix (default 8 for 64-bit hash).
        filename: Optional filename override if passing a PIL Image.

    Returns:
        ImageHashes object or None if the image cannot be loaded.
    """
    filepath_str = ""
    img: Optional[Image.Image] = None

    if isinstance(image_or_path, (str, Path)):
        p = Path(image_or_path)
        filepath_str = str(p.resolve())
        filename = filename or p.name
        img = load_image(p, to_rgb=False)
        if img is None:
            return None
    elif isinstance(image_or_path, Image.Image):
        img = image_or_path
        filename = filename or "in_memory_image"
    else:
        return None

    try:
        # pHash: DCT based, robust to frequency-preserving modifications
        ph = imagehash.phash(img, hash_size=hash_size)
        # dHash: gradient based, robust to contrast changes
        dh = imagehash.dhash(img, hash_size=hash_size)
        # aHash: average based, baseline comparison
        ah = imagehash.average_hash(img, hash_size=hash_size)
        # wHash: DWT wavelet based, robust to noise and compression
        wh = imagehash.whash(img, hash_size=hash_size)

        return ImageHashes(
            filename=filename,
            filepath=filepath_str,
            phash=str(ph),
            dhash=str(dh),
            ahash=str(ah),
            whash=str(wh),
        )
    except Exception:
        return None


def hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Calculate the Hamming distance between two hex hash strings."""
    try:
        h1 = imagehash.hex_to_hash(hex_hash1)
        h2 = imagehash.hex_to_hash(hex_hash2)
        return int(h1 - h2)
    except Exception:
        # Fallback to bitwise XOR bitcount
        val1 = int(hex_hash1, 16)
        val2 = int(hex_hash2, 16)
        return bin(val1 ^ val2).count("1")


def hash_distance_to_similarity(dist: int, max_expected_dist: int = 32) -> float:
    """Convert Hamming distance (0 to 64) into a normalized similarity score in [0.0, 1.0].

    Unrelated natural images typically have Hamming distance ~32 (random bit distribution).
    Distance <= 10 indicates strong similarity.
    Distance 0 indicates identical hashes.
    """
    if dist <= 0:
        return 1.0
    if dist >= max_expected_dist:
        return 0.0
    return round(float(1.0 - (dist / max_expected_dist)), 4)


def compare_image_hashes(h1: ImageHashes, h2: ImageHashes) -> Dict[str, Union[int, float]]:
    """Compare two ImageHashes objects and compute distances and normalized similarities."""
    phash_dist = hamming_distance(h1.phash, h2.phash)
    dhash_dist = hamming_distance(h1.dhash, h2.dhash)
    ahash_dist = hamming_distance(h1.ahash, h2.ahash)
    whash_dist = hamming_distance(h1.whash, h2.whash)

    # Primary similarity metric is based primarily on pHash, with wHash & dHash support
    phash_sim = hash_distance_to_similarity(phash_dist)
    dhash_sim = hash_distance_to_similarity(dhash_dist)
    whash_sim = hash_distance_to_similarity(whash_dist)

    # Composite hash similarity: weighted average (50% pHash, 30% wHash, 20% dHash)
    composite_sim = round(0.5 * phash_sim + 0.3 * whash_sim + 0.2 * dhash_sim, 4)

    return {
        "phash_dist": phash_dist,
        "dhash_dist": dhash_dist,
        "ahash_dist": ahash_dist,
        "whash_dist": whash_dist,
        "phash_similarity": phash_sim,
        "composite_similarity": composite_sim,
    }


def compute_hashes_for_directory(
    directory: Union[str, Path],
    recursive: bool = False,
) -> Dict[str, ImageHashes]:
    """Scan a directory, extract metadata, and compute hashes for all valid images."""
    meta_list = scan_directory(directory, recursive=recursive)
    hashes: Dict[str, ImageHashes] = {}
    for meta in meta_list:
        if meta.is_valid:
            h = compute_hashes(meta.filepath, filename=meta.filename)
            if h:
                hashes[meta.filename] = h
    return hashes


def pairwise_hash_matrix(
    hashes_dict: Dict[str, ImageHashes],
) -> pd.DataFrame:
    """Generate a pairwise comparison table for all pairs in hashes_dict."""
    filenames = sorted(hashes_dict.keys())
    rows = []

    for name1, name2 in combinations(filenames, 2):
        h1 = hashes_dict[name1]
        h2 = hashes_dict[name2]
        comp = compare_image_hashes(h1, h2)
        rows.append({
            "Image 1": name1,
            "Image 2": name2,
            "pHash Dist": comp["phash_dist"],
            "wHash Dist": comp["whash_dist"],
            "dHash Dist": comp["dhash_dist"],
            "pHash Sim": comp["phash_similarity"],
            "Hash Similarity": comp["composite_similarity"],
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="Hash Similarity", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    """CLI tool to compute perceptual hashes and display pairwise similarities."""
    parser = argparse.ArgumentParser(
        description="Compute perceptual hashes and compare image similarities."
    )
    parser.add_argument("paths", nargs="+", help="Image directories or files to compare.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively scan.")
    parser.add_argument("-o", "--output-csv", help="Save pairwise comparisons to CSV.")
    parser.add_argument("--top", type=int, default=20, help="Display top N similar pairs.")

    args = parser.parse_args()

    collected_hashes: Dict[str, ImageHashes] = {}
    for p_str in args.paths:
        p = Path(p_str)
        if p.is_dir():
            dir_hashes = compute_hashes_for_directory(p, recursive=args.recursive)
            collected_hashes.update(dir_hashes)
        elif p.is_file():
            h = compute_hashes(p)
            if h:
                collected_hashes[h.filename] = h

    if not collected_hashes:
        print("No valid images found for hashing.")
        return

    print(f"Computed perceptual hashes for {len(collected_hashes)} image(s).")
    df = pairwise_hash_matrix(collected_hashes)

    if df.empty:
        print("Need at least 2 images for pairwise comparison.")
        return

    try:
        from tabulate import tabulate

        print("\nTop Similar Image Pairs by Perceptual Hash:")
        print(tabulate(df.head(args.top), headers="keys", tablefmt="rounded_outline", showindex=False))
    except ImportError:
        print(df.head(args.top).to_string(index=False))

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved pairwise hash comparison to: {out_path}")


if __name__ == "__main__":
    main()
