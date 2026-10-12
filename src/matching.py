"""SIFT feature extraction, matching, and combined similarity scoring module.

Provides:
1. SIFT feature detection and descriptor extraction with local caching.
2. Robust pairwise feature matching using Lowe's ratio test and RANSAC geometric verification.
3. Combination of perceptual hash similarity and SIFT geometric evidence into a unified score.
4. Identification of transformation clues (scale changes, cropping, area ratios).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.hashing import ImageHashes, compare_image_hashes, compute_hashes
from src.metadata import ImageMetadata, extract_metadata, load_image, scan_directory


@dataclass
class SIFTFeatures:
    """Cached SIFT keypoints and descriptors for an image."""

    filename: str
    filepath: str
    image_shape: Tuple[int, int]  # (height, width)
    num_keypoints: int
    keypoints: Sequence[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]

    @property
    def has_descriptors(self) -> bool:
        return self.descriptors is not None and len(self.descriptors) > 0


@dataclass
class PairwiseMatchResult:
    """Comprehensive comparison results between two images."""

    img1: str
    img2: str
    path1: str
    path2: str
    # Perceptual hash metrics
    phash_dist: int
    hash_similarity: float
    # SIFT metrics
    num_kp1: int
    num_kp2: int
    num_raw_matches: int
    num_good_matches: int
    num_inliers: int
    inlier_ratio: float
    sift_similarity: float
    # Combined score
    combined_score: float
    # Transformation evidence clues
    clues: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("clues", None)
        return d


class FeatureExtractor:
    """Extracts and caches SIFT features for images."""

    def __init__(self, n_features: int = 2000):
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self._cache: Dict[str, SIFTFeatures] = {}

    def extract_from_path(self, filepath: Union[str, Path]) -> Optional[SIFTFeatures]:
        path_str = str(Path(filepath).resolve())
        if path_str in self._cache:
            return self._cache[path_str]

        pil_img = load_image(filepath, to_rgb=True)
        if pil_img is None:
            return None

        # Convert PIL to Grayscale uint8 for SIFT
        img_np = np.array(pil_img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        kp, des = self.sift.detectAndCompute(gray, None)
        kp = kp or []

        features = SIFTFeatures(
            filename=Path(filepath).name,
            filepath=path_str,
            image_shape=(h, w),
            num_keypoints=len(kp),
            keypoints=kp,
            descriptors=des,
        )
        self._cache[path_str] = features
        return features

    def clear_cache(self) -> None:
        self._cache.clear()


def match_sift_features(
    feat1: SIFTFeatures,
    feat2: SIFTFeatures,
    ratio_threshold: float = 0.75,
    ransac_threshold: float = 5.0,
) -> Dict[str, Any]:
    """Perform Lowe's ratio test and RANSAC homography estimation between two feature sets.

    Returns:
        Dictionary containing raw matches, good matches, inliers, inlier ratio,
        estimated homography, and scale/transformation clues.
    """
    if not feat1.has_descriptors or not feat2.has_descriptors:
        return {
            "num_raw_matches": 0,
            "num_good_matches": 0,
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "sift_similarity": 0.0,
            "homography": None,
            "clues": {},
        }

    # Descriptors must have at least 2 points to run KNN match
    if len(feat1.descriptors) < 2 or len(feat2.descriptors) < 2:
        return {
            "num_raw_matches": 0,
            "num_good_matches": 0,
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "sift_similarity": 0.0,
            "homography": None,
            "clues": {},
        }

    # Match descriptors using FLANN or BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(feat1.descriptors, feat2.descriptors, k=2)

    good_matches = []
    for match in raw_matches:
        if len(match) == 2:
            m, n = match
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    num_good = len(good_matches)
    min_kp = min(feat1.num_keypoints, feat2.num_keypoints)

    num_inliers = 0
    inlier_ratio = 0.0
    homography = None
    clues: Dict[str, Any] = {}

    # Geometric verification with RANSAC
    if num_good >= 4:
        src_pts = np.float32([feat1.keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([feat2.keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        if mask is not None:
            num_inliers = int(np.sum(mask))
            inlier_ratio = round(num_inliers / max(1, num_good), 4)
            homography = H

            # Analyze geometric transformation clues from Homography
            if H is not None and not np.isnan(H).any() and not np.isinf(H).any():
                det = np.linalg.det(H[:2, :2])
                scale_est = round(float(np.sqrt(abs(det))), 3)
                clues["estimated_scale_h1_to_h2"] = scale_est

    # Compute SIFT similarity score
    # High inliers (e.g. >= 20) on natural images strongly indicates shared visual lineage
    inlier_saturation = 25.0
    inlier_score = min(1.0, num_inliers / inlier_saturation)
    coverage_score = min(1.0, (num_inliers / max(1, min_kp)) * 3.0)

    if num_inliers < 4:
        sift_sim = 0.0
    else:
        sift_sim = round(0.65 * inlier_score + 0.35 * coverage_score, 4)

    return {
        "num_raw_matches": len(raw_matches),
        "num_good_matches": num_good,
        "num_inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
        "sift_similarity": sift_sim,
        "homography": homography,
        "clues": clues,
    }


def calculate_combined_similarity(
    hash_sim: float,
    sift_sim: float,
    num_inliers: int,
    phash_dist: int,
) -> float:
    """Combine perceptual hash similarity and SIFT feature matching score.

    Design rationale:
    - SIFT is invariant to scale and rotation, and handles cropping exceptionally well.
    - Perceptual hash is fast and handles recompression, blur, and color adjustment well,
      but degrades under severe cropping.
    - If SIFT finds strong geometric inliers (>= 15), we trust SIFT even if hash is lower.
    - If pHash is very close (distance <= 4) and SIFT agrees with inliers (>= 6), score is high.
    - If both are weak, score is close to 0.
    """
    # Baseline weighted combination
    base_score = 0.45 * hash_sim + 0.55 * sift_sim

    # Crop / Partial-overlap boost: when SIFT has confident geometric inliers
    if num_inliers >= 15 and sift_sim >= 0.6:
        # Boost combined score for cropped or transformed variants
        boosted = 0.85 * sift_sim + 0.15 * hash_sim
        base_score = max(base_score, boosted)

    # Identical / minor edit boost: near-zero pHash distance and geometric confirmation
    if phash_dist <= 2 and num_inliers >= 6:
        base_score = max(base_score, 0.90 + 0.10 * sift_sim)

    # If SIFT inliers are negligible (< 4) and hash similarity is marginal (< 0.65), suppress false positives
    if num_inliers < 4 and hash_sim < 0.65:
        base_score = min(base_score, 0.20)

    return round(float(np.clip(base_score, 0.0, 1.0)), 4)


def compare_image_pair(
    path1: Union[str, Path],
    path2: Union[str, Path],
    extractor: FeatureExtractor,
    meta1: Optional[ImageMetadata] = None,
    meta2: Optional[ImageMetadata] = None,
    hashes1: Optional[ImageHashes] = None,
    hashes2: Optional[ImageHashes] = None,
) -> Optional[PairwiseMatchResult]:
    """Perform full pairwise forensic comparison between two images."""
    p1 = Path(path1)
    p2 = Path(path2)

    # 1. Hashes
    h1 = hashes1 or compute_hashes(p1)
    h2 = hashes2 or compute_hashes(p2)
    if h1 is None or h2 is None:
        return None

    hash_comp = compare_image_hashes(h1, h2)
    phash_dist = int(hash_comp["phash_dist"])
    hash_sim = float(hash_comp["composite_similarity"])

    # 2. SIFT Features
    feat1 = extractor.extract_from_path(p1)
    feat2 = extractor.extract_from_path(p2)
    if feat1 is None or feat2 is None:
        return None

    sift_match = match_sift_features(feat1, feat2)
    num_raw = sift_match["num_raw_matches"]
    num_good = sift_match["num_good_matches"]
    num_inliers = sift_match["num_inliers"]
    inlier_ratio = sift_match["inlier_ratio"]
    sift_sim = sift_match["sift_similarity"]

    # 3. Combined Score
    combined = calculate_combined_similarity(
        hash_sim=hash_sim,
        sift_sim=sift_sim,
        num_inliers=num_inliers,
        phash_dist=phash_dist,
    )

    # 4. Metadata-based transformation clues
    clues = sift_match.get("clues", {}).copy()
    m1 = meta1 or extract_metadata(p1)
    m2 = meta2 or extract_metadata(p2)

    if m1.is_valid and m2.is_valid:
        area1 = (m1.width or 1) * (m1.height or 1)
        area2 = (m2.width or 1) * (m2.height or 1)
        clues["area_ratio"] = round(area1 / max(1, area2), 3)
        clues["size_ratio"] = round(m1.file_size_bytes / max(1, m2.file_size_bytes), 3)

    return PairwiseMatchResult(
        img1=p1.name,
        img2=p2.name,
        path1=str(p1.resolve()),
        path2=str(p2.resolve()),
        phash_dist=phash_dist,
        hash_similarity=hash_sim,
        num_kp1=feat1.num_keypoints,
        num_kp2=feat2.num_keypoints,
        num_raw_matches=num_raw,
        num_good_matches=num_good,
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        sift_similarity=sift_sim,
        combined_score=combined,
        clues=clues,
    )


def compare_image_collection(
    file_paths: Sequence[Union[str, Path]],
    threshold: float = 0.0,
) -> Tuple[List[PairwiseMatchResult], pd.DataFrame]:
    """Compare all unique pairs in a collection of images.

    Returns:
        List of PairwiseMatchResult objects and a sorted pandas DataFrame.
    """
    valid_paths = [Path(p) for p in file_paths if Path(p).is_file()]
    valid_paths.sort(key=lambda p: p.name.lower())

    extractor = FeatureExtractor()
    meta_cache: Dict[str, ImageMetadata] = {}
    hash_cache: Dict[str, ImageHashes] = {}

    # Pre-extract metadata and hashes
    for p in valid_paths:
        m = extract_metadata(p)
        if m.is_valid:
            meta_cache[str(p.resolve())] = m
            h = compute_hashes(p, filename=m.filename)
            if h:
                hash_cache[str(p.resolve())] = h

    filtered_paths = [p for p in valid_paths if str(p.resolve()) in hash_cache]
    results: List[PairwiseMatchResult] = []

    for p1, p2 in combinations(filtered_paths, 2):
        key1 = str(p1.resolve())
        key2 = str(p2.resolve())
        res = compare_image_pair(
            path1=p1,
            path2=p2,
            extractor=extractor,
            meta1=meta_cache.get(key1),
            meta2=meta_cache.get(key2),
            hashes1=hash_cache.get(key1),
            hashes2=hash_cache.get(key2),
        )
        if res and res.combined_score >= threshold:
            results.append(res)

    # Sort descending by combined score
    results.sort(key=lambda r: r.combined_score, reverse=True)

    rows = []
    for r in results:
        rows.append({
            "Image 1": r.img1,
            "Image 2": r.img2,
            "Combined Score": r.combined_score,
            "Hash Sim": r.hash_similarity,
            "SIFT Sim": r.sift_similarity,
            "Inliers": r.num_inliers,
            "Good Matches": r.num_good_matches,
            "pHash Dist": r.phash_dist,
        })

    df = pd.DataFrame(rows)
    return results, df


def main() -> None:
    """CLI tool to run pairwise image comparison and display similarity table."""
    parser = argparse.ArgumentParser(
        description="Compare image features using Perceptual Hash + SIFT."
    )
    parser.add_argument("paths", nargs="+", help="Directories or files to compare.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Scan recursively.")
    parser.add_argument("-t", "--threshold", type=float, default=0.35, help="Score threshold.")
    parser.add_argument("-o", "--output-csv", help="Save pairwise matches to CSV.")
    parser.add_argument("--top", type=int, default=25, help="Display top N pairs.")

    args = parser.parse_args()

    file_list: List[Path] = []
    for p_str in args.paths:
        p = Path(p_str)
        if p.is_dir():
            meta_list = scan_directory(p, recursive=args.recursive)
            file_list.extend([Path(m.filepath) for m in meta_list if m.is_valid])
        elif p.is_file():
            file_list.append(p)

    if len(file_list) < 2:
        print("Need at least 2 valid image files to compare.")
        return

    print(f"Comparing {len(file_list)} images across all pairs...")
    results, df = compare_image_collection(file_list, threshold=args.threshold)

    print(f"Found {len(results)} pair(s) with combined score >= {args.threshold}.\n")

    try:
        from tabulate import tabulate

        print(tabulate(df.head(args.top), headers="keys", tablefmt="rounded_outline", showindex=False))
    except ImportError:
        print(df.head(args.top).to_string(index=False))

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Full export with paths
        export_df = pd.DataFrame([r.to_dict() for r in results])
        export_df.to_csv(out_path, index=False)
        print(f"\nSaved pairwise matches to: {out_path}")


if __name__ == "__main__":
    main()
