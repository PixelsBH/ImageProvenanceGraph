"""Unit tests for the provenance graph reconstruction module."""

from pathlib import Path
import networkx as nx
import pytest

from src.graph_builder import (
    build_provenance_graph,
    estimate_edge_direction,
    export_graph_to_dataframe,
    render_graph_plot,
)
from src.matching import PairwiseMatchResult
from src.metadata import ImageMetadata


@pytest.fixture
def sample_metadata():
    """Create metadata objects for parent, resized child, and cropped child."""
    meta_parent = ImageMetadata(
        filename="parent.jpg",
        filepath="/fake/parent.jpg",
        is_valid=True,
        width=1000,
        height=800,
        megapixels=0.8,
        file_size_bytes=500000,
        file_size_human="500.0 KB",
    )
    meta_resized = ImageMetadata(
        filename="child_resized.jpg",
        filepath="/fake/child_resized.jpg",
        is_valid=True,
        width=500,
        height=400,
        megapixels=0.2,
        file_size_bytes=150000,
        file_size_human="150.0 KB",
    )
    meta_compressed = ImageMetadata(
        filename="child_compressed.jpg",
        filepath="/fake/child_compressed.jpg",
        is_valid=True,
        width=1000,
        height=800,
        megapixels=0.8,
        file_size_bytes=80000,
        file_size_human="80.0 KB",
    )
    meta_identical = ImageMetadata(
        filename="variant_identical.jpg",
        filepath="/fake/variant_identical.jpg",
        is_valid=True,
        width=1000,
        height=800,
        megapixels=0.8,
        file_size_bytes=500000,
        file_size_human="500.0 KB",
    )
    return {
        "parent.jpg": meta_parent,
        "child_resized.jpg": meta_resized,
        "child_compressed.jpg": meta_compressed,
        "variant_identical.jpg": meta_identical,
    }


def test_estimate_direction_downscale(sample_metadata):
    meta_p = sample_metadata["parent.jpg"]
    meta_c = sample_metadata["child_resized.jpg"]

    match = PairwiseMatchResult(
        img1="parent.jpg",
        img2="child_resized.jpg",
        path1=meta_p.filepath,
        path2=meta_c.filepath,
        phash_dist=2,
        hash_similarity=0.95,
        num_kp1=100,
        num_kp2=80,
        num_raw_matches=50,
        num_good_matches=40,
        num_inliers=35,
        inlier_ratio=0.87,
        sift_similarity=0.9,
        combined_score=0.93,
    )

    direction, conf, reason = estimate_edge_direction(meta_p, meta_c, match)
    assert direction == "A_TO_B"
    assert "Downscaled" in reason
    assert conf >= 0.7


def test_estimate_direction_recompression(sample_metadata):
    meta_p = sample_metadata["parent.jpg"]
    meta_c = sample_metadata["child_compressed.jpg"]

    match = PairwiseMatchResult(
        img1="parent.jpg",
        img2="child_compressed.jpg",
        path1=meta_p.filepath,
        path2=meta_c.filepath,
        phash_dist=1,
        hash_similarity=0.98,
        num_kp1=100,
        num_kp2=95,
        num_raw_matches=60,
        num_good_matches=50,
        num_inliers=45,
        inlier_ratio=0.9,
        sift_similarity=0.9,
        combined_score=0.95,
    )

    direction, conf, reason = estimate_edge_direction(meta_p, meta_c, match)
    assert direction == "A_TO_B"
    assert "Recompression" in reason


def test_estimate_direction_unknown(sample_metadata):
    meta_p = sample_metadata["parent.jpg"]
    meta_i = sample_metadata["variant_identical.jpg"]

    match = PairwiseMatchResult(
        img1="parent.jpg",
        img2="variant_identical.jpg",
        path1=meta_p.filepath,
        path2=meta_i.filepath,
        phash_dist=0,
        hash_similarity=1.0,
        num_kp1=100,
        num_kp2=100,
        num_raw_matches=80,
        num_good_matches=75,
        num_inliers=70,
        inlier_ratio=0.93,
        sift_similarity=1.0,
        combined_score=1.0,
    )

    direction, conf, reason = estimate_edge_direction(meta_p, meta_i, match)
    assert direction == "UNKNOWN"
    assert "Identical dimensions" in reason


def test_build_provenance_graph(sample_metadata):
    meta_p = sample_metadata["parent.jpg"]
    meta_c = sample_metadata["child_resized.jpg"]

    match = PairwiseMatchResult(
        img1="parent.jpg",
        img2="child_resized.jpg",
        path1=meta_p.filepath,
        path2=meta_c.filepath,
        phash_dist=2,
        hash_similarity=0.95,
        num_kp1=100,
        num_kp2=80,
        num_raw_matches=50,
        num_good_matches=40,
        num_inliers=35,
        inlier_ratio=0.87,
        sift_similarity=0.9,
        combined_score=0.93,
    )

    G = build_provenance_graph([match], sample_metadata, similarity_threshold=0.5)

    assert G.number_of_nodes() == 4
    assert G.has_edge("parent.jpg", "child_resized.jpg")
    assert nx.is_directed_acyclic_graph(G)

    df = export_graph_to_dataframe(G)
    assert len(df) == 1
    assert df["Source (Parent)"].iloc[0] == "parent.jpg"
    assert df["Target (Child)"].iloc[0] == "child_resized.jpg"


def test_render_graph_plot(sample_metadata, tmp_path: Path):
    G = nx.DiGraph()
    G.add_node("A", width=800, height=600)
    G.add_node("B", width=400, height=300)
    G.add_edge("A", "B", similarity=0.9, direction="A_TO_B")

    plot_file = tmp_path / "test_graph.png"
    fig = render_graph_plot(G, output_path=plot_file)
    assert fig is not None
    assert plot_file.exists()
    assert plot_file.stat().st_size > 0
