"""Unit tests for the evaluation module."""

import csv
from pathlib import Path
import networkx as nx
import pytest

from src.evaluation import (
    evaluate_provenance_graph,
    load_manifest,
    print_evaluation_report,
)


@pytest.fixture
def mock_manifest_file(tmp_path: Path) -> Path:
    """Create a temporary manifest CSV file."""
    manifest_path = tmp_path / "mock_manifest.csv"
    rows = [
        ["filename", "parent", "transformation"],
        ["orig1.jpg", "None", "original"],
        ["orig1_resized.jpg", "orig1.jpg", "resize"],
        ["orig1_cropped.jpg", "orig1.jpg", "crop"],
        ["orig2.jpg", "None", "original"],
        ["orig2_compressed.jpg", "orig2.jpg", "compression"],
        ["unrelated.jpg", "None", "unrelated"],
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return manifest_path


def test_load_manifest(mock_manifest_file: Path):
    manifest = load_manifest(mock_manifest_file)
    assert len(manifest) == 6
    assert manifest["orig1.jpg"]["parent"] is None
    assert manifest["orig1_resized.jpg"]["parent"] == "orig1.jpg"
    assert manifest["orig1_resized.jpg"]["transformation"] == "resize"
    assert manifest["orig1_cropped.jpg"]["root"] == "orig1.jpg"


def test_evaluate_provenance_graph(mock_manifest_file: Path):
    manifest = load_manifest(mock_manifest_file)

    # Build a simulated graph
    G = nx.DiGraph()
    for name in manifest.keys():
        G.add_node(name)

    # Add correct edges
    G.add_edge("orig1.jpg", "orig1_resized.jpg", similarity=0.9, direction="A_TO_B")
    G.add_edge("orig1.jpg", "orig1_cropped.jpg", similarity=0.85, direction="A_TO_B")
    G.add_edge("orig2.jpg", "orig2_compressed.jpg", similarity=0.95, direction="A_TO_B")

    metrics = evaluate_provenance_graph(G, manifest)

    assert metrics.pairwise_precision == 1.0
    assert metrics.pairwise_recall > 0.5
    assert metrics.pairwise_f1 > 0.6
    assert metrics.lineage_recall == 1.0
    assert metrics.tp_lineage == 3

    assert metrics.transformation_breakdown["resize"]["recall"] == 1.0
    assert metrics.transformation_breakdown["crop"]["recall"] == 1.0
    assert metrics.transformation_breakdown["compression"]["recall"] == 1.0

    report = print_evaluation_report(metrics)
    assert "EVALUATION REPORT" in report
    assert "100.0%" in report
