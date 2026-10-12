"""Evaluation module for Image Provenance Graph reconstruction.

Compares reconstructed graphs and pairwise similarities against ground-truth
lineage recorded in manifest.csv. Calculates:
1. Related-image pair detection: Precision, Recall, F1-score.
2. Directed parent-child lineage reconstruction: Precision, Recall, F1-score.
3. Robustness breakdown by transformation category (resizing, cropping, compression, brightness).
4. Qualitative success and failure case recording.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import pandas as pd

from src.graph_builder import build_provenance_graph
from src.matching import PairwiseMatchResult, compare_image_collection
from src.metadata import ImageMetadata, extract_metadata, scan_directory


@dataclass
class EvaluationMetrics:
    """Summary of provenance evaluation metrics."""

    # Pairwise detection metrics
    tp_pairs: int
    fp_pairs: int
    fn_pairs: int
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float

    # Directed lineage metrics
    tp_lineage: int
    fp_lineage: int
    fn_lineage: int
    lineage_precision: float
    lineage_recall: float
    lineage_f1: float

    # Transformation breakdown: category -> {total, detected, recall}
    transformation_breakdown: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=dict)

    # Success and failure case logs
    successful_cases: List[Dict[str, Any]] = field(default_factory=list)
    failure_cases: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_manifest(manifest_path: Union[str, Path]) -> Dict[str, Dict[str, str]]:
    """Load ground-truth manifest CSV.

    Expected columns: filename, parent, transformation
    Returns mapping: filename -> {'parent': parent, 'transformation': trans, 'root': root}
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"].strip()
            parent = row["parent"].strip()
            if parent in ("", "None", "null", "none"):
                parent = None
            trans = row.get("transformation", "unknown").strip()
            records[fname] = {
                "parent": parent,
                "transformation": trans,
                "root": parent if parent else fname,
            }

    # Propagate roots for multi-hop lineage
    for fname, data in records.items():
        curr = data["parent"]
        visited = {fname}
        while curr and curr in records and records[curr]["parent"] and curr not in visited:
            visited.add(curr)
            curr = records[curr]["parent"]
        if curr:
            data["root"] = curr

    return records


def evaluate_provenance_graph(
    G: nx.DiGraph,
    manifest: Dict[str, Dict[str, str]],
    pairwise_results: Optional[Sequence[PairwiseMatchResult]] = None,
) -> EvaluationMetrics:
    """Evaluate reconstructed graph against manifest ground truth."""
    # 1. Ground truth definitions
    gt_parent_child_edges: Set[Tuple[str, str]] = set()
    gt_related_pairs: Set[Tuple[str, str]] = set()
    gt_variants_by_transform: Dict[str, List[str]] = {}

    known_files = set(manifest.keys())

    for fname, data in manifest.items():
        parent = data["parent"]
        root = data["root"]
        trans = data["transformation"]

        if parent:
            gt_parent_child_edges.add((parent, fname))
            gt_variants_by_transform.setdefault(trans, []).append(fname)

        # Related pairs: any two files sharing the same non-None root
        for other_fname, other_data in manifest.items():
            if fname < other_fname:
                if root and other_data["root"] and root == other_data["root"]:
                    # Same lineage family
                    gt_related_pairs.add((fname, other_fname))

    # 2. Predicted definitions from graph
    pred_edges: Set[Tuple[str, str]] = set()
    pred_undirected_pairs: Set[Tuple[str, str]] = set()

    for u, v, data in G.edges(data=True):
        if u in known_files and v in known_files:
            pred_edges.add((u, v))
            pair = tuple(sorted([u, v]))
            pred_undirected_pairs.add(pair)

    # 3. Evaluate Pairwise Connection (Undirected)
    tp_pairs = len(pred_undirected_pairs & gt_related_pairs)
    fp_pairs = len(pred_undirected_pairs - gt_related_pairs)
    fn_pairs = len(gt_related_pairs - pred_undirected_pairs)

    p_prec = round(tp_pairs / max(1, tp_pairs + fp_pairs), 4)
    p_rec = round(tp_pairs / max(1, tp_pairs + fn_pairs), 4)
    p_f1 = round((2 * p_prec * p_rec) / max(1e-6, p_prec + p_rec), 4) if (p_prec + p_rec) > 0 else 0.0

    # 4. Evaluate Directed Lineage Edges (u -> v where u is parent of v)
    tp_dir = len(pred_edges & gt_parent_child_edges)
    fp_dir = len(pred_edges - gt_parent_child_edges)
    fn_dir = len(gt_parent_child_edges - pred_edges)

    d_prec = round(tp_dir / max(1, tp_dir + fp_dir), 4)
    d_rec = round(tp_dir / max(1, tp_dir + fn_dir), 4)
    d_f1 = round((2 * d_prec * d_rec) / max(1e-6, d_prec + d_rec), 4) if (d_prec + d_rec) > 0 else 0.0

    # 5. Transformation breakdown
    breakdown: Dict[str, Dict[str, Union[int, float]]] = {}
    for trans, variants in gt_variants_by_transform.items():
        total = len(variants)
        detected = 0
        for v in variants:
            parent = manifest[v]["parent"]
            # Check if edge (parent, v) or undirected link was found
            if G.has_edge(parent, v) or G.has_edge(v, parent):
                detected += 1
        recall = round(detected / max(1, total), 4)
        breakdown[trans] = {
            "total": total,
            "detected": detected,
            "recall": recall,
        }

    # 6. Success and Failure Cases
    successful_cases = []
    failure_cases = []

    # Check direct parent-child pairs
    for parent, child in gt_parent_child_edges:
        trans = manifest.get(child, {}).get("transformation", "unknown")
        if G.has_edge(parent, child):
            edge_data = G[parent][child]
            successful_cases.append({
                "parent": parent,
                "child": child,
                "transformation": trans,
                "direction": edge_data.get("direction"),
                "confidence": edge_data.get("confidence"),
                "reason": edge_data.get("reason"),
                "similarity": edge_data.get("similarity"),
            })
        elif G.has_edge(child, parent):
            # Reversed direction or unknown
            edge_data = G[child][parent]
            failure_cases.append({
                "parent": parent,
                "child": child,
                "transformation": trans,
                "status": "Inverted Direction / Ambiguous",
                "predicted_direction": edge_data.get("direction"),
                "reason": edge_data.get("reason"),
            })
        else:
            failure_cases.append({
                "parent": parent,
                "child": child,
                "transformation": trans,
                "status": "Missing Connection (False Negative)",
                "reason": "Similarity below threshold or unlinked",
            })

    # Check for False Positives (connected across unrelated images)
    for u, v in pred_undirected_pairs:
        if (u, v) not in gt_related_pairs:
            failure_cases.append({
                "parent": u,
                "child": v,
                "transformation": "unrelated",
                "status": "Incorrect Connection (False Positive)",
                "reason": "System falsely connected unrelated images",
            })

    return EvaluationMetrics(
        tp_pairs=tp_pairs,
        fp_pairs=fp_pairs,
        fn_pairs=fn_pairs,
        pairwise_precision=p_prec,
        pairwise_recall=p_rec,
        pairwise_f1=p_f1,
        tp_lineage=tp_dir,
        fp_lineage=fp_dir,
        fn_lineage=fn_dir,
        lineage_precision=d_prec,
        lineage_recall=d_rec,
        lineage_f1=d_f1,
        transformation_breakdown=breakdown,
        successful_cases=successful_cases,
        failure_cases=failure_cases,
    )


def run_pipeline_and_evaluate(
    manifest_path: Union[str, Path],
    data_paths: Sequence[Union[str, Path]],
    threshold: float = 0.45,
) -> Tuple[EvaluationMetrics, nx.DiGraph]:
    """Run full pipeline on provided data directories and evaluate against manifest."""
    manifest = load_manifest(manifest_path)

    # Scan and match
    file_list: List[Path] = []
    metadata_map: Dict[str, ImageMetadata] = {}

    for p_str in data_paths:
        p = Path(p_str)
        if p.is_dir():
            for m in scan_directory(p):
                if m.is_valid:
                    metadata_map[m.filename] = m
                    file_list.append(Path(m.filepath))
        elif p.is_file():
            m = extract_metadata(p)
            if m.is_valid:
                metadata_map[m.filename] = m
                file_list.append(p)

    pairwise_results, _ = compare_image_collection(file_list, threshold=threshold)
    G = build_provenance_graph(pairwise_results, metadata_map, similarity_threshold=threshold)

    metrics = evaluate_provenance_graph(G, manifest, pairwise_results)
    return metrics, G


def print_evaluation_report(metrics: EvaluationMetrics) -> str:
    """Format evaluation metrics into an informative text report."""
    lines = []
    lines.append("=" * 60)
    lines.append(" IMAGE PROVENANCE GRAPH EVALUATION REPORT")
    lines.append("=" * 60)

    lines.append("\n1. Related-Image Detection (Undirected Pairs):")
    lines.append(f"   - True Positives (TP) : {metrics.tp_pairs}")
    lines.append(f"   - False Positives (FP): {metrics.fp_pairs}")
    lines.append(f"   - False Negatives (FN): {metrics.fn_pairs}")
    lines.append(f"   - Precision           : {metrics.pairwise_precision * 100:.1f}%")
    lines.append(f"   - Recall              : {metrics.pairwise_recall * 100:.1f}%")
    lines.append(f"   - F1-Score            : {metrics.pairwise_f1 * 100:.1f}%")

    lines.append("\n2. Directed Lineage Reconstruction (Parent -> Child):")
    lines.append(f"   - Direct Hits (TP)    : {metrics.tp_lineage}")
    lines.append(f"   - Extra/Inverted (FP) : {metrics.fp_lineage}")
    lines.append(f"   - Missing Edges (FN)  : {metrics.fn_lineage}")
    lines.append(f"   - Direction Precision : {metrics.lineage_precision * 100:.1f}%")
    lines.append(f"   - Direction Recall    : {metrics.lineage_recall * 100:.1f}%")
    lines.append(f"   - Lineage F1-Score    : {metrics.lineage_f1 * 100:.1f}%")

    lines.append("\n3. Performance by Transformation Category:")
    for trans, stats in metrics.transformation_breakdown.items():
        rec = float(stats["recall"]) * 100
        lines.append(f"   - {trans:<24}: {stats['detected']}/{stats['total']} detected ({rec:.1f}%)")

    lines.append(f"\n4. Qualitative Analysis:")
    lines.append(f"   - Successful reconstructions: {len(metrics.successful_cases)}")
    lines.append(f"   - Edge failures / ambiguities: {len(metrics.failure_cases)}")

    if metrics.failure_cases:
        lines.append("\n   Notable Failure / Ambiguity Cases:")
        for fc in metrics.failure_cases[:5]:
            lines.append(f"     • {fc.get('parent')} -> {fc.get('child')} [{fc.get('transformation')}]: {fc.get('status')} ({fc.get('reason')})")

    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> None:
    """CLI tool to execute evaluation against manifest.csv."""
    parser = argparse.ArgumentParser(
        description="Evaluate Image Provenance Graph against ground-truth manifest."
    )
    parser.add_argument(
        "-m",
        "--manifest",
        default="data/manifest.csv",
        help="Path to manifest.csv ground truth.",
    )
    parser.add_argument(
        "-d",
        "--data",
        nargs="+",
        default=["data/originals", "data/variants"],
        help="Paths to image directories.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.45,
        help="Similarity threshold.",
    )
    parser.add_argument(
        "-o",
        "--output-json",
        default="outputs/evaluation_report.json",
        help="Path to export evaluation metrics JSON.",
    )

    args = parser.parse_args()

    print(f"Loading manifest: {args.manifest}")
    print(f"Analyzing images from: {args.data}")

    metrics, G = run_pipeline_and_evaluate(
        manifest_path=args.manifest,
        data_paths=args.data,
        threshold=args.threshold,
    )

    report_str = print_evaluation_report(metrics)
    print("\n" + report_str)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\nSaved metrics JSON to: {out_p}")


if __name__ == "__main__":
    main()
