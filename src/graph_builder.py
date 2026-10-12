"""Provenance graph reconstruction and lineage analysis module.

Builds a directed provenance graph from pairwise image similarities:
1. Treats each analyzed image as a graph node with full metadata.
2. Creates edges between images exceeding a similarity threshold.
3. Estimates parent-to-child lineage direction using multi-factor forensic clues:
   - Resolution/pixel area downscaling
   - Field-of-view cropping
   - File size & JPEG compression loss
   - EXIF capture timestamps
4. Flags ambiguous relationships as UNKNOWN direction.
5. Visualizes and exports the provenance graph.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from src.matching import PairwiseMatchResult, compare_image_collection
from src.metadata import ImageMetadata, extract_metadata, scan_directory


def parse_exif_timestamp(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse standard EXIF timestamp format 'YYYY:MM:DD HH:MM:SS' into datetime."""
    if not dt_str or dt_str == "None":
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M"):
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    return None


def estimate_edge_direction(
    meta_a: Optional[ImageMetadata],
    meta_b: Optional[ImageMetadata],
    match_result: PairwiseMatchResult,
    resolution_ratio_threshold: float = 1.15,
) -> Tuple[str, float, str]:
    """Infer likely parent-to-child direction between image A and image B.

    Returns:
        (direction, confidence, reason)
        direction is one of:
          - "A_TO_B" (A is parent -> B is child)
          - "B_TO_A" (B is parent -> A is child)
          - "UNKNOWN" (inconclusive evidence)
    """
    if meta_a is None or meta_b is None or not meta_a.is_valid or not meta_b.is_valid:
        return "UNKNOWN", 0.0, "Missing or invalid metadata"

    votes_a_to_b = 0.0
    votes_b_to_a = 0.0
    reasons_a_to_b: List[str] = []
    reasons_b_to_a: List[str] = []

    # 1. Pixel Area / Resolution Clue (Downscaling is irreversible)
    area_a = (meta_a.width or 0) * (meta_a.height or 0)
    area_b = (meta_b.width or 0) * (meta_b.height or 0)

    if area_a > 0 and area_b > 0:
        area_ratio = area_a / area_b
        if area_ratio >= resolution_ratio_threshold:
            votes_a_to_b += 2.0
            reasons_a_to_b.append(
                f"Downscaled ({meta_a.width}x{meta_a.height} -> {meta_b.width}x{meta_b.height})"
            )
        elif area_ratio <= (1.0 / resolution_ratio_threshold):
            votes_b_to_a += 2.0
            reasons_b_to_a.append(
                f"Downscaled ({meta_b.width}x{meta_b.height} -> {meta_a.width}x{meta_a.height})"
            )

    # 2. Cropping Clue (SIFT inlier coverage vs image dimensions)
    # If SIFT similarity is high but pHash is weak, likely a crop
    if match_result.num_inliers >= 10 and match_result.phash_dist >= 20:
        if area_a > area_b:
            votes_a_to_b += 2.5
            reasons_a_to_b.append("Cropped field-of-view")
        elif area_b > area_a:
            votes_b_to_a += 2.5
            reasons_b_to_a.append("Cropped field-of-view")

    # 3. File Size / Recompression Clue (for images with similar dimensions)
    size_a = meta_a.file_size_bytes
    size_b = meta_b.file_size_bytes
    if area_a > 0 and area_b > 0 and 0.9 <= (area_a / area_b) <= 1.1:
        # Same dimensions: check compression/file size ratio
        if size_a >= 1.6 * size_b and size_b > 0:
            votes_a_to_b += 1.5
            reasons_a_to_b.append(f"Recompression/quality loss ({meta_a.file_size_human} -> {meta_b.file_size_human})")
        elif size_b >= 1.6 * size_a and size_a > 0:
            votes_b_to_a += 1.5
            reasons_b_to_a.append(f"Recompression/quality loss ({meta_b.file_size_human} -> {meta_a.file_size_human})")

    # 4. EXIF Timestamp Clue
    date_a = parse_exif_timestamp(meta_a.exif_data.get("DateTimeOriginal") or meta_a.exif_data.get("DateTime"))
    date_b = parse_exif_timestamp(meta_b.exif_data.get("DateTimeOriginal") or meta_b.exif_data.get("DateTime"))

    if date_a and date_b and date_a != date_b:
        if date_a < date_b:
            votes_a_to_b += 1.0
            reasons_a_to_b.append(f"Earlier capture timestamp ({date_a.strftime('%Y-%m-%d %H:%M')})")
        else:
            votes_b_to_a += 1.0
            reasons_b_to_a.append(f"Earlier capture timestamp ({date_b.strftime('%Y-%m-%d %H:%M')})")

    # Determine verdict based on accumulated evidence
    total_votes = votes_a_to_b + votes_b_to_a
    if total_votes == 0:
        return "UNKNOWN", 0.3, "Identical dimensions & no directional evidence"

    margin = abs(votes_a_to_b - votes_b_to_a)
    if margin < 1.0:
        # Inconclusive / conflicting evidence
        return "UNKNOWN", 0.4, "Conflicting or balanced directional clues"

    if votes_a_to_b > votes_b_to_a:
        confidence = round(min(0.95, 0.5 + (margin / 6.0)), 2)
        reason = "; ".join(reasons_a_to_b)
        return "A_TO_B", confidence, reason
    else:
        confidence = round(min(0.95, 0.5 + (margin / 6.0)), 2)
        reason = "; ".join(reasons_b_to_a)
        return "B_TO_A", confidence, reason


def build_provenance_graph(
    pairwise_results: Sequence[PairwiseMatchResult],
    metadata_map: Dict[str, ImageMetadata],
    similarity_threshold: float = 0.45,
    resolve_cycles: bool = True,
) -> nx.DiGraph:
    """Build a directed lineage graph from pairwise match results.

    Args:
        pairwise_results: List of pairwise match evaluations.
        metadata_map: Dictionary mapping filename/path to ImageMetadata.
        similarity_threshold: Minimum combined similarity score to form an edge.
        resolve_cycles: If True, prunes low-confidence cycle edges to maintain a DAG.

    Returns:
        networkx.DiGraph representing the provenance structure.
    """
    G = nx.DiGraph()

    # 1. Add all images as nodes
    for name, meta in metadata_map.items():
        if meta.is_valid:
            G.add_node(
                meta.filename,
                filepath=meta.filepath,
                width=meta.width,
                height=meta.height,
                megapixels=meta.megapixels,
                size_human=meta.file_size_human,
                format=meta.format,
                exif_summary=meta.exif_summary,
            )

    # 2. Add edges for pairs exceeding threshold
    for match in pairwise_results:
        if match.combined_score < similarity_threshold:
            continue

        meta_a = metadata_map.get(match.img1) or metadata_map.get(match.path1)
        meta_b = metadata_map.get(match.img2) or metadata_map.get(match.path2)

        direction, confidence, reason = estimate_edge_direction(meta_a, meta_b, match)

        edge_attrs = {
            "weight": match.combined_score,
            "similarity": match.combined_score,
            "hash_sim": match.hash_similarity,
            "sift_sim": match.sift_similarity,
            "inliers": match.num_inliers,
            "confidence": confidence,
            "direction": direction,
            "reason": reason,
        }

        if direction == "A_TO_B":
            G.add_edge(match.img1, match.img2, **edge_attrs)
        elif direction == "B_TO_A":
            G.add_edge(match.img2, match.img1, **edge_attrs)
        else:
            # UNKNOWN direction: add edge with direction='UNKNOWN' (or bidirectional)
            G.add_edge(match.img1, match.img2, **edge_attrs)

    # 3. Cycle resolution: prune lowest confidence edge in any simple cycle
    if resolve_cycles and not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            if len(cycle) >= 2:
                # Find edge with lowest confidence in the cycle
                cycle_edges = list(zip(cycle, cycle[1:] + [cycle[0]]))
                min_edge = min(
                    cycle_edges,
                    key=lambda e: G[e[0]][e[1]].get("confidence", 0.0) if G.has_edge(e[0], e[1]) else 1.0,
                )
                if G.has_edge(min_edge[0], min_edge[1]):
                    G.remove_edge(min_edge[0], min_edge[1])

    return G


def export_graph_to_dataframe(G: nx.DiGraph) -> pd.DataFrame:
    """Export provenance graph edges into a clean pandas DataFrame."""
    rows = []
    for u, v, data in G.edges(data=True):
        rows.append({
            "Source (Parent)": u,
            "Target (Child)": v,
            "Similarity": round(data.get("similarity", 0.0), 3),
            "Confidence": round(data.get("confidence", 0.0), 2),
            "Direction": data.get("direction", "UNKNOWN"),
            "Inliers": data.get("inliers", 0),
            "Reason": data.get("reason", ""),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="Similarity", ascending=False).reset_index(drop=True)
    return df


def render_graph_plot(
    G: nx.DiGraph,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Image Provenance Graph",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Render a clean, publication-ready network diagram of the provenance graph."""
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty Graph (No nodes)", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    # Layout: hierarchical if possible, else spring layout
    pos = nx.spring_layout(G, k=1.2, seed=42)

    # Classify nodes: Roots (in-degree == 0, out-degree > 0) vs Leaves vs Isolated
    root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) > 0]
    leaf_nodes = [n for n in G.nodes() if G.in_degree(n) > 0]
    isolated_nodes = [n for n in G.nodes() if G.degree(n) == 0]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=root_nodes, node_color="#2b8a3e", node_size=1800, alpha=0.9, ax=ax, label="Originals / Roots"
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=leaf_nodes, node_color="#1971c2", node_size=1400, alpha=0.9, ax=ax, label="Derived Variants"
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=isolated_nodes, node_color="#adb5bd", node_size=900, alpha=0.6, ax=ax, label="Unrelated / Isolated"
    )

    # Draw edges: differentiate confident directed vs unknown
    directed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("direction") != "UNKNOWN"]
    unknown_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("direction") == "UNKNOWN"]

    if directed_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=directed_edges, edge_color="#343a40", width=2.0,
            arrows=True, arrowstyle="-|>", arrowsize=18, ax=ax
        )
    if unknown_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=unknown_edges, edge_color="#fa5252", width=1.8,
            style="dashed", arrows=False, ax=ax
        )

    # Draw labels
    labels = {n: n.replace(".jpg", "").replace(".png", "") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold", ax=ax)

    # Edge labels showing similarity score
    edge_labels = {
        (u, v): f"{d.get('similarity', 0.0):.2f}"
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.axis("off")
    fig.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"Graph plot saved to: {out}")

    return fig


def main() -> None:
    """CLI tool to build provenance graph and export visualization and edge tables."""
    parser = argparse.ArgumentParser(
        description="Reconstruct Image Provenance Graph from an image collection."
    )
    parser.add_argument("paths", nargs="+", help="Directories or images to analyze.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively scan.")
    parser.add_argument("-t", "--threshold", type=float, default=0.45, help="Similarity threshold.")
    parser.add_argument("-o", "--output-edges", help="Save edge list to CSV.")
    parser.add_argument("-g", "--output-graph", help="Save graph image (e.g. outputs/graph.png).")

    args = parser.parse_args()

    # 1. Collect and scan metadata
    metadata_map: Dict[str, ImageMetadata] = {}
    file_list: List[Path] = []
    for p_str in args.paths:
        p = Path(p_str)
        if p.is_dir():
            scanned = scan_directory(p, recursive=args.recursive)
            for m in scanned:
                if m.is_valid:
                    metadata_map[m.filename] = m
                    file_list.append(Path(m.filepath))
        elif p.is_file():
            m = extract_metadata(p)
            if m.is_valid:
                metadata_map[m.filename] = m
                file_list.append(p)

    if len(file_list) < 2:
        print("Need at least 2 valid image files to build provenance graph.")
        return

    print(f"Analyzing {len(file_list)} images...")
    pairwise_results, _ = compare_image_collection(file_list, threshold=args.threshold)

    # 2. Build graph
    G = build_provenance_graph(
        pairwise_results=pairwise_results,
        metadata_map=metadata_map,
        similarity_threshold=args.threshold,
    )

    print(f"\nConstructed Provenance Graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Is DAG: {nx.is_directed_acyclic_graph(G)}")

    # 3. Export table
    edges_df = export_graph_to_dataframe(G)
    if not edges_df.empty:
        try:
            from tabulate import tabulate

            print("\nReconstructed Provenance Edges:")
            print(tabulate(edges_df, headers="keys", tablefmt="rounded_outline", showindex=False))
        except ImportError:
            print(edges_df.to_string(index=False))

    if args.output_edges:
        out_edges = Path(args.output_edges)
        out_edges.parent.mkdir(parents=True, exist_ok=True)
        edges_df.to_csv(out_edges, index=False)
        print(f"Saved edges to: {out_edges}")

    if args.output_graph:
        render_graph_plot(G, output_path=args.output_graph)


if __name__ == "__main__":
    main()
