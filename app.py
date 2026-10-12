"""Streamlit Web Application for Image Provenance Graph Reconstruction.

Provides an interactive GUI to:
1. Ingest image collections (sample dataset, custom folder, or direct file uploads).
2. Inspect image metadata and EXIF properties in tabular and gallery views.
3. Compute perceptual hashes and SIFT feature matches with adjustable parameters.
4. Visualize reconstructed lineage graphs with directional forensic justifications.
5. Evaluate results against ground-truth manifests (precision, recall, F1, transformation breakdown).
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from PIL import Image
import streamlit as st

from src.evaluation import evaluate_provenance_graph, load_manifest
from src.graph_builder import (
    build_provenance_graph,
    export_graph_to_dataframe,
    render_graph_plot,
)
from src.matching import (
    FeatureExtractor,
    PairwiseMatchResult,
    compare_image_collection,
    compare_image_pair,
)
from src.metadata import (
    ImageMetadata,
    extract_metadata,
    load_image,
    metadata_to_dataframe,
    scan_directory,
)

# Page configuration
st.set_page_config(
    page_title="Image Provenance Graph",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished aesthetics
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #e0f2fe;
        color: #0369a1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_cached_collection(
    file_paths: List[Path],
    similarity_threshold: float,
) -> Tuple[Dict[str, ImageMetadata], List[PairwiseMatchResult], nx.DiGraph, pd.DataFrame]:
    """Execute metadata extraction, pairwise matching, and graph construction."""
    metadata_map: Dict[str, ImageMetadata] = {}
    valid_paths: List[Path] = []

    for p in file_paths:
        m = extract_metadata(p)
        metadata_map[m.filename] = m
        if m.is_valid:
            valid_paths.append(p)

    pairwise_results, _ = compare_image_collection(valid_paths, threshold=similarity_threshold)
    G = build_provenance_graph(pairwise_results, metadata_map, similarity_threshold=similarity_threshold)
    edges_df = export_graph_to_dataframe(G)

    return metadata_map, pairwise_results, G, edges_df


def main():
    st.markdown('<div class="main-header">🕸️ Image Provenance Graph Reconstruction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">A multimedia forensic framework for visual lineage analysis and image relationship estimation.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Forensic Scope & Methodology", expanded=False):
        st.info(
            "**Important Forensic Notice:**\n"
            "- Lineage direction and connections are **estimates** based on perceptual hashing, SIFT local feature geometry, and metadata heuristics.\n"
            "- Transformations like downscaling and cropping are irreversible and provide directional evidence.\n"
            "- EXIF metadata may be absent, stripped, or modified by editing software. Ambiguous relationships are classified as **UNKNOWN**."
        )

    # Sidebar: Data Source & Hyperparameters
    st.sidebar.header("📁 Image Collection")
    source_choice = st.sidebar.radio(
        "Source Selection",
        ["Sample Dataset (data/)", "Upload Images", "Local Directory"],
        index=0,
    )

    target_files: List[Path] = []
    temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None

    if source_choice == "Sample Dataset (data/)":
        orig_dir = Path("data/originals")
        var_dir = Path("data/variants")
        if orig_dir.is_dir():
            target_files.extend(list(orig_dir.glob("*.*")))
        if var_dir.is_dir():
            target_files.extend(list(var_dir.glob("*.*")))
        target_files = sorted(list(set(target_files)), key=lambda p: p.name.lower())
        st.sidebar.success(f"Loaded {len(target_files)} sample files.")

    elif source_choice == "Upload Images":
        uploaded_files = st.sidebar.file_uploader(
            "Upload image files",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tif"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = Path(temp_dir_obj.name)
            for uf in uploaded_files:
                save_p = temp_dir / uf.name
                save_p.write_bytes(uf.getbuffer())
                target_files.append(save_p)
            st.sidebar.success(f"Uploaded {len(target_files)} file(s).")

    elif source_choice == "Local Directory":
        dir_input = st.sidebar.text_input("Folder Path", value="data/originals")
        input_path = Path(dir_input)
        if input_path.is_dir():
            target_files = sorted(list(input_path.glob("*.*")), key=lambda p: p.name.lower())
            st.sidebar.success(f"Found {len(target_files)} file(s).")
        else:
            st.sidebar.error("Directory not found.")

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Analysis Parameters")
    similarity_thresh = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.45,
        step=0.05,
        help="Minimum combined similarity score required to establish a lineage edge.",
    )

    if len(target_files) < 2:
        st.warning("Please provide at least 2 image files to perform provenance analysis.")
        return

    # Run analysis
    with st.spinner("Analyzing metadata, computing hashes, and matching SIFT features..."):
        metadata_map, pairwise_results, G, edges_df = load_cached_collection(
            target_files, similarity_threshold=similarity_thresh
        )

    # Top KPI Metrics
    valid_count = sum(1 for m in metadata_map.values() if m.is_valid)
    invalid_count = len(metadata_map) - valid_count
    connected_components = nx.number_weakly_connected_components(G) if G.number_of_nodes() > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Images", len(metadata_map))
    col2.metric("Valid Images", valid_count, f"-{invalid_count} invalid" if invalid_count else "100%")
    col3.metric("Lineage Edges", G.number_of_edges(), f"Threshold: {similarity_thresh}")
    col4.metric("Image Clusters", connected_components)

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Image Metadata",
        "🔍 Pairwise Similarity",
        "🕸️ Provenance Graph",
        "📈 Ground-Truth Evaluation",
    ])

    # TAB 1: METADATA
    with tab1:
        st.subheader("Image Collection Inventory")
        meta_df = metadata_to_dataframe(list(metadata_map.values()), include_full_path=False)
        st.dataframe(meta_df, use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Download Metadata CSV",
            data=meta_df.to_csv(index=False),
            file_name="image_metadata.csv",
            mime="text/csv",
        )

        st.markdown("### Thumbnail Gallery")
        valid_items = [m for m in metadata_map.values() if m.is_valid]
        num_cols = 4
        for i in range(0, len(valid_items), num_cols):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                if i + j < len(valid_items):
                    item = valid_items[i + j]
                    with cols[j]:
                        try:
                            im = load_image(item.filepath)
                            if im:
                                st.image(im, use_container_width=True, caption=item.filename)
                                st.caption(f"{item.width}x{item.height} | {item.file_size_human} | {item.format}")
                        except Exception:
                            st.write(f"Cannot display {item.filename}")

    # TAB 2: PAIRWISE SIMILARITY
    with tab2:
        st.subheader("Pairwise Similarity & Visual Matching")
        st.markdown(
            "Shows image pairs ranked by their **Combined Score** (synthesizing pHash Hamming distance and SIFT geometric inliers)."
        )

        if not pairwise_results:
            st.info("No image pairs surpassed the chosen similarity threshold.")
        else:
            pair_rows = []
            for r in pairwise_results:
                pair_rows.append({
                    "Image 1": r.img1,
                    "Image 2": r.img2,
                    "Combined Score": r.combined_score,
                    "Hash Sim": r.hash_similarity,
                    "pHash Dist": r.phash_dist,
                    "SIFT Sim": r.sift_similarity,
                    "Inliers": r.num_inliers,
                    "Good Matches": r.num_good_matches,
                })
            pairs_df = pd.DataFrame(pair_rows)
            st.dataframe(pairs_df, use_container_width=True, hide_index=True)

            st.download_button(
                "📥 Download Pairwise Scores CSV",
                data=pairs_df.to_csv(index=False),
                file_name="pairwise_scores.csv",
                mime="text/csv",
            )

        # Side-by-Side Visual Comparison Inspector
        st.markdown("### 🔬 Inspect Pairwise Evidence")
        valid_names = [m.filename for m in metadata_map.values() if m.is_valid]
        if len(valid_names) >= 2:
            inspect_col1, inspect_col2 = st.columns(2)
            with inspect_col1:
                sel_img1 = st.selectbox("Select Image 1", valid_names, index=0)
            with inspect_col2:
                sel_img2 = st.selectbox("Select Image 2", valid_names, index=1 if len(valid_names) > 1 else 0)

            if sel_img1 != sel_img2:
                m1 = metadata_map[sel_img1]
                m2 = metadata_map[sel_img2]
                extractor = FeatureExtractor()
                comp_res = compare_image_pair(
                    m1.filepath, m2.filepath, extractor, meta1=m1, meta2=m2
                )
                if comp_res:
                    comp_c1, comp_c2 = st.columns(2)
                    with comp_c1:
                        st.image(load_image(m1.filepath), caption=f"{m1.filename} ({m1.width}x{m1.height})", use_container_width=True)
                    with comp_c2:
                        st.image(load_image(m2.filepath), caption=f"{m2.filename} ({m2.width}x{m2.height})", use_container_width=True)

                    st.markdown(
                        f"**Combined Score:** `{comp_res.combined_score:.3f}` | "
                        f"**Hash Similarity:** `{comp_res.hash_similarity:.3f}` (pHash Dist: `{comp_res.phash_dist}`) | "
                        f"**SIFT Inliers:** `{comp_res.num_inliers}` | "
                        f"**Good Matches:** `{comp_res.num_good_matches}`"
                    )

    # TAB 3: PROVENANCE GRAPH
    with tab3:
        st.subheader("Reconstructed Lineage Graph")
        st.markdown(
            "Lineage graph displaying parent-to-child provenance. "
            "**Green nodes** denote estimated source roots, **blue nodes** denote derived variants, and **gray nodes** represent isolated/unrelated images."
        )

        fig = render_graph_plot(G, title=f"Image Provenance Graph (Threshold: {similarity_thresh})")
        st.pyplot(fig)

        # Save buffer for image download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button(
            "📥 Download Graph Image (PNG)",
            data=buf.getvalue(),
            file_name="provenance_graph.png",
            mime="image/png",
        )

        st.markdown("### Lineage Edges & Forensic Reasoning")
        if not edges_df.empty:
            st.dataframe(edges_df, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Download Edges CSV",
                data=edges_df.to_csv(index=False),
                file_name="provenance_edges.csv",
                mime="text/csv",
            )
        else:
            st.info("No directed edges met the threshold criteria.")

    # TAB 4: GROUND-TRUTH EVALUATION
    with tab4:
        st.subheader("Ground-Truth Evaluation (Step 6)")
        manifest_path = Path("data/manifest.csv")

        if not manifest_path.is_file():
            st.info("No ground-truth manifest found at `data/manifest.csv`. Upload a manifest to evaluate accuracy.")
            uploaded_manifest = st.file_uploader("Upload manifest.csv", type=["csv"])
            if uploaded_manifest:
                manifest_temp = Path(tempfile.gettempdir()) / "uploaded_manifest.csv"
                manifest_temp.write_bytes(uploaded_manifest.getbuffer())
                manifest_path = manifest_temp

        if manifest_path.is_file():
            try:
                manifest = load_manifest(manifest_path)
                metrics = evaluate_provenance_graph(G, manifest, pairwise_results)

                # Metric Cards
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric("Pairwise Precision", f"{metrics.pairwise_precision * 100:.1f}%", f"TP: {metrics.tp_pairs}")
                m_c2.metric("Pairwise Recall", f"{metrics.pairwise_recall * 100:.1f}%", f"FN: {metrics.fn_pairs}")
                m_c3.metric("Pairwise F1-Score", f"{metrics.pairwise_f1 * 100:.1f}%")
                m_c4.metric("Lineage Recall", f"{metrics.lineage_recall * 100:.1f}%", f"Hits: {metrics.tp_lineage}")

                st.markdown("### Robustness by Transformation Type")
                breakdown_rows = []
                for trans, stats in metrics.transformation_breakdown.items():
                    breakdown_rows.append({
                        "Transformation": trans,
                        "Detected": stats["detected"],
                        "Total Known": stats["total"],
                        "Recall Rate": f"{stats['recall'] * 100:.1f}%",
                    })
                st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

                st.markdown("### Qualitative Case Inspector")
                succ_col, fail_col = st.columns(2)
                with succ_col:
                    st.markdown(f"**Successful Reconstructions ({len(metrics.successful_cases)}):**")
                    for case in metrics.successful_cases[:6]:
                        st.success(
                            f"**{case['parent']} ➔ {case['child']}** ({case['transformation']})\n\n"
                            f"*Reason:* {case['reason']} | *Conf:* {case['confidence']}"
                        )

                with fail_col:
                    st.markdown(f"**Ambiguous / Edge Failures ({len(metrics.failure_cases)}):**")
                    if not metrics.failure_cases:
                        st.info("No failure or ambiguous cases encountered!")
                    else:
                        for case in metrics.failure_cases[:6]:
                            st.warning(
                                f"**{case['parent']} ➔ {case['child']}**\n\n"
                                f"*{case.get('status')}:* {case.get('reason')}"
                            )

                st.download_button(
                    "📥 Download Evaluation JSON",
                    data=json.dumps(metrics.to_dict(), indent=2),
                    file_name="evaluation_metrics.json",
                    mime="application/json",
                )
            except Exception as err:
                st.error(f"Error evaluating graph: {err}")


if __name__ == "__main__":
    main()
