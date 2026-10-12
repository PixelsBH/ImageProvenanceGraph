# Image Provenance Graph Reconstruction

A multimedia forensic framework for visual lineage analysis. This system compares collections of related images to estimate relationships and construct a provenance graph showing which images were derived from others.

---

## Quickstart

### 1. Environment Setup

Ensure Python 3.11+ is installed. Activate or create the virtual environment and install dependencies:

```bash
# Create virtual environment
uv venv --python 3.11 .venv

# Activate (Bash/Zsh):
source .venv/bin/activate

# Activate (Fish shell):
source .venv/bin/activate.fish

# Install required dependencies
uv pip install -r requirements.txt
```

### 2. Generate Sample Dataset

Generate test originals, transformed variants (resized, cropped, compressed, brightened), distractors, and manifest:

```bash
python scripts/generate_sample_data.py
```

### 3. Launch the Interactive Web Dashboard (Step 5)

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The web dashboard allows you to:
- Select sample images, upload custom files, or browse local directories.
- Inspect metadata and image thumbnails in an interactive gallery.
- Compare pairwise similarity with side-by-side evidence inspection.
- Render and inspect SIFT keypoint inlier correspondence lines (`cv2.drawMatches`) verified via RANSAC.
- Visualize and download the reconstructed provenance graph and edge list.
- Evaluate reconstruction accuracy against ground truth with interactive metric cards.

---

## Command-Line Tools

### 4. Read Images and Inspect Metadata (Step 2)

```bash
# Inspect original images
python -m src.metadata data/originals

# Inspect all images and export metadata table to CSV
python -m src.metadata data/originals data/variants -o outputs/metadata.csv
```

### 5. Perceptual Hashing & Similarity (Step 3)

```bash
python -m src.hashing data/originals data/variants --top 15
```

### 6. SIFT Feature Matching & Combined Similarity (Step 3)

```bash
python -m src.matching data/originals data/variants --threshold 0.35 -o outputs/pairwise_matches.csv
```

### 7. Reconstruct Provenance Graph (Step 4)

```bash
python -m src.graph_builder data/originals data/variants --threshold 0.45 -g outputs/provenance_graph.png -o outputs/edges.csv
```

### 8. Run Ground-Truth Evaluation (Step 6)

Compare reconstructed graphs against known ground truth in `data/manifest.csv`:

```bash
python -m src.evaluation -m data/manifest.csv -d data/originals data/variants -t 0.45 -o outputs/evaluation_report.json
```

### 9. Run Test Suite

Execute the comprehensive automated test suite with pytest:

```bash
pytest -v
```

---

## Project Structure

```text
ImageProvenanceGraph/
├── app.py                # Streamlit web application dashboard (Step 5)
├── src/
│   ├── __init__.py
│   ├── metadata.py       # Image loading, validation, EXIF extraction, table display (Step 2)
│   ├── hashing.py        # Perceptual hashing (pHash, dHash, wHash) and comparisons (Step 3)
│   ├── matching.py       # SIFT feature extraction, RANSAC matching, combined scoring (Step 3)
│   ├── graph_builder.py  # NetworkX provenance graph, direction estimation, DAG pruning (Step 4)
│   └── evaluation.py     # Evaluation against ground truth: precision, recall, F1 (Step 6)
├── scripts/
│   └── generate_sample_data.py # Synthetic dataset generator for testing
├── data/
│   ├── originals/        # Source images
│   ├── variants/         # Edited/transformed variants
│   └── manifest.csv      # Ground-truth transformation log
├── outputs/
│   ├── metadata.csv          # Exported metadata reports
│   ├── pairwise_matches.csv  # SIFT + Hash pairwise similarity scores
│   ├── edges.csv             # Directed provenance edges with forensic evidence
│   ├── provenance_graph.png  # Rendered graph visualization
│   └── evaluation_report.json# Ground-truth evaluation metrics
├── tests/
│   ├── test_metadata.py  # Unit tests for metadata and validation
│   ├── test_hashing.py   # Unit tests for perceptual hashing
│   ├── test_matching.py  # Unit tests for SIFT feature matching
│   ├── test_graph.py     # Unit tests for graph construction and direction estimation
│   └── test_evaluation.py# Unit tests for ground-truth evaluation
├── pytest.ini
├── requirements.txt
├── README.md
└── plan.md
```
