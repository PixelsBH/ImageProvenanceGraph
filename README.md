# Image Provenance Graph Reconstruction

A multimedia forensic framework for visual lineage analysis. This system compares collections of related images to estimate relationships and construct a provenance graph showing which images were derived from others.

---

## Quickstart

### 1. Environment Setup

Ensure Python 3.11+ is installed. Activate or create the virtual environment and install dependencies:

```bash
# Create and activate virtual environment
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install required dependencies
uv pip install -r requirements.txt
```

### 2. Generate Sample Dataset

Generate test originals, transformed variants (resized, cropped, compressed, brightened), distractors, and manifest:

```bash
python scripts/generate_sample_data.py
```

### 3. Read Images and Inspect Metadata (Step 2)

Inspect an image collection and display a formatted metadata table:

```bash
# Inspect original images
python -m src.metadata data/originals

# Inspect all images and export metadata table to CSV
python -m src.metadata data/originals data/variants -o outputs/metadata.csv
```

### 4. Perceptual Hashing & Similarity (Step 3)

Compute perceptual hashes (pHash, dHash, wHash, aHash) and calculate pairwise visual similarity:

```bash
python -m src.hashing data/originals data/variants --top 15
```

### 5. SIFT Feature Matching & Combined Similarity (Step 3)

Extract local SIFT descriptors, execute Lowe's ratio test and RANSAC geometric verification, and compute combined similarity scores:

```bash
python -m src.matching data/originals data/variants --threshold 0.35 -o outputs/pairwise_matches.csv
```

### 6. Reconstruct Provenance Graph (Step 4)

Construct a NetworkX directed graph, estimate parent-to-child lineage directions based on multi-factor forensic clues (resolution downsampling, field-of-view cropping, compression artifacts, EXIF timestamps), and export edge tables and visualizations:

```bash
python -m src.graph_builder data/originals data/variants --threshold 0.45 -g outputs/provenance_graph.png -o outputs/edges.csv
```

### 7. Run Unit Tests

Execute the comprehensive automated test suite with pytest:

```bash
pytest -v
```

---

## Project Structure

```text
ImageProvenanceGraph/
├── src/
│   ├── __init__.py
│   ├── metadata.py       # Image loading, validation, EXIF extraction, table display
│   ├── hashing.py        # Perceptual hashing (pHash, dHash, wHash) and comparisons
│   ├── matching.py       # SIFT feature extraction, RANSAC matching, combined scoring
│   └── graph_builder.py  # NetworkX provenance graph, direction estimation, DAG pruning, plotting
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
│   └── provenance_graph.png  # Rendered graph visualization
├── tests/
│   ├── test_metadata.py  # Unit tests for metadata and validation
│   ├── test_hashing.py   # Unit tests for perceptual hashing
│   ├── test_matching.py  # Unit tests for SIFT feature matching
│   └── test_graph.py     # Unit tests for graph construction and direction estimation
├── pytest.ini
├── requirements.txt
├── README.md
└── plan.md
```
