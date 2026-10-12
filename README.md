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

### 4. Run Unit Tests

Execute the automated test suite with pytest:

```bash
pytest -v
```

---

## Project Structure

```text
ImageProvenanceGraph/
├── src/
│   ├── __init__.py
│   └── metadata.py       # Image loading, validation, EXIF extraction, and table display
├── scripts/
│   └── generate_sample_data.py # Synthetic dataset generator for testing
├── data/
│   ├── originals/        # Source images
│   ├── variants/         # Edited/transformed variants
│   └── manifest.csv      # Ground-truth transformation log
├── outputs/
│   └── metadata.csv      # Exported metadata reports
├── tests/
│   └── test_metadata.py  # Unit tests for metadata and validation
├── pytest.ini
├── requirements.txt
├── README.md
└── plan.md
```
