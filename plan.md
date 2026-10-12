# Image Provenance Graph Reconstruction
## A Multimedia Forensic Framework for Visual Lineage Analysis

> **Project goal:** Build a demonstrable forensic tool that takes a collection of related images and estimates their parent–child relationships, then visualizes those relationships as a provenance graph.
>
> **Scope note:** This is an evidence-supported *lineage estimation* system, not a guaranteed reconstruction of every edit ever made to an image. Metadata and visual similarity can be missing, misleading, or ambiguous.

---

## 1. Problem Statement

When investigators encounter multiple versions of an image, it can be difficult to determine which files are related and how they may have been derived from one another. This project will combine image metadata, perceptual hashing, and local visual-feature matching to estimate relationships between related images and present the results as an interpretable graph.

## 2. Objectives

1. Extract useful metadata from images, where available.
2. Compare images using complementary similarity signals.
3. Estimate likely relationships between image versions.
4. Construct and visualize a directed provenance graph.
5. Evaluate relationship detection against a known ground-truth dataset.
6. Explain uncertainty and avoid presenting inferred relationships as proven facts.

## 3. Target Scope and Non-Goals

### In scope
- JPEG and PNG inputs initially.
- Images derived from a known source through operations such as resizing, JPEG re-encoding, brightness/contrast adjustment, and cropping.
- Metadata extraction, perceptual hashing, SIFT matching, graph construction, and a simple web interface.
- A repeatable evaluation on synthetic image variants with known lineage.

### Out of scope for the first version
- Reconstructing arbitrary edits to unrelated images.
- Proving who edited an image or when an edit occurred.
- Reliably identifying all splicing, generative-AI edits, or malicious manipulation.
- Treating EXIF timestamps or software tags as trustworthy proof.
- Training a custom deep-learning model.

## 4. Proposed Architecture

```text
Image collection
      |
      v
Input validation + file inventory
      |
      +----------------------+
      |                      |
      v                      v
Metadata extraction     Image normalization
(EXIF / dimensions)     (decode, grayscale)
      |                      |
      +-----------+----------+
                  |
                  v
       Pairwise similarity analysis
       - Perceptual hash distance
       - SIFT keypoint matches
       - Metadata clues (weak evidence)
                  |
                  v
       Similarity score + evidence record
                  |
                  v
       Relationship candidate selection
                  |
                  v
       Directed provenance graph
                  |
                  v
       Interactive graph + evidence panel
                  |
                  v
       Evaluation against ground truth
```

## 5. Technology Stack

### Core
- **Python 3.11+**
- **OpenCV** (`opencv-contrib-python` if required for SIFT in your environment)
- **Pillow** for image loading and EXIF access
- **ImageHash** for perceptual hashing
- **NetworkX** for graph construction and graph metrics
- **pytest** for tests

### Interface (choose one)
- **Recommended first:** Streamlit for a fast upload-and-visualize demo.
- Alternative: Flask backend + a separate frontend if the team wants more UI work.
- For graph display, use `pyvis` or Plotly; first confirm the chosen library works with the interface framework.

### Suggested dependencies
```text
opencv-contrib-python
Pillow
ImageHash
NetworkX
streamlit
pyvis
pandas
pytest
```

Use a virtual environment. Pin versions in `requirements.txt` after the first successful setup. Avoid installing both `opencv-python` and `opencv-contrib-python` into the same environment.

## 6. Data Model

Each image node should include:
- `image_id`: stable ID within the current dataset
- `filename`
- `width`, `height`, `format`, `file_size_bytes`
- `exif`: selected fields, if present
- `phash`: perceptual hash representation
- `feature_count`: number of extracted SIFT keypoints
- `source_id` and `parent_id`: ground truth only; never use these as model inputs
- `analysis_notes`: warnings, e.g. missing metadata or insufficient features

Each candidate edge should include:
- `source_image`
- `target_image`
- `direction`: proposed parent → child
- `phash_distance`
- `sift_good_matches`
- `sift_match_ratio`
- `metadata_clues`
- `combined_score`
- `decision` and `confidence_band`
- `evidence_explanation`

**Important:** Keep ground-truth labels separate from calculated features. Do not let the known parent/source labels leak into scoring.

## 7. Matching and Scoring Strategy

Implement the pipeline incrementally. Keep individual signals visible instead of hiding everything inside one opaque score.

### Signal A — Perceptual hash
- Calculate a pHash for every image.
- Compare hashes using Hamming distance.
- Lower distance usually means more similar overall appearance.
- Cropping, large edits, and substantial color changes may reduce its usefulness.

### Signal B — SIFT feature matching
- Extract SIFT keypoints and descriptors.
- Match descriptors between image pairs using a suitable OpenCV matcher.
- Use Lowe's ratio test to filter ambiguous matches.
- Optionally use RANSAC/homography to estimate geometrically consistent matches.
- Record both raw match count and a normalized match ratio; raw counts alone favor larger or more textured images.
- Cropping and resizing can preserve local features, but low-texture images may produce few useful matches.

### Signal C — Metadata clues
- Record image dimensions, format, file size, EXIF capture time, camera model, and software tag when available.
- Treat metadata as supporting context only. Metadata can be removed, altered, copied, or rewritten during export.
- Do not assume that a later timestamp or a software tag proves parent–child direction.

### Combined score
Start with a transparent, configurable weighted score based on normalized pHash similarity and SIFT match evidence. Use metadata as a separate supporting clue initially rather than a strong score component. Tune thresholds on a validation split, then freeze them before the final test.

Do not pick arbitrary weights and report them as objectively correct. Document how thresholds were selected and include ablation tests:
1. pHash only
2. SIFT only
3. pHash + SIFT
4. pHash + SIFT + metadata context

## 8. Inferring Direction

Similarity tells you that images may be related; it does **not** reliably tell you which image came first.

For the first version, use known edit operations and dataset manifests to evaluate parent-child direction. The app itself must not use those ground-truth labels during inference.

For a practical heuristic, consider:
- Original dimensions versus downscaled dimensions.
- Evidence of crop/resize relationships.
- Metadata/software clues, with low weight.
- Whether one image is a plausible transformed version of another.

Keep direction as `unknown` when evidence is insufficient. Do not force every pair into a directed edge. A defensible graph with uncertain edges is better than a confidently wrong lineage.

## 9. Dataset Plan

Create a reproducible dataset from around 10–20 source photographs. Prefer images with varied textures, subjects, and resolutions, and use images you are allowed to use.

For each source image, create a small lineage tree with scripts, for example:

```text
source_01.jpg
├── resize_01.jpg
│   └── jpeg_reencode_01.jpg
├── brightness_01.jpg
└── crop_01.jpg
```

Recommended transformations:
- Resize
- JPEG recompression at several quality settings
- Brightness / contrast adjustment
- Mild color adjustment
- Cropping
- Combinations of two transformations

Store a manifest such as `data/ground_truth/manifest.csv` containing:
- `filename`
- `source_id`
- `parent_filename`
- `operations`

Keep the manifest out of the inference pipeline. Use it only for evaluation.

### Dataset split
Split by original source image, not by individual variant, so variants of the same source do not appear in both tuning and test sets. With a small project dataset, report this limitation clearly.

## 10. Evaluation Plan

Evaluate two different tasks separately.

### A. Related-image retrieval / pair classification
- Precision, recall, and F1 for identifying related pairs.
- Precision-recall trade-off at different thresholds.
- Compare the individual signals and their combination.

### B. Parent-child edge reconstruction
- Edge precision, recall, and F1 against the ground-truth lineage.
- Report direction accuracy only for edges where the system predicts a direction.
- Count unknown / rejected edges separately.
- Show graph examples where the system succeeds and fails.

Also measure:
- Runtime as the number of images increases.
- Number of candidate pairs examined.
- Robustness by transformation type.

For `N` images, all-pairs matching requires `N(N-1)/2` comparisons. This is fine for a small demo dataset, but should be discussed as a scalability limitation.

Do not promise high accuracy before running experiments. Report actual results and explain failure cases.

## 11. Implementation Milestones

### Milestone 0 — Setup and design
- [ ] Create repository and virtual environment.
- [ ] Install dependencies and verify OpenCV SIFT availability.
- [ ] Agree on input formats, data schema, and the first demo workflow.
- [ ] Create a small sample dataset and a ground-truth manifest.

**Deliverable:** A working environment and a documented sample dataset.

### Milestone 1 — Image ingestion and metadata
- [ ] Validate file type and handle unreadable images.
- [ ] Extract dimensions, format, file size, and available EXIF fields.
- [ ] Generate a file inventory table.
- [ ] Add unit tests for images with and without EXIF.

**Deliverable:** Image inventory and metadata report.

### Milestone 2 — Similarity features
- [ ] Implement pHash extraction and Hamming distance.
- [ ] Implement SIFT extraction and descriptor matching.
- [ ] Add Lowe's ratio test and optional RANSAC geometric verification.
- [ ] Store pairwise results in a CSV/JSON file.
- [ ] Test on exact duplicates, resized images, recompressed images, crops, and unrelated images.

**Deliverable:** Pairwise similarity report with interpretable evidence.

### Milestone 3 — Scoring and candidate edges
- [ ] Normalize the individual similarity signals.
- [ ] Implement configurable thresholds and a transparent combined score.
- [ ] Select candidate edges while avoiding redundant all-to-all links.
- [ ] Add `unknown` direction and uncertain-edge handling.
- [ ] Record a short reason for each accepted/rejected edge.

**Deliverable:** Candidate parent-child edge list.

### Milestone 4 — Graph reconstruction
- [ ] Create nodes for images and directed edges for proposed lineage.
- [ ] Detect cycles and resolve or flag contradictory relationships.
- [ ] Generate a graph visualization.
- [ ] Show each edge's evidence and confidence band.
- [ ] Export graph data and an image of the graph.

**Deliverable:** Provenance graph for a sample collection.

### Milestone 5 — Interface
- [ ] Build a simple dataset upload / folder selection workflow.
- [ ] Show thumbnails and metadata for each image.
- [ ] Display similarity scores and candidate relationships.
- [ ] Display the graph and an edge-evidence panel.
- [ ] Provide clear warnings that results are estimates, not proof.

**Deliverable:** End-to-end demonstration.

### Milestone 6 — Evaluation and polish
- [ ] Run tests on a held-out source-image split.
- [ ] Calculate precision, recall, F1, and direction accuracy where applicable.
- [ ] Run ablation comparisons.
- [ ] Document failure cases and limitations.
- [ ] Prepare demo examples, slides, and a reproducible run guide.

**Deliverable:** Final report, results tables, and presentation-ready demo.

## 12. Suggested Repository Structure

```text
image-provenance-graph/
├── app.py
├── requirements.txt
├── README.md
├── plan.md
├── src/
│   ├── ingest.py
│   ├── metadata.py
│   ├── hashing.py
│   ├── sift_matching.py
│   ├── similarity.py
│   ├── lineage.py
│   ├── graph_builder.py
│   └── evaluation.py
├── scripts/
│   ├── generate_dataset.py
│   └── run_evaluation.py
├── data/
│   ├── originals/
│   ├── variants/
│   └── ground_truth/
│       └── manifest.csv
├── outputs/
│   ├── pairwise_scores.csv
│   ├── candidate_edges.csv
│   └── metrics.json
└── tests/
    ├── test_metadata.py
    ├── test_hashing.py
    ├── test_matching.py
    └── test_graph.py
```

Keep generated data and outputs out of version control if they become large. Include a small, redistributable sample or instructions for recreating the dataset.

## 13. Suggested Division of Work (3 Members)

### Member A — Ingestion and metadata
- File validation, image inventory, EXIF extraction.
- pHash implementation and tests.
- Dataset generation support.

### Member B — Visual matching and lineage
- SIFT matching, ratio test, optional RANSAC.
- Similarity normalization and scoring.
- Candidate-edge and direction heuristics.

### Member C — Graph, interface, and evaluation integration
- NetworkX graph generation and visualization.
- Streamlit UI.
- Evaluation scripts and report integration.

All members should understand the end-to-end pipeline, the meaning of each score, and the limitations of direction inference.

## 14. Demo Script (5–7 Minutes)

1. Upload a small collection containing originals, edited variants, and unrelated images.
2. Show the extracted metadata and missing-metadata warnings.
3. Show the pairwise similarity evidence.
4. Generate and inspect the provenance graph.
5. Open an edge and explain why the system proposed it.
6. Compare the predicted graph against known ground truth.
7. Show precision/recall/F1 and one failure case.

## 15. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Similarity does not establish lineage | Separate pair similarity from direction inference |
| EXIF is missing or manipulated | Treat metadata as weak supporting evidence |
| SIFT fails on smooth or heavily edited images | Use multiple signals and expose insufficient evidence |
| Too many false graph edges | Tune thresholds on validation data; reject weak candidates |
| Graph direction is unreliable | Allow unknown direction and evaluate direction separately |
| Dataset is too easy | Include unrelated images and combined transformations |
| Scope grows too large | Finish the end-to-end baseline before adding features |

## 16. Minimum Viable Product vs Stretch Goals

### Minimum viable product (must finish first)
- Metadata extraction
- pHash similarity
- SIFT feature matching
- Candidate related-image pairs
- Basic provenance graph
- Ground-truth evaluation
- Simple Streamlit interface

### Stretch goals (only after MVP works)
- RANSAC-based geometric verification
- Interactive graph filtering and edge explanations
- Additional image transformations
- Robustness comparison across transformation types
- Optional OCR or embedding-based retrieval (clearly labeled as an extension, not forensic proof)

## 17. Definition of Done

The project is ready when:
- [ ] A new image collection can be analyzed end-to-end.
- [ ] The system shows metadata and pairwise visual evidence.
- [ ] It generates a graph without using ground-truth labels at inference time.
- [ ] Evaluation is repeatable and reports actual metrics.
- [ ] At least one successful and one failed reconstruction are documented.
- [ ] A new machine can run the demo using the README instructions.
- [ ] The team can explain the system's assumptions, errors, and limitations.

## 18. First Session Checklist

Start small and get a vertical slice working:

1. Create the repository and Python virtual environment.
2. Install the core dependencies.
3. Put 3 original images into `data/originals/`.
4. Generate resized, recompressed, brightness-adjusted, and cropped variants.
5. Implement metadata extraction and pHash first.
6. Print a pairwise similarity table.
7. Add SIFT only after the baseline works.

**Do not start with the graph UI.** First establish that the image comparisons provide useful evidence; then build the lineage and visualization layers around them.
