# Image Provenance Graph Reconstruction
### A Multimedia Forensic Framework for Visual Lineage Analysis

## 1. Project Overview

Build a simple tool that compares a collection of related images and creates a graph showing which images are likely to be derived from others.

For example, an original image may be resized, cropped, or compressed. The system compares these versions and estimates their relationships.

**Important:** The tool estimates relationships based on image similarity. It cannot prove the exact history of an image.

## 2. Main Features

- Upload or select a collection of images.
- Display basic image information such as dimensions and file type.
- Compare images using perceptual hashing and SIFT feature matching.
- Estimate which images are related.
- Display the relationships as a graph.
- Evaluate the results using images whose edit history is already known.

## 3. Technology Stack

- **Python** — main programming language
- **Pillow** — image loading and metadata
- **ImageHash** — perceptual hashing
- **OpenCV** — SIFT feature matching
- **NetworkX** — building the graph
- **Streamlit** — simple web interface
- **pytest** — optional testing

## 4. How It Will Work

```text
Input images
     |
     v
Read image information
     |
     v
Calculate perceptual hashes
     |
     v
Compare SIFT features
     |
     v
Calculate image similarity
     |
     v
Connect likely related images
     |
     v
Display provenance graph
     |
     v
Compare results with known relationships
```

## 5. Dataset

Create a small dataset using around 10 original images. For each original, generate a few edited versions.

Include transformations such as:
- Resizing
- Cropping
- JPEG compression
- Brightness adjustment

Keep a simple CSV file recording each image's original and parent image. Use this information only to evaluate the results, not when comparing images.

Also include a few unrelated images to check whether the system incorrectly connects them.

## 6. Implementation Plan

### Step 1 — Set up the project
- [x] Create the repository and Python virtual environment.
- [x] Install the required libraries.
- [x] Create folders for original images, edited images, and results.

### Step 2 — Read images and metadata
- [x] Load images and handle invalid files.
- [x] Record filenames, dimensions, file types, and available EXIF data.
- [x] Display the image information in a table.

### Step 3 — Compare images
- [ ] Calculate a perceptual hash for each image.
- [ ] Compare hashes to find visually similar images.
- [ ] Use OpenCV SIFT to compare local image features.
- [ ] Combine the results into a simple similarity score.

### Step 4 — Build the graph
- [ ] Treat each image as a node.
- [ ] Connect images whose similarity scores pass a chosen threshold.
- [ ] Use known transformations and available clues to estimate edge direction.
- [ ] Leave the direction unknown when there is not enough evidence.

### Step 5 — Build the interface
- [ ] Create a simple Streamlit interface.
- [ ] Allow users to provide a collection of images.
- [ ] Display image details and similarity scores.
- [ ] Show the generated graph and its connections.

### Step 6 — Test and evaluate
- [ ] Test on resized, cropped, compressed, and brightness-adjusted images.
- [ ] Compare the generated graph with the known relationships.
- [ ] Calculate precision, recall, and F1-score for predicted connections.
- [ ] Record examples where the system succeeds or fails.

## 7. Suggested Folder Structure

```text
image-provenance-graph/
├── app.py
├── requirements.txt
├── README.md
├── plan.md
├── src/
│   ├── metadata.py
│   ├── hashing.py
│   ├── matching.py
│   ├── graph_builder.py
│   └── evaluation.py
├── data/
│   ├── originals/
│   ├── variants/
│   └── manifest.csv
└── outputs/
```

## 8. Division of Work

For a group of three:

- **Member 1:** Image loading, metadata extraction, and perceptual hashing.
- **Member 2:** SIFT matching and similarity scoring.
- **Member 3:** Graph generation, Streamlit interface, and evaluation integration.

All members should understand the complete pipeline and help with final testing and presentation.

## 9. Evaluation

Evaluate two things:

1. **Related-image detection:** How accurately does the system identify related image pairs?
2. **Graph reconstruction:** How closely does the generated graph match the known parent-child relationships?

Use precision, recall, and F1-score. Report actual results rather than assuming the system will be highly accurate.

## 10. Minimum Deliverable

The project is complete when it can:

- Analyze a collection of images.
- Show basic metadata and similarity scores.
- Generate a graph of likely image relationships.
- Compare the predicted connections with known relationships.
- Run through a simple interface for the project demonstration.

## 11. Limitations

- Similarity does not prove that one image was derived from another.
- EXIF metadata may be missing or altered.
- Cropping and major edits can make matching difficult.
- The system may create incorrect connections or fail to identify real ones.
- Parent-child direction may be uncertain.

## 12. First Tasks

Start with a small working version:

1. Set up Python and install Pillow, ImageHash, OpenCV, and NetworkX.
2. Add three original images.
3. Generate resized, compressed, cropped, and brightness-adjusted versions.
4. Implement perceptual hashing and print a table of image similarities.
5. Add SIFT matching once the hash comparison works.
6. Build the graph and interface after the comparisons are working.

**Keep the first version small.** Focus on comparing related images, generating a clear graph, and measuring how well it works. Add extra features only if time permits.
