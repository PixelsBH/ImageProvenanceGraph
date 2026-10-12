"""Image metadata extraction and validation module.

This module provides functionality to:
1. Load and validate image files (handling corrupt/unsupported files gracefully).
2. Extract technical metadata (filename, dimensions, format, color mode, file size).
3. Safely decode and normalize EXIF metadata (camera, datetime, software, orientation).
4. Format and present image information in tabular format (DataFrame, CLI table).
"""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from PIL import ExifTags, Image, UnidentifiedImageError

# Common image extensions supported for scanning
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
}


def format_bytes(size_bytes: int) -> str:
    """Format file size in bytes to a human-readable string."""
    if size_bytes < 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    size = float(size_bytes)
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    return f"{size:.1f} {units[unit_idx]}" if unit_idx > 0 else f"{int(size)} B"


def calculate_sha256(filepath: Union[str, Path], block_size: int = 65536) -> str:
    """Calculate SHA256 checksum of a file for exact duplicate checking."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sanitize_exif_val(val: Any) -> Any:
    """Convert raw EXIF values into clean, serializable human-readable values."""
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="replace").strip("\x00")
        except Exception:
            return f"<{len(val)} bytes>"
    if hasattr(val, "numerator") and hasattr(val, "denominator"):
        # Handle IFDRational
        if val.denominator != 0:
            return round(float(val), 4)
        return 0
    if isinstance(val, tuple):
        return tuple(_sanitize_exif_val(item) for item in val)
    if isinstance(val, list):
        return [_sanitize_exif_val(item) for item in val]
    if isinstance(val, dict):
        return {str(k): _sanitize_exif_val(v) for k, v in val.items()}
    return val


def extract_exif(img: Image.Image) -> Dict[str, Any]:
    """Safely extract and decode all available EXIF metadata from a PIL Image.

    Returns a dictionary of readable tag name -> sanitized value.
    """
    exif_data: Dict[str, Any] = {}
    try:
        raw_exif = img.getexif()
        if not raw_exif:
            return exif_data

        # Primary EXIF tags
        for tag_id, value in raw_exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, f"Tag_{tag_id}")
            exif_data[tag_name] = _sanitize_exif_val(value)

        # Extended IFDs (EXIF IFD, GPS IFD) if available in modern Pillow
        if hasattr(ExifTags, "IFD"):
            try:
                exif_ifd = raw_exif.get_ifd(ExifTags.IFD.Exif)
                for tag_id, value in exif_ifd.items():
                    tag_name = ExifTags.TAGS.get(tag_id, f"ExifTag_{tag_id}")
                    if tag_name not in exif_data:
                        exif_data[tag_name] = _sanitize_exif_val(value)
            except Exception:
                pass

            try:
                gps_ifd = raw_exif.get_ifd(ExifTags.IFD.GPSInfo)
                if gps_ifd:
                    gps_data = {}
                    for tag_id, value in gps_ifd.items():
                        tag_name = ExifTags.GPSTAGS.get(tag_id, f"GPSTag_{tag_id}")
                        gps_data[tag_name] = _sanitize_exif_val(value)
                    exif_data["GPSInfo"] = gps_data
            except Exception:
                pass

    except Exception:
        # Silently catch corrupted EXIF blocks without breaking image extraction
        pass

    return exif_data


def summarize_exif(exif_data: Dict[str, Any]) -> str:
    """Generate a compact one-line summary from extracted EXIF data."""
    if not exif_data:
        return "None"

    parts = []
    # Camera Make and Model
    make = exif_data.get("Make", "").strip()
    model = exif_data.get("Model", "").strip()
    if make or model:
        camera = f"{make} {model}".strip()
        parts.append(f"Camera: {camera}")

    # Timestamp
    dt = (
        exif_data.get("DateTimeOriginal")
        or exif_data.get("DateTimeDigitized")
        or exif_data.get("DateTime")
    )
    if dt:
        parts.append(f"Date: {dt}")

    # Software / Tool
    software = exif_data.get("Software")
    if software:
        parts.append(f"Software: {software.strip()}")

    # Orientation
    orientation = exif_data.get("Orientation")
    if orientation is not None:
        parts.append(f"Orientation: {orientation}")

    if not parts:
        # Fallback to key count if specific fields aren't found
        return f"{len(exif_data)} EXIF tag(s) present"

    return " | ".join(parts)


@dataclass
class ImageMetadata:
    """Represents metadata extracted from a single image file."""

    filename: str
    filepath: str
    is_valid: bool
    error_message: Optional[str] = None
    file_size_bytes: int = 0
    file_size_human: str = "0 B"
    format: Optional[str] = None
    mode: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    aspect_ratio: Optional[float] = None
    megapixels: Optional[float] = None
    exif_data: Dict[str, Any] = field(default_factory=dict)
    exif_summary: str = "None"
    sha256: Optional[str] = None
    last_modified: Optional[str] = None

    def to_dict(self, include_raw_exif: bool = False) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        data = asdict(self)
        if not include_raw_exif:
            data.pop("exif_data", None)
        return data


def load_image(filepath: Union[str, Path], to_rgb: bool = False) -> Optional[Image.Image]:
    """Safely open and return a PIL Image, or None if invalid/corrupt.

    Note: The returned image is loaded in memory and uncoupled from open file descriptors.
    """
    path = Path(filepath)
    if not path.is_file():
        return None
    try:
        with Image.open(path) as img:
            img.load()  # Verify underlying data stream
            loaded = img.copy()
            if to_rgb and loaded.mode != "RGB":
                loaded = loaded.convert("RGB")
            return loaded
    except Exception:
        return None


def extract_metadata(filepath: Union[str, Path]) -> ImageMetadata:
    """Extract metadata and validate an image file.

    Handles non-existent files, corrupt data, and unsupported formats gracefully.
    """
    path = Path(filepath)
    filename = path.name

    if not path.exists():
        return ImageMetadata(
            filename=filename,
            filepath=str(path.resolve()),
            is_valid=False,
            error_message="File does not exist",
        )

    if not path.is_file():
        return ImageMetadata(
            filename=filename,
            filepath=str(path.resolve()),
            is_valid=False,
            error_message="Path is a directory or special file, not a regular file",
        )

    try:
        stat = path.stat()
        file_size_bytes = stat.st_size
        file_size_human = format_bytes(file_size_bytes)
        mod_time = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    except Exception as err:
        return ImageMetadata(
            filename=filename,
            filepath=str(path.resolve()),
            is_valid=False,
            error_message=f"Failed to read file stats: {err}",
        )

    try:
        # Check if file is completely empty (0 bytes)
        if file_size_bytes == 0:
            return ImageMetadata(
                filename=filename,
                filepath=str(path.resolve()),
                is_valid=False,
                error_message="Empty file (0 bytes)",
                file_size_bytes=0,
                file_size_human="0 B",
                last_modified=mod_time,
            )

        with Image.open(path) as img:
            img.verify()  # Validate headers and integrity

        # Reopen to read image attributes (verify() resets file pointer)
        with Image.open(path) as img:
            width, height = img.size
            img_format = img.format or path.suffix.lstrip(".").upper()
            img_mode = img.mode
            exif_dict = extract_exif(img)

        aspect_ratio = round(width / height, 2) if height > 0 else None
        megapixels = round((width * height) / 1_000_000.0, 2)
        exif_summary = summarize_exif(exif_dict)
        sha256 = calculate_sha256(path)

        return ImageMetadata(
            filename=filename,
            filepath=str(path.resolve()),
            is_valid=True,
            file_size_bytes=file_size_bytes,
            file_size_human=file_size_human,
            format=img_format,
            mode=img_mode,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            megapixels=megapixels,
            exif_data=exif_dict,
            exif_summary=exif_summary,
            sha256=sha256,
            last_modified=mod_time,
        )

    except UnidentifiedImageError:
        return ImageMetadata(
            filename=filename,
            filepath=str(path.resolve()),
            is_valid=False,
            error_message="Unsupported format or unreadable image header",
            file_size_bytes=file_size_bytes,
            file_size_human=file_size_human,
            last_modified=mod_time,
        )
    except Exception as err:
        return ImageMetadata(
            filename=filename,
            filepath=str(path.resolve()),
            is_valid=False,
            error_message=f"Corrupt or invalid image: {err}",
            file_size_bytes=file_size_bytes,
            file_size_human=file_size_human,
            last_modified=mod_time,
        )


def scan_directory(
    directory_path: Union[str, Path],
    recursive: bool = False,
    supported_extensions_only: bool = False,
) -> List[ImageMetadata]:
    """Scan a directory and extract metadata for all files found.

    Args:
        directory_path: Path to directory to scan.
        recursive: If True, recursively scan subdirectories.
        supported_extensions_only: If True, only evaluate files with typical image extensions.

    Returns:
        List of ImageMetadata objects (both valid and invalid).
    """
    target_dir = Path(directory_path)
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory_path}")

    files: List[Path] = []
    if recursive:
        for root, _, filenames in os.walk(target_dir):
            for name in filenames:
                files.append(Path(root) / name)
    else:
        for entry in target_dir.iterdir():
            if entry.is_file():
                files.append(entry)

    # Sort files alphabetically for deterministic output
    files.sort(key=lambda p: p.name.lower())

    results: List[ImageMetadata] = []
    for f in files:
        if supported_extensions_only and f.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        results.append(extract_metadata(f))

    return results


def metadata_to_dataframe(
    records: Sequence[ImageMetadata],
    include_full_path: bool = False,
    include_sha256: bool = False,
) -> pd.DataFrame:
    """Convert a sequence of ImageMetadata records into a pandas DataFrame.

    Formats fields cleanly for display in tables and reports.
    """
    rows = []
    for rec in records:
        row = {
            "Filename": rec.filename,
            "Valid": "✓ Yes" if rec.is_valid else "✗ No",
            "Dimensions": f"{rec.width}x{rec.height}" if rec.is_valid else "-",
            "Format": rec.format if rec.is_valid else "-",
            "Mode": rec.mode if rec.is_valid else "-",
            "Size": rec.file_size_human,
            "Megapixels": f"{rec.megapixels:.2f} MP" if rec.megapixels is not None else "-",
            "Aspect Ratio": str(rec.aspect_ratio) if rec.aspect_ratio is not None else "-",
            "EXIF Summary": rec.exif_summary if rec.is_valid else "-",
            "Status / Error": rec.error_message if not rec.is_valid else "OK",
        }
        if include_full_path:
            row["Filepath"] = rec.filepath
        if include_sha256:
            row["SHA256"] = rec.sha256[:12] + "..." if rec.sha256 else "-"
        rows.append(row)

    return pd.DataFrame(rows)


def display_metadata_table(
    records_or_df: Union[Sequence[ImageMetadata], pd.DataFrame],
    table_format: str = "rounded_outline",
) -> str:
    """Format metadata into a printable ASCII/Unicode table string using tabulate.

    If tabulate is unavailable, falls back to pandas DataFrame string representation.
    """
    if isinstance(records_or_df, pd.DataFrame):
        df = records_or_df
    else:
        df = metadata_to_dataframe(records_or_df)

    if df.empty:
        return "No image files found or analyzed."

    try:
        from tabulate import tabulate

        return tabulate(df, headers="keys", tablefmt=table_format, showindex=False)
    except ImportError:
        return df.to_string(index=False)


def main() -> None:
    """Command-line entry point to inspect images and display metadata table."""
    parser = argparse.ArgumentParser(
        description="Inspect image collection and display metadata table."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more image files or directories to scan.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively scan directories.",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        type=str,
        default=None,
        help="Path to export metadata table as CSV.",
    )
    parser.add_argument(
        "--table-format",
        type=str,
        default="rounded_outline",
        help="Table formatting style (rounded_outline, grid, github, simple, etc.).",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Include full filepaths in the table output.",
    )

    args = parser.parse_args()

    collected_records: List[ImageMetadata] = []
    for p_str in args.paths:
        p = Path(p_str)
        if p.is_dir():
            collected_records.extend(scan_directory(p, recursive=args.recursive))
        elif p.is_file():
            collected_records.append(extract_metadata(p))
        else:
            print(f"Warning: '{p_str}' does not exist.")

    if not collected_records:
        print("No files analyzed.")
        return

    df = metadata_to_dataframe(
        collected_records,
        include_full_path=args.show_paths,
        include_sha256=False,
    )

    table_str = display_metadata_table(df, table_format=args.table_format)
    print("\n" + table_str + "\n")

    valid_count = sum(1 for r in collected_records if r.is_valid)
    invalid_count = len(collected_records) - valid_count
    print(
        f"Analyzed {len(collected_records)} file(s): "
        f"{valid_count} valid image(s), {invalid_count} invalid/corrupt."
    )

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Export comprehensive CSV with full details
        export_df = pd.DataFrame([r.to_dict(include_raw_exif=False) for r in collected_records])
        export_df.to_csv(out_path, index=False)
        print(f"Saved metadata to: {out_path}")


if __name__ == "__main__":
    main()
