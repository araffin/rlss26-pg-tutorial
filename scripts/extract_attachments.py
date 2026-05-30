#!/usr/bin/env python3
"""
Extract notebook attachments to image files inside images/.

For each attachment found in a markdown cell, a meaningful filename is derived
using the following priority:
  1. The `alt` attribute of the <img> tag that references the attachment.
  2. A slug built from the nearest markdown heading in the same cell.
  3. A fallback of "<notebook_stem>_cell<N>_img<M>".

If two attachments would resolve to the same filename, a numeric suffix is added.
"""

import base64
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIME_TO_EXT: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


def _mime_to_ext(mime_type: str) -> str:
    """Return the file extension (with leading dot) for a MIME type."""
    return _MIME_TO_EXT.get(mime_type.lower(), ".png")


def slugify(text: str) -> str:
    """Turn arbitrary text into a safe, lowercase filename stem."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_") or "image"


def find_alt(source: str, uuid_stem: str) -> str | None:
    """
    Return the alt text of the <img> tag whose src references *uuid_stem*,
    or None if not found.

    Handles both forms:
      <img src="attachment:UUID.ext" alt="...">
      ![alt text](attachment:UUID.ext)
    """
    # HTML img tag
    pattern_html = (
        r'<img\b[^>]*\bsrc=["\']attachment:'
        + re.escape(uuid_stem)
        + r'[^"\']*["\'][^>]*\balt=["\']([^"\']+)["\']'
        r'|'
        r'<img\b[^>]*\balt=["\']([^"\']+)["\'][^>]*\bsrc=["\']attachment:'
        + re.escape(uuid_stem)
        + r'[^"\']*["\']'
    )
    m = re.search(pattern_html, source, re.IGNORECASE)
    if m:
        alt = m.group(1) or m.group(2)
        return alt.strip() if alt else None

    # Markdown-style ![alt](attachment:...)
    pattern_md = (
        r'!\[([^\]]*)\]\(attachment:'
        + re.escape(uuid_stem)
        + r'[^)]*\)'
    )
    m = re.search(pattern_md, source)
    if m:
        return m.group(1).strip() or None

    return None


def nearest_heading(source: str) -> str | None:
    """Return the text of the last markdown heading before any attachment."""
    headings = re.findall(r"^#{1,6}\s+(.+)", source, re.MULTILINE)
    return headings[-1].strip() if headings else None


def derive_name(source: str, uuid_stem: str, ext: str,
                fallback: str, used: set[str]) -> str:
    """
    Build a unique, meaningful filename (with extension) for one attachment.
    """
    # 1. alt attribute
    alt = find_alt(source, uuid_stem)
    if alt:
        stem = slugify(Path(alt).stem)  # strip extension if alt already has one
    else:
        # 2. nearest heading
        heading = nearest_heading(source)
        stem = slugify(heading) if heading else slugify(fallback)

    candidate = f"{stem}{ext}"
    # Ensure uniqueness
    if candidate not in used:
        return candidate
    i = 2
    while f"{stem}_{i}{ext}" in used:
        i += 1
    return f"{stem}_{i}{ext}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_notebook(nb_path: Path, images_dir: Path) -> dict[str, str]:
    """
    Extract all attachments from *nb_path* into *images_dir*.

    Returns a mapping  {uuid_filename: saved_filename}  for use by the
    replace script.
    """
    with nb_path.open(encoding="utf-8") as f:
        nb = json.load(f)

    mapping: dict[str, str] = {}   # uuid_filename -> saved filename
    used_names: set[str] = set()

    for cell_idx, cell in enumerate(nb.get("cells", [])):
        attachments: dict = cell.get("attachments", {})
        if not attachments:
            continue

        source: str = "".join(cell.get("source", []))

        for img_idx, (uuid_filename, mime_data) in enumerate(attachments.items()):
            uuid_stem = Path(uuid_filename).stem
            mime_type, b64_data = next(iter(mime_data.items()))
            ext = _mime_to_ext(mime_type)

            fallback = f"{nb_path.stem}_cell{cell_idx}_img{img_idx}"
            name = derive_name(source, uuid_stem, ext, fallback, used_names)
            used_names.add(name)
            mapping[uuid_filename] = name

            # Decode and write
            raw = base64.b64decode(b64_data)
            dest = images_dir / name
            dest.write_bytes(raw)
            print(f"  {uuid_filename}  ->  {dest}")

    return mapping


def main(notebooks_dir: str = "notebooks", images_dir: str = "images") -> None:
    nb_dir = Path(notebooks_dir)
    img_dir = Path(images_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    notebooks = sorted(nb_dir.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {nb_dir}", file=sys.stderr)
        sys.exit(1)

    all_mappings: dict[str, dict[str, str]] = {}
    for nb_path in notebooks:
        print(f"\n[{nb_path.name}]")
        mapping = extract_notebook(nb_path, img_dir)
        if mapping:
            all_mappings[str(nb_path)] = mapping
        else:
            print("  (no attachments)")

    # Persist the mapping so the replace script can reuse it
    mapping_file = img_dir / "_attachment_mapping.json"
    with mapping_file.open("w", encoding="utf-8") as f:
        json.dump(all_mappings, f, indent=2)
    print(f"\nMapping saved to {mapping_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", default="notebooks",
                        help="Directory containing .ipynb files (default: notebooks)")
    parser.add_argument("--images", default="images",
                        help="Output directory for images (default: images)")
    args = parser.parse_args()
    main(args.notebooks, args.images)
