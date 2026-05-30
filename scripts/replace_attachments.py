#!/usr/bin/env python3
"""
Replace attachment references in notebooks with links to extracted images.

Reads the mapping produced by extract_attachments.py
(images/_attachment_mapping.json) and rewrites every notebook in-place:

  - <img src="attachment:UUID.ext" ...>  →  <img src="<base_url>/name.ext" ...>
  - ![alt](attachment:UUID.ext)          →  ![alt](<base_url>/name.ext)
  - The `attachments` key is removed from each cell once all its refs are replaced.

Run extract_attachments.py first.

The --base-url controls the prefix used for image paths.  Two GitHub styles
are supported out of the box (replace USER/REPO/BRANCH as needed):

  Relative (default when no flag given):
    images/

  Raw githubusercontent (recommended — no redirect):
    https://raw.githubusercontent.com/USER/REPO/refs/heads/BRANCH/images

  GitHub blob + ?raw=true (also works, but redirects to the above):
    https://github.com/USER/REPO/blob/BRANCH/images
    (the script automatically appends ?raw=true for this form)
"""

import json
import re
import shutil
import sys
from pathlib import Path

DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/araffin/rlss26-pg-tutorial"
    "/refs/heads/master/images"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_image_url(base_url: str, saved_name: str) -> str:
    """
    Combine *base_url* and *saved_name* into a full image URL/path.

    If base_url looks like a GitHub blob URL (contains '/blob/'), the
    '?raw=true' query parameter is appended automatically.
    """
    url = f"{base_url.rstrip('/')}/{saved_name}"
    if "/blob/" in base_url and "?raw=true" not in url:
        url += "?raw=true"
    return url


def replace_refs(source: str, uuid_filename: str, image_url: str) -> str:
    """
    Replace every reference to *uuid_filename* in *source* with *image_url*.
    Handles HTML <img> tags and markdown ![alt](...) syntax.
    """
    uuid_escaped = re.escape(uuid_filename)

    # HTML: src="attachment:UUID.ext"
    source = re.sub(
        r'(src=["\'])attachment:' + uuid_escaped + r'(["\'])',
        r'\g<1>' + image_url + r'\2',
        source,
        flags=re.IGNORECASE,
    )

    # Markdown: (attachment:UUID.ext)
    source = re.sub(
        r'\(attachment:' + uuid_escaped + r'\)',
        f'({image_url})',
        source,
    )

    return source


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def replace_notebook(nb_path: Path, mapping: dict[str, str],
                     base_url: str, backup: bool = True) -> int:
    """
    Rewrite *nb_path* in-place using *mapping* {uuid_filename: saved_name}.
    Returns the number of replacements made.
    """
    with nb_path.open(encoding="utf-8") as f:
        nb = json.load(f)

    replacements = 0

    for cell in nb.get("cells", []):
        attachments: dict = cell.get("attachments", {})
        if not attachments:
            continue

        resolved_uuids: set[str] = set()

        for uuid_filename, saved_name in mapping.items():
            if uuid_filename not in attachments:
                continue

            image_url = build_image_url(base_url, saved_name)
            new_source = [
                replace_refs(line, uuid_filename, image_url)
                for line in cell["source"]
            ]

            changed = new_source != cell["source"]
            cell["source"] = new_source
            resolved_uuids.add(uuid_filename)
            if changed:
                replacements += 1
                print(f"  replaced: {uuid_filename}  ->  {image_url}")

        # Remove resolved attachments from the cell
        for uuid_filename in resolved_uuids:
            del cell["attachments"][uuid_filename]

        # Drop the key entirely when empty
        if not cell["attachments"]:
            del cell["attachments"]

    if replacements:
        if backup:
            shutil.copy2(nb_path, nb_path.with_suffix(".ipynb.bak"))
        with nb_path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"  -> {nb_path} rewritten ({replacements} replacement(s))")

    return replacements


def main(notebooks_dir: str = "notebooks", images_dir: str = "images",
         base_url: str = DEFAULT_BASE_URL, no_backup: bool = False) -> None:
    img_dir = Path(images_dir)
    mapping_file = img_dir / "_attachment_mapping.json"

    if not mapping_file.exists():
        print(
            f"Mapping file not found: {mapping_file}\n"
            "Run extract_attachments.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with mapping_file.open(encoding="utf-8") as f:
        all_mappings: dict[str, dict[str, str]] = json.load(f)

    nb_dir = Path(notebooks_dir)
    total = 0

    for nb_path_str, mapping in all_mappings.items():
        nb_path = Path(nb_path_str)
        if not nb_path.exists():
            # Try relative to notebooks_dir in case paths shifted
            nb_path = nb_dir / nb_path.name
        if not nb_path.exists():
            print(f"  WARNING: notebook not found: {nb_path_str}", file=sys.stderr)
            continue

        print(f"\n[{nb_path.name}]")
        total += replace_notebook(nb_path, mapping, base_url,
                                  backup=not no_backup)

    if total == 0:
        print("\nNothing to replace (already done or no attachments found).")
    else:
        print(f"\nDone — {total} attachment(s) replaced across all notebooks.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebooks", default="notebooks",
                        help="Directory containing .ipynb files (default: notebooks)")
    parser.add_argument("--images", default="images",
                        help="Images directory used by extract_attachments.py (default: images)")
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, dest="base_url",
        help=(
            "Base URL (or path) prepended to each image filename. "
            "Defaults to the raw.githubusercontent.com URL for this repo. "
            "Use 'images' for relative paths."
        ),
    )
    parser.add_argument("--no-backup", action="store_true",
                        help="Do not create .ipynb.bak backup files")
    args = parser.parse_args()
    main(args.notebooks, args.images, args.base_url, args.no_backup)
