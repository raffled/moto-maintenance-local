"""
Save extracted PageImage objects to disk and build a page → file-path index.

Images are stored under:
    {images_dir}/{manual_stem}/p{page_num:04d}_{idx:02d}.jpg|png

The index returned by save_images() maps each page number to an ordered list
of file paths matching the content-stream order from parse.py.  That ordering
preserves the visual sequence of diagram callouts (A, B, 1, 2, …) within a page.
"""

from pathlib import Path

from ingestion.parse import ParsedPage


def save_images(
    pages: list[ParsedPage],
    manual_stem: str,
    images_dir: Path,
) -> dict[int, list[str]]:
    """
    Write each PageImage to disk and return {page_num: [file_path, ...]} index.
    Pages without images are omitted from the result.
    """
    out_dir = Path(images_dir) / manual_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[int, list[str]] = {}
    for page in pages:
        if not page.images:
            continue
        paths = []
        for idx, img in enumerate(page.images):
            ext = ".jpg" if img.format == "JPEG" else ".png"
            fname = f"p{page.page_num:04d}_{idx:02d}{ext}"
            fpath = out_dir / fname
            fpath.write_bytes(img.data)
            paths.append(str(fpath))
        result[page.page_num] = paths

    return result
