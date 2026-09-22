"""Shared synthetic fixtures: a generated multi-page PDF and a hand-written
PAGE XML, all produced in a per-test tmp dir. 100% synthetic, never real data.

`data_env` repoints every data-directory constant (DOCUMENTS_DIR,
GROUND_TRUTH_DIR, OUTPUTS_DIR, RAW_DIR, MANIFEST_PATH, and the *DATA_DIR used to
derive them) onto a per-test tmp dir, so tests never touch the real data/ tree.
"""

import os

import fitz  # pymupdf

import pytest

import eval_ocr_htr.albert as albert
import eval_ocr_htr.data as data
import eval_ocr_htr.page as page
import eval_ocr_htr.lines as lines
import eval_ocr_htr.report as report


@pytest.fixture
def data_env(tmp_path, monkeypatch):
    """Point all data-dir globals at tmp_path and create the dir skeleton."""
    d = tmp_path / "data"
    (d / "documents").mkdir(parents=True)
    (d / "ground_truth").mkdir()
    (d / "outputs").mkdir()

    docs = str(d / "documents")
    gt = str(d / "ground_truth")
    outputs = str(d / "outputs")
    raw = str(d / "outputs" / "raw")
    manifest = str(d / "manifest.csv")

    # Patch only the attributes each module actually holds. albert carries
    # DATA_DIR/RAW_DIR and all dir globals; data re-exports the four path globals
    # (no DATA_DIR/RAW_DIR). page/lines resolve dirs through those; report binds
    # OUTPUTS_DIR directly.
    for name, val in zip(
        ("DATA_DIR", "DOCUMENTS_DIR", "GROUND_TRUTH_DIR", "MANIFEST_PATH",
         "OUTPUTS_DIR", "RAW_DIR"),
        (str(d), docs, gt, manifest, outputs, raw),
    ):
        monkeypatch.setattr(albert, name, val)
    for name, val in zip(
        ("DOCUMENTS_DIR", "GROUND_TRUTH_DIR", "MANIFEST_PATH", "OUTPUTS_DIR"),
        (docs, gt, manifest, outputs),
    ):
        monkeypatch.setattr(data, name, val)
    monkeypatch.setattr(report, "OUTPUTS_DIR", outputs)

    return {"dir": str(d), "documents": docs, "gt": gt, "outputs": outputs, "raw": raw}


def make_pdf(path, n_pages=3, page_text="Synthetic page"):
    """Generate a synthetic single-page-image PDF with n_pages pages using pymupdf.
    Each page carries a tiny embedded raster so the document behaves like a scan."""
    doc = fitz.open()
    for _ in range(n_pages):
        pageobj = doc.new_page(width=300, height=400)
        pix = fitz.Pixmap(
            fitz.csRGB, 60, 80,
            bytes(b"\xff\xff\xff") * (60 * 80), False,
        )
        pageobj.insert_image(fitz.Rect(0, 0, 300, 400), pixmap=pix)
        pageobj.insert_text((10, 390), page_text)  # synthetic text on the page
    doc.save(path)
    doc.close()


def write_pdf(data_env, doc_id="doc-1", n_pages=3, page_text="Synthetic page"):
    path = os.path.join(data_env["documents"], f"{doc_id}.pdf")
    make_pdf(path, n_pages=n_pages, page_text=page_text)
    return path


def write_reference(data_env, doc_id, page, text):
    d = os.path.join(data_env["gt"], doc_id)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"p{page:02d}.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return p


def make_page_xml(path, lines_, image_width=600.0, image_height=800.0):
    """Write a minimal hand-authored PAGE XML file with one TextRegion per line.
    `lines_` is a list of (text,) tuples; each gets a synthetic bbox and reading id.
    Returns the number of regions written."""
    ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<PcGts xmlns="{ns}">\n'
        f'  <Page imageWidth="{image_width:.1f}" imageHeight="{image_height:.1f}">\n'
    )
    body = []
    for i, (text,) in enumerate(lines_):
        x0, y0, x1, y1 = 20.0, 20.0 + 40 * i, 580.0, 40.0 + 40 * i
        body.append(
            f'    <TextRegion id="region-{i}">\n'
            f'      <Coords points="{x0:.1f},{y0:.1f} {x1:.1f},{y0:.1f} '
            f'{x1:.1f},{y1:.1f} {x0:.1f},{y1:.1f}"/>\n'
            f'      <TextEquiv><Unicode>{_esc(text)}</Unicode></TextEquiv>\n'
            "    </TextRegion>\n"
        )
    footer = "  </Page>\n</PcGts>\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + "".join(body) + footer)
    return len(lines_)


def write_page_xml(data_env, doc_id, page, lines_):
    d = os.path.join(data_env["gt"], doc_id)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"p{page:02d}.xml")
    make_page_xml(p, lines_)
    return p


def _esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
