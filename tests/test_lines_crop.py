"""Unit tests for the PAGE XML region parsing and line cropping in line mode.
All fixtures are synthetic PDF / PAGE XML files.
"""

import os

import fitz  # pymupdf

from eval_ocr_htr import lines
from tests.conftest import write_pdf, write_page_xml


def test_region_boxes_ordering(data_env):
    p = write_page_xml(data_env, "doc-1", 0, [("Première ligne",), ("Deuxième ligne",)])
    boxes = lines.region_boxes(p)
    assert len(boxes) == 2
    assert boxes[0]["id"] == "region-0"
    assert boxes[0]["text"] == "Première ligne"
    assert boxes[0]["x0"] < boxes[0]["x1"]
    assert boxes[0]["y0"] < boxes[0]["y1"]


def test_read_document_lines_pages(data_env):
    write_pdf(data_env, "doc-1", n_pages=2)
    write_page_xml(data_env, "doc-1", 0, [("A",), ("B",)])
    write_page_xml(data_env, "doc-1", 1, [("C",)])
    rows = lines.read_document_lines("doc-1")
    assert [r["page"] for r in rows] == [0, 0, 1]
    assert [r["ordre"] for r in rows] == [0, 1, 0]


def test_crop_line_png_returns_bytes(data_env):
    write_pdf(data_env, "doc-1", n_pages=1)
    with fitz.open(os.path.join(data_env["documents"], "doc-1.pdf")) as doc:
        src = lines.embed_pixmap(doc, 0)
        box = {"x0": 20.0, "y0": 20.0, "x1": 580.0, "y1": 60.0, "iw": 600, "ih": 800}
        sx = src.width / 600.0
        sy = src.height / 800.0
        png = lines.crop_line_png(src, box, sx, sy)
        assert png is not None
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_crop_line_png_out_of_bounds_returns_none(data_env):
    write_pdf(data_env, "doc-1", n_pages=1)
    with fitz.open(os.path.join(data_env["documents"], "doc-1.pdf")) as doc:
        src = lines.embed_pixmap(doc, 0)
        # box completely outside the image -> None
        box = {"x0": 10_000.0, "y0": 10_000.0, "x1": 10_100.0, "y1": 10_100.0,
               "iw": 600, "ih": 800}
        png = lines.crop_line_png(src, box, src.width / 600.0, src.height / 800.0)
        assert png is None
