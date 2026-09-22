"""Unit tests for per-page latency / page counting with a synthetic generated
PDF. No network, no real data.
"""

from eval_ocr_htr import page
from eval_ocr_htr.albert import MODEL_OCR, MODELS_PER_PAGE
from tests.conftest import write_pdf


def test_page_count_generated_pdf(data_env):
    write_pdf(data_env, "doc-1", n_pages=3)
    assert page.page_count("doc-1") == 3


def test_page_latency_per_page_model(data_env):
    write_pdf(data_env, "doc-1", n_pages=2)
    rec = {"model": MODELS_PER_PAGE[0], "duration_s": 12.5}
    assert page.page_latency(rec, "doc-1") == 12.5


def test_page_latency_whole_document_with_pages_covered(data_env):
    write_pdf(data_env, "doc-1", n_pages=4)
    rec = {"model": MODEL_OCR, "duration_s": 8.0, "pages_covered": 4}
    assert page.page_latency(rec, "doc-1") == 8.0


def test_baseline_text_embedded_layer(data_env):
    write_pdf(data_env, "doc-1", n_pages=1, page_text="Synthetic page")
    text = page.baseline_text("doc-1", 0)
    assert "Synthetic" in text
