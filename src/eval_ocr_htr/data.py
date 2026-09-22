"""Data layout accessors: documents, ground truth, manifest.

Layout (all under data/):
    documents/<doc_id>.pdf
    ground_truth/<doc_id>/pNN.txt        reference text per page (optional)
    ground_truth/<doc_id>/pNN.xml        PAGE XML per page (required for line mode)
    manifest.csv                         optional; doc_id,category
    outputs/                             API response cache + generated reports

There is deliberately no mapping file: the ground truth is considered to be
already associated page by page by the presence of pNN.txt / pNN.xml files.
"""

import csv
import os

from eval_ocr_htr.albert import (
    DOCUMENTS_DIR, GROUND_TRUTH_DIR, MANIFEST_PATH, OUTPUTS_DIR,
)


def list_documents():
    """Sorted list of doc_ids (PDF stems) present under data/documents/."""
    if not os.path.isdir(DOCUMENTS_DIR):
        return []
    return sorted(
        f[:-4] for f in os.listdir(DOCUMENTS_DIR)
        if f.endswith(".pdf")
    )


def pdf_path(doc_id):
    return os.path.join(DOCUMENTS_DIR, f"{doc_id}.pdf")


def exists(doc_id):
    return os.path.exists(pdf_path(doc_id))


def _gt_dir(doc_id):
    return os.path.join(GROUND_TRUTH_DIR, doc_id)


def reference_page(doc_id, page):
    """Optional reference text for a page (pNN.txt), or None."""
    path = os.path.join(_gt_dir(doc_id), f"p{page:02d}.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def page_xml_path(doc_id, page):
    """PAGE XML file for a page (required for line mode)."""
    return os.path.join(_gt_dir(doc_id), f"p{page:02d}.xml")


def has_ground_truth(doc_id):
    """A document has page ground truth if at least one pNN.txt exists."""
    d = _gt_dir(doc_id)
    if not os.path.isdir(d):
        return False
    return any(f.startswith("p") and f.endswith(".txt") for f in os.listdir(d))


def reference_pages(doc_id):
    """Ordered list of page indices that have a pNN.txt reference file."""
    d = _gt_dir(doc_id)
    if not os.path.isdir(d):
        return []
    pages = []
    i = 0
    while os.path.exists(os.path.join(d, f"p{i:02d}.txt")):
        pages.append(i)
        i += 1
    return pages


def xml_pages(doc_id):
    """Ordered list of page indices that have a pNN.xml PAGE XML file."""
    d = _gt_dir(doc_id)
    if not os.path.isdir(d):
        return []
    pages = []
    i = 0
    while os.path.exists(os.path.join(d, f"p{i:02d}.xml")):
        pages.append(i)
        i += 1
    return pages


def manifest_categories():
    """{doc_id: category} from data/manifest.csv, if present. Column order:
    doc_id,category. Any other rows are ignored."""
    if not os.path.exists(MANIFEST_PATH):
        return {}
    result = {}
    with open(MANIFEST_PATH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            did = (row.get("doc_id") or "").strip()
            cat = (row.get("category") or "").strip()
            if did:
                result[did] = cat
    return result
