"""Whole-document (full-page) OCR mode.

Two inference paths:
  - whole-document OCR endpoint: the whole PDF in one request (Mistral OCR);
  - per-page vision model: one request per page (gemma, lightonocr).

Conflicts between the two are resolved through the on-disk raw-response cache:
per-page records are written for both paths and the report always regenerates from
the cache, never re-calling the API.
"""

import base64
import difflib
import json
import os
import time

import fitz  # pymupdf

from eval_ocr_htr import albert
from eval_ocr_htr.albert import (
    ALL_MODELS, MODEL_OCR, MODELS_PER_PAGE, RAW_DIR, clean, degenerate_signals,
    ensure_dirs, is_near_empty, log, read_raw, raw_path, request_with_retry,
    require_api_key, write_raw,
)
from eval_ocr_htr.data import exists, pdf_path, reference_pages
from eval_ocr_htr.metrics import (
    cer_with_insertions, cer_without_insertions, median, normalize, page_metrics,
)


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def page_count(doc_id: str) -> int:
    with fitz.open(pdf_path(doc_id)) as doc:
        return doc.page_count


def baseline_text(doc_id: str, page: int) -> str:
    """Embedded PDF text layer, via pymupdf get_text()."""
    with fitz.open(pdf_path(doc_id)) as doc:
        return doc[page].get_text().strip()


def render_png_b64(doc_id: str, page: int) -> str:
    with fitz.open(pdf_path(doc_id)) as doc:
        pix = doc[page].get_pixmap(dpi=albert.OCR_DPI)
        return base64.b64encode(pix.tobytes("png")).decode()


# ---------------------------------------------------------------------------
# API callers
# ---------------------------------------------------------------------------

def call_whole_document(base_url, key, doc_id):
    """POST /v1/ocr with the whole PDF as a base64 data URI. Returns the JSON dict."""
    with open(pdf_path(doc_id), "rb") as fh:
        pdf_b64 = base64.b64encode(fh.read()).decode()
    payload = {
        "model": MODEL_OCR,
        "document": {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{pdf_b64}",
        },
    }
    r = request_with_retry(base_url, "POST", "/v1/ocr", key, json=payload)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {}


def call_per_page(base_url, key, model, doc_id, page):
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": albert.HTR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{render_png_b64(doc_id, page)}"
                        },
                    },
                ],
            }
        ],
    }
    r = request_with_retry(base_url, "POST", "/v1/chat/completions", key, json=payload)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {}


def _get(d, *keys, default=None):
    return albert._get(d, *keys, default=default)


# ---------------------------------------------------------------------------
# Cost / impact extraction
# ---------------------------------------------------------------------------

def costs_from_whole_document(j):
    cost = _get(j, "usage", "cost", default=0.0) or 0.0
    kWh = _get(j, "usage", "impacts", "kWh", default=0.0) or 0.0
    kgCO2 = _get(j, "usage", "impacts", "kgCO2eq", default=0.0) or 0.0
    pages_proc = _get(j, "usage_info", "pages_processed", default=0)
    return cost, kWh, kgCO2, pages_proc


def costs_from_per_page(j):
    cost = _get(j, "usage", "cost", default=0.0) or 0.0
    kWh = _get(j, "usage", "impacts", "kWh", default=0.0) or 0.0
    kgCO2 = _get(j, "usage", "impacts", "kgCO2eq", default=0.0) or 0.0
    if not kWh:
        kWh = _get(j, "usage", "carbon", "kWh", "min", default=0.0) or 0.0
    if not kgCO2:
        kgCO2 = _get(j, "usage", "carbon", "kgCO2eq", "min", default=0.0) or 0.0
    return cost, kWh, kgCO2


# ---------------------------------------------------------------------------
# Per-model processing (with cache resume)
# ---------------------------------------------------------------------------

def process_per_page(base_url, key, ctx, doc_id, model, pages):
    """Process one per-page model over the given page list. Skips cached pages."""
    for page in pages:
        cached = read_raw(model, page)
        if cached is not None:
            log(f"cache {model} page {page} (API skip), status={cached['status_code']}")
            ctx["cost_cumul"] += float(cached.get("cost", 0.0))
            continue
        t0 = time.time()
        status, j = call_per_page(base_url, key, model, doc_id, page)
        dur = time.time() - t0
        if status == 200:
            content = _get(j, "choices", 0, "message", "content", default="") or ""
            cost, kWh, kgCO2 = costs_from_per_page(j)
        else:
            content = ""
            cost = kWh = kgCO2 = 0.0
        rec = {
            "model": model,
            "page": page,
            "status_code": status,
            "duration_s": round(dur, 2),
            "cost": cost,
            "kWh": kWh,
            "kgCO2eq": kgCO2,
            "output": content,
            "error": None if status == 200 else _get(j, "error", default=None),
        }
        write_raw(rec, model, page)
        ctx["cost_cumul"] += cost
        log(
            f"page {page} model {model} status {status} duration {dur:.1f}s "
            f"cumulative cost {ctx['cost_cumul']:.6f} EUR"
        )


def process_whole_document(base_url, key, ctx, doc_id, pages):
    """Whole-PDF OCR; extract one per-page cache record. Skips if all pages cached."""
    model = MODEL_OCR
    missing = [p for p in pages if read_raw(model, p) is None]
    if not missing:
        for page in pages:
            cached = read_raw(model, page)
            ctx["cost_cumul"] += float(cached.get("cost", 0.0))
        log(f"cache {model} document {doc_id} (whole already processed, API skip)")
        return
    t0 = time.time()
    status, j = call_whole_document(base_url, key, doc_id)
    dur = time.time() - t0
    cost_total, kWh_total, kgCO2_total, pages_proc = costs_from_whole_document(j)
    if status == 200:
        pages_list = j.get("pages", [])
        n = max(len(pages_list), 1)
    else:
        pages_list = []
        n = 1
    # Attribute a fair share of cost/impacts AND latency per page. The whole-document
    # call always covers the whole PDF in one request, so the per-page latency is the
    # call latency divided by the number of pages actually processed; the divisor is
    # recorded as pages_covered so the cache is self-describing.
    pages_proc_val = pages_proc if pages_proc else n
    per_cost = cost_total / pages_proc_val
    per_kwh = kWh_total / pages_proc_val
    per_kg = kgCO2_total / pages_proc_val
    per_dur = dur / pages_proc_val
    by_idx = {}
    for p in pages_list:
        idx = p.get("page_index") or p.get("index")
        md = p.get("markdown", "")
        if idx is None:
            continue
        by_idx[int(idx)] = md
    for page in pages:
        cached = read_raw(model, page)
        if cached is not None:
            ctx["cost_cumul"] += float(cached.get("cost", 0.0))
            continue
        md = by_idx.get(page, "")
        rec = {
            "model": model,
            "page": page,
            "status_code": status,
            "duration_s": round(per_dur, 2),
            "pages_covered": pages_proc_val,
            "cost": per_cost,
            "kWh": per_kwh,
            "kgCO2eq": per_kg,
            "output": md,
            "error": None if status == 200 else _get(j, "error", default=None),
        }
        write_raw(rec, model, page)
        ctx["cost_cumul"] += per_cost
        log(
            f"page {page} model {model} status {status} duration {dur:.1f}s "
            f"cumulative cost {ctx['cost_cumul']:.6f} EUR"
        )


# ---------------------------------------------------------------------------
# Report helpers shared with lines mode
# ---------------------------------------------------------------------------

def page_latency(rec: dict, doc_id: str) -> float:
    """Per-page latency of a cached record. The whole-document model stores the
    per-page latency in `duration_s` (call latency / number of pages covered);
    per-page vision records are already per-page. Every record written by this
    code includes `pages_covered`."""
    return rec.get("duration_s") or 0.0


def cer_distance(ref: str, hyp: str) -> float:
    """Normalized (by ref length) edit distance via SequenceMatcher."""
    if not ref:
        return 0.0
    ratio = difflib.SequenceMatcher(None, ref, hyp).ratio()
    ed = (1.0 - ratio) * (len(ref) + len(hyp)) / 2.0
    return ed / len(ref)


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def _fence(text: str) -> str:
    """Return a code-fence marker longer than the longest run of backticks in
    `text` (min 3). Guarantees the transcription cannot close its own block."""
    longest = 0
    run = 0
    for ch in text:
        if ch == "`":
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    return "`" * max(longest + 1, 3)


# ---------------------------------------------------------------------------
# Full-page metrics (per document)
# ---------------------------------------------------------------------------

def compute_document_metrics(doc_id, pages=None):
    """Compute per-page per-model metrics for a document from its cache and gt.

    Returns (metrics, summary) where metrics maps (model, page)->page_metrics
    and summary maps model -> aggregated dict."""
    if pages is None:
        pages = reference_pages(doc_id)
    page_metrics_map = {}
    for model in ALL_MODELS:
        for page in pages:
            rec = read_raw(model, page)
            if rec is None or rec.get("status_code") != 200:
                continue
            out = rec.get("output", "") or ""
            ref = albert_reference(doc_id, page)
            if ref is None:
                continue
            page_metrics_map[(model, page)] = page_metrics(ref, out)
    summary = {}
    for model in ALL_MODELS:
        rows = {p: page_metrics_map[(model, p)]
                for p in pages if (model, p) in page_metrics_map}
        if not rows:
            continue
        S_t = sum(r["S_norm"] for r in rows.values())
        D_t = sum(r["D_norm"] for r in rows.values())
        I_t = sum(r["I_norm"] for r in rows.values())
        N_t = sum(r["N_norm"] for r in rows.values())
        S_s_t = sum(r["S_strict"] for r in rows.values())
        D_s_t = sum(r["D_strict"] for r in rows.values())
        I_s_t = sum(r["I_strict"] for r in rows.values())
        N_s_t = sum(r["N_strict"] for r in rows.values())
        summary[model] = {
            "pages": list(rows.keys()),
            "N_total": N_t,
            "S": S_t, "D": D_t, "I": I_t,
            "cer_ref_norm": cer_without_insertions(S_t, D_t, N_t),
            "cer_raw_norm": cer_with_insertions(S_t, D_t, I_t, N_t),
            "surplus_norm": I_t / N_t if N_t else None,
            "cer_ref_strict": cer_without_insertions(S_s_t, D_s_t, N_s_t),
            "cer_raw_strict": cer_with_insertions(S_s_t, D_s_t, I_s_t, N_s_t),
            "surplus_strict": I_s_t / N_s_t if N_s_t else None,
            "word_recall": sum(r["word_recall"] for r in rows.values()) / len(rows),
            "cer_ref_norm_median": _median([r["cer_ref_norm"] for r in rows.values()]),
            "cer_ref_strict_median": _median([r["cer_ref_strict"] for r in rows.values()]),
            "cer_raw_norm_median": _median([r["cer_raw_norm"] for r in rows.values()]),
            "surplus_norm_median": _median([r["surplus_norm"] for r in rows.values()]),
        }
    return page_metrics_map, summary


def albert_reference(doc_id, page):
    from eval_ocr_htr.data import reference_page
    return reference_page(doc_id, page)


def _median(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def cer_document_section(doc_id: str) -> str:
    """Markdown 'Transcription quality (CER)' section for a document with ground
    truth. Computes from local cache; no API call. No reference/output text is
    emitted — only ids, counts and metrics."""
    _, summary = compute_document_metrics(doc_id)
    if not summary:
        return ""
    lines = [f"## Transcription quality (CER) — {doc_id}", ""]
    ref_pages = summary[next(iter(summary))]["pages"]
    n_pages = len(ref_pages)
    lines.append(
        "**Caveat:** the reference is the manuscript only; per-page vision models "
        f"also restitute the pre-printed form text, which the surplus penalizes. "
        "Short sample — treat figures as indicative, not conclusive."
    )
    lines.append("")
    lines.append(
        "| model | CER ref. norm. | CER ref. strict | CER raw | surplus | "
        "word recall | pages covered |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for model in ALL_MODELS:
        if model not in summary:
            continue
        s = summary[model]
        lines.append(
            f"| {model} | {_fmt(s['cer_ref_norm'])} | {_fmt(s['cer_ref_strict'])} | "
            f"{_fmt(s['cer_raw_norm'])} | {_fmt(s['surplus_norm'])} | "
            f"{_fmt(s['word_recall'])} | {len(s['pages'])}/{n_pages} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_document(base_url, key, doc_id, interval, ctx):
    if not exists(doc_id):
        import sys
        sys.exit(f"PDF not found: {pdf_path(doc_id)}")
    pages = [p for p in interval if p < page_count(doc_id)]
    log(f"run — document {doc_id}, pages {pages[0]+1 if pages else '-'}.."
        f"{pages[-1]+1 if pages else '-'} ({len(pages)} pages)")
    models = albert.available_models(base_url, key)
    for model in models:
        if model == MODEL_OCR:
            process_whole_document(base_url, key, ctx, doc_id, pages)
        else:
            process_per_page(base_url, key, ctx, doc_id, model, pages)


def run_probe(base_url, key, doc_id, interval, ctx):
    pages = list(interval) or [0]
    models = albert.available_models(base_url, key)
    log(f"PROBE — document {doc_id}, pages {[p + 1 for p in pages]}")
    print()
    header = f"{'model':<20}{'HTTP':<6}{'length':<10}{'dur(s)':<10}{'cost(EUR)':<14}"
    print(header)
    print("-" * len(header))
    for model in models:
        if model == MODEL_OCR:
            process_whole_document(base_url, key, ctx, doc_id, pages)
        else:
            process_per_page(base_url, key, ctx, doc_id, model, pages)
        rec = read_raw(model, pages[0])
        out = (rec or {}).get("output", "")
        print(
            f"{model:<20}{rec['status_code'] if rec else '?':<6}"
            f"{len(out) if rec else 0:<10}"
            f"{(rec or {}).get('duration_s', 0):<10.1f}"
            f"{(rec or {}).get('cost', 0):<14.6f}"
        )
    print()
    log(f"PROBE done. Cumulative cost of run: {ctx['cost_cumul']:.6f} EUR.")
