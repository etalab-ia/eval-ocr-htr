"""Report generation for the full-page (whole-document) mode.

Builds the comparison and synthesis Markdown from the on-disk cache only — never
calls the API. Also handles linear extrapolation of cost / carbon / duration to a
target page count, and the per-document CER sections.

Privacy by design: no transcribed or reference text is printed or written — only
ids, page numbers, lengths, counts and metrics.
"""

import json
import os
import statistics

import fitz  # pymupdf

from eval_ocr_htr.albert import (
    ALL_MODELS, OUTPUTS_DIR, clean, degenerate_signals,
    read_raw, is_near_empty,
)
from eval_ocr_htr.data import (
    exists, manifest_categories, pdf_path, reference_pages,
)
from eval_ocr_htr.metrics import median
from eval_ocr_htr.page import (
    _fence, cer_document_section, cer_distance, compute_document_metrics,
    page_count, page_latency,
)


# ---------------------------------------------------------------------------
# Compare report (per document, verbatim sections are fenced, not shown in console)
# ---------------------------------------------------------------------------

def build_compare(doc_id: str, pages) -> str:
    lines = [f"# OCR/HTR comparison — {doc_id}", ""]
    with fitz.open(pdf_path(doc_id)) as doc:
        for page in pages:
            if page >= doc.page_count:
                break
            lines.append(f"## Page {page + 1}")
            lines.append("")
            for model in ALL_MODELS:
                rec = read_raw(model, page)
                lines.append(f"### {model}")
                if rec is None:
                    lines.append("(not processed / absent from cache)")
                    lines.append("")
                    continue
                if rec.get("status_code") != 200:
                    lines.append(
                        f"_(HTTP failure {rec['status_code']}: {rec.get('error')})_"
                    )
                    lines.append("")
                    continue
                out = clean(rec.get("output", ""))
                f = _fence(out)
                lines.append(f)
                sigs = degenerate_signals(out)
                if sigs:
                    lines.append(
                        f"[page unusable: {len(out)} chars, "
                        f"signals triggered: {', '.join(sigs)}]"
                    )
                else:
                    lines.append(out if out else "(empty)")
                lines.append(f)
                lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregated model stats
# ---------------------------------------------------------------------------

def model_stats(model: str, doc_ids: list, gt_pages: dict):
    """Aggregate stats across cached raw records for a model."""
    n_ok = 0
    n_err = 0
    n_degen = 0
    signals = {"repetition": 0, "absurd_number_run": 0, "illegible_dominance": 0}
    durs = []
    costs = []
    kgs = []
    lens = []
    cers = []
    for doc_id in doc_ids:
        nb = page_count(doc_id)
        for page in range(nb):
            rec = read_raw(model, page)
            if rec is None:
                continue
            if rec.get("status_code") != 200:
                n_err += 1
                continue
            out = clean(rec.get("output", "") or "")
            sigs = degenerate_signals(out)
            if sigs:
                n_degen += 1
                for s in sigs:
                    signals[s] += 1
                costs.append(rec.get("cost", 0.0))
                kgs.append(rec.get("kgCO2eq", 0.0))
                continue
            n_ok += 1
            durs.append(page_latency(rec, doc_id))
            costs.append(rec.get("cost", 0.0))
            kgs.append(rec.get("kgCO2eq", 0.0))
            lens.append(len(out))
            ref = gt_pages.get((doc_id, page))
            if ref is not None:
                cers.append(cer_distance(clean(ref), out))
    tot_cost = sum(costs)
    tot_kg = sum(kgs)
    denom = n_ok + n_degen
    return {
        "model": model,
        "n_ok": n_ok,
        "n_err": n_err,
        "n_degen": n_degen,
        "signals": signals,
        "lat_med": median(durs),
        "cost_total": tot_cost,
        "cost_page": (tot_cost / denom) if denom else 0.0,
        "kg_total": tot_kg,
        "kg_page": (tot_kg / denom) if denom else 0.0,
        "len_med": median(lens),
        "cer_med": median(cers) if cers else None,
    }


def gt_pages_map(doc_ids) -> dict:
    """{(doc_id, page): reference text} from data/ground_truth/<doc_id>/pNN.txt."""
    mapping = {}
    for doc_id in doc_ids:
        for page in reference_pages(doc_id):
            from eval_ocr_htr.data import reference_page
            mapping[(doc_id, page)] = reference_page(doc_id, page)
    return mapping


def pages_useful(doc_ids: list) -> set:
    """Set of (doc_id, page) that are 'useful': a page is useful when no model
    produces a near-empty output for it (blank scans are excluded, without being
    counted as failures and without being useful)."""
    useful = set()
    for doc_id in doc_ids:
        nb = page_count(doc_id)
        for page in range(nb):
            produced = False
            all_contentful = True
            for model in ALL_MODELS:
                rec = read_raw(model, page)
                if rec is None or rec.get("status_code") != 200:
                    continue
                produced = True
                if is_near_empty(rec.get("output", "")):
                    all_contentful = False
                    break
            if produced and all_contentful:
                useful.add((doc_id, page))
    return useful


def unusable_by_doc(model, doc_ids, useful) -> dict:
    """{doc_id: n} unusable pages by document, among useful pages only."""
    counts = {}
    for doc_id in doc_ids:
        nb = page_count(doc_id)
        n = 0
        for page in range(nb):
            if (doc_id, page) not in useful:
                continue
            rec = read_raw(model, page)
            if rec is None or rec.get("status_code") != 200:
                continue
            if degenerate_signals(clean(rec.get("output", "") or "")):
                n += 1
        counts[doc_id] = n
    return counts


def docs_with_ground_truth(doc_ids: list) -> list:
    """Documents that have an associated page reference (pNN.txt) under gt/."""
    return [d for d in doc_ids if reference_pages(d)]


# ---------------------------------------------------------------------------
# Extrapolation / report rendering
# ---------------------------------------------------------------------------

def _extrap_report(s, extrapolate_pages, per_doc_lines=()):
    """Markdown block for linear extrapolation. Used by the synthesis."""
    n = s["n_ok"] + s["n_degen"]
    L = []
    if n:
        L.append(
            f"**Linear extrapolation to {extrapolate_pages:,} pages** "
            f"(from the actually-processed sample of {n} page(s), "
            "simple multiplication — not a measurement on the platform):"
        )
        L.append("")
        L.append("| Extrapolated metric | Value |")
        L.append("|---|---|")
        L.append(f"| Cost | {s['cost_page'] * extrapolate_pages:.2f} EUR |")
        L.append(f"| kgCO2eq | {s['kg_page'] * extrapolate_pages:.2f} |")
        dur_seq_days = s["lat_med"] * extrapolate_pages / 86400.0
        dur_par_days = dur_seq_days / 10.0
        L.append(f"| Processing duration, sequential | {dur_seq_days:.1f} d |")
        L.append(f"| Duration with 10 parallel workers | {dur_par_days:.1f} d |")
        L.append("")
        L.append(
            "*Note: the real parallelism is bounded by Albert's throughput limits, "
            "which are per-user and per-model.*"
        )
        L.append("")
    return "\n".join(L)


def build_summary(doc_ids: list, gt: dict, extrapolate_pages: int) -> str:
    lines = ["# OCR/HTR benchmark synthesis", ""]
    useful = pages_useful(doc_ids)
    n_useful = len(useful)
    lines.append(
        f"Corpus analysed: {len(doc_ids)} document(s), {n_useful} useful page(s)."
    )
    cats = manifest_categories()
    if cats:
        by_cat = {}
        for (d, _p) in useful:
            cat = cats.get(d, "other")
            by_cat[cat] = by_cat.get(cat, 0) + 1
        cat_str = ", ".join(f"{k}: {v}" for k, v in sorted(by_cat.items()))
        lines.append(f"Useful pages by category: {cat_str}.")
    lines.append(
        "The extrapolation below is a simple linear multiplication from the "
        "actually-processed sample (the number of processed pages is given for "
        "each model in its section) — not a measurement on the platform."
    )
    lines.append("")

    med_w, med_h, med_dpi = pdf_resolution_stats(doc_ids)
    lines.append(
        f"Source: median embedded-image resolution of processed PDFs = "
        f"{med_w:.0f}x{med_h:.0f} px, i.e. ~{med_dpi:.0f} dpi estimated for an A4 page "
        "(HTR usually wants 300 dpi)."
    )
    lines.append("")

    for model in ALL_MODELS:
        s = model_stats(model, doc_ids, gt)
        total = s["n_ok"] + s["n_degen"]
        degen_pct = (s["n_degen"] / total * 100.0) if total else 0.0
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Pages processed | {total} |")
        lines.append(f"| Failures (HTTP) | {s['n_err']} |")
        lines.append(
            f"| Unusable pages | {s['n_degen']} / {total} pages ({degen_pct:.0f} %) |"
        )
        triggered = [
            f"{s['signals'].get(moniker, 0)} {moniker}"
            for moniker in ("repetition", "absurd_number_run", "illegible_dominance")
            if s["signals"].get(moniker, 0)
        ]
        sig_parts = ", ".join(triggered) if triggered else "none"
        lines.append(f"*(breakdown by triggered signal: {sig_parts})*")
        lines.append(f"| Median latency / page | {s['lat_med']:.1f}s |")
        lines.append(f"| Total cost | {s['cost_total']:.6f} EUR |")
        lines.append(f"| Cost per page | {s['cost_page']:.6f} EUR |")
        lines.append(f"| Total kgCO2eq | {s['kg_total']:.6f} |")
        lines.append(f"| kgCO2eq per page | {s['kg_page']:.8f} |")
        lines.append(f"| Median output length | {s['len_med']:.0f} chars |")
        if s["cer_med"] is not None:
            lines.append(f"| Median CER | {s['cer_med']:.3f} |")
        lines.append("")
        per_doc = unusable_by_doc(model, doc_ids, useful)
        for doc_id in doc_ids:
            n_use = sum(1 for c, _ in useful if c == doc_id)
            n_non = per_doc.get(doc_id, 0)
            lines.append(
                f"Unusable pages per document | {doc_id}: "
                f"{n_non} / {n_use} useful pages"
            )
        lines.append("")
        lines.append(_extrap_report(s, extrapolate_pages))
        lines.append("")

    for ct in docs_with_ground_truth(doc_ids):
        section = cer_document_section(ct)
        if section:
            lines.append(section)
    return "\n".join(lines)


def pdf_resolution_stats(doc_ids: list) -> tuple:
    """Median embedded-image resolution (pixels) and estimated A4 dpi across all
    processed PDF pages, via pymupdf. dpi is estimated as pixel width / printed
    width (points/72), i.e. the density the image is placed on the page."""
    widths, heights, dpis = [], [], []
    for doc_id in doc_ids:
        try:
            with fitz.open(pdf_path(doc_id)) as doc:
                for pageobj in doc:
                    for im in pageobj.get_image_info():
                        w, h = im.get("width"), im.get("height")
                        if not w or not h:
                            continue
                        widths.append(w)
                        heights.append(h)
                        printed_w_in = (im.get("bbox") or [0, 0, 0, 0])[2] / 72.0
                        if printed_w_in > 0:
                            dpis.append(w / printed_w_in)
        except Exception:
            continue
    return (median(widths), median(heights), median(dpis))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_full_page_cer_detail(doc_id):
    """Per-page per-model CER detail file (metrics only, no text)."""
    page_det, _summary = compute_document_metrics(doc_id)
    detail = {
        "doc": doc_id,
        "pages": sorted({p for _, p in page_det}),
        "models": ALL_MODELS,
        "pages_per_model": {
            m: sorted(p for (_m, p) in page_det if _m == m) for m in ALL_MODELS
        },
        "per_page_per_model": {
            f"{_m}__p{p:02d}": page_det[(_m, p)]
            for (_m, p) in sorted(page_det, key=lambda k: (k[1], str(k[0])))
        },
    }
    path = os.path.join(OUTPUTS_DIR, f"cer_{doc_id}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(detail, fh, ensure_ascii=False, indent=2)


def write_outputs(doc_ids: list, extrapolate_pages: int):
    """Regenerate compare + synthesis from current cache. No API call."""
    gt = {}
    try:
        gt = gt_pages_map(doc_ids)
    except Exception:
        gt = {}
    for doc_id in doc_ids:
        if not exists(doc_id):
            continue
        with fitz.open(pdf_path(doc_id)) as doc:
            n = doc.page_count
        pages = range(0, n)
        with open(os.path.join(OUTPUTS_DIR, f"compare_{doc_id}.md"), "w",
                  encoding="utf-8") as fh:
            fh.write(build_compare(doc_id, pages))
    with open(os.path.join(OUTPUTS_DIR, "synthesis.md"), "w",
              encoding="utf-8") as fh:
        fh.write(build_summary(doc_ids, gt, extrapolate_pages))
    for ct in docs_with_ground_truth(doc_ids):
        write_full_page_cer_detail(ct)
