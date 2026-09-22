"""Line-by-line HTR mode on a PAGE XML segmentation.

Each TextRegion of a PAGE XML file is treated as one handwritten line; the embedded
PDF image is cropped to each region at the PDF scale, each line is transcribed
individually by the configured models, then compared to the per-line reference
carried in the same PAGE XML.

Reuses albert.py (API calls, retries/backoff, _get, clean, degenerate_signals,
HTR_PROMPT) and metrics.py (levenshtein_counts, normalize) — no reimplementation.

Privacy by design: this module never prints or writes any transcribed/reference
text — only identifiers, page/line ids, lengths, counts and metrics.
"""

import base64
import json
import os
import random
import time
import xml.etree.ElementTree as ET

import fitz  # pymupdf

from eval_ocr_htr import albert
from eval_ocr_htr.albert import (
    ALL_MODELS, HTR_PROMPT, MODEL_OCR, _get, clean, degenerate_signals,
    log, request_with_retry,
)
from eval_ocr_htr.data import page_xml_path
from eval_ocr_htr.metrics import levenshtein_counts, line_metrics, normalize
from eval_ocr_htr.page import _median

PAGE_XML_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
NS = {"p": PAGE_XML_NS}

# Line prompt: derived from HTR_PROMPT, adapted to a single handwritten line.
LINE_PROMPT = HTR_PROMPT + (
    " Il s'agit d'une unique ligne manuscrite : transcris uniquement le texte de "
    "cette ligne, sur une seule ligne de sortie, sans remise en page markdown."
)

LINE_MARGIN_FRAC = 0.10   # +10% box height top/bottom
LINE_MARGIN_PX = 5        # +5 px left/right
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 12345

# Line cache layout for one document.
def _raw_lines_dir(doc_id):
    return os.path.join(albert.RAW_DIR, "lines", doc_id)


def _crop_dir(doc_id):
    return os.path.join(albert.OUTPUTS_DIR, "lines", doc_id)


# ---------------------------------------------------------------------------
# PAGE XML / region helpers
# ---------------------------------------------------------------------------

def region_boxes(xml_path):
    """Return list of dicts in document (reading) order: id, x0,y0,x1,y1 in XML
    image space (imageWidth/Height), and text (raw Unicode, may contain the
    reference). Never emits text; only used to build the reference line file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    page = root.find(".//p:Page", NS)
    if page is None:
        return []
    iw = float(page.get("imageWidth"))
    ih = float(page.get("imageHeight"))
    regions = []
    for r in root.findall(".//p:TextRegion", NS):
        rid = r.get("id")
        coords = r.find("p:Coords", NS)
        pts = []
        if coords is not None and coords.get("points"):
            for tok in coords.get("points").split():
                x, y = tok.split(",")
                pts.append((float(x), float(y)))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        te = r.find(".//p:TextEquiv/p:Unicode", NS)
        text = (te.text if (te is not None and te.text) else "").strip()
        regions.append({
            "id": rid, "x0": min(xs) if xs else None, "y0": min(ys) if ys else None,
            "x1": max(xs) if xs else None, "y1": max(ys) if ys else None,
            "iw": iw, "ih": ih, "text": text,
        })
    return regions


def embed_pixmap(doc, page_index):
    """Full embedded image of a PDF page as a fitz.Pixmap (native resolution,
    no re-render)."""
    imgs = doc[page_index].get_images(full=True)
    xref = imgs[0][0]
    return fitz.Pixmap(doc, xref)


def crop_pixmap_rect(src, rect):
    """Copy a rectangular window of a Pixmap into a new free Pixmap (PNG-able).
    Fallback because the fitz Pixmap-clip constructor isn't available in this
    build. Copies only the samples of the ROI."""
    w = rect.width
    h = rect.height
    n = src.n
    out = bytearray(w * h * n)
    for y in range(h):
        sy = rect.y0 + y
        s0 = (sy * src.width + rect.x0) * n
        s1 = s0 + w * n
        out[y * w * n:(y + 1) * w * n] = src.samples[s0:s1]
    cs = src.colorspace if src.colorspace else fitz.csRGB
    return fitz.Pixmap(cs, w, h, bytes(out), False)


def crop_line_png(src, box, sx, sy):
    """Scale a region bbox (XML image space) to src image space, add margins,
    bound to image, and return PNG bytes. box is {x0,y0,x1,y1}."""
    ix0 = box["x0"] * sx
    iy0 = box["y0"] * sy
    ix1 = box["x1"] * sx
    iy1 = box["y1"] * sy
    box_h = max(iy1 - iy0, 1)
    m_top = box_h * LINE_MARGIN_FRAC
    m_bot = box_h * LINE_MARGIN_FRAC
    x0 = int(ix0) - LINE_MARGIN_PX
    x1 = int(ix1) + LINE_MARGIN_PX
    y0 = int(iy0) - int(m_top)
    y1 = int(iy1) + int(m_bot)
    W = src.width
    H = src.height
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    rect = fitz.IRect(x0, y0, x1, y1)
    crop = crop_pixmap_rect(src, rect)
    try:
        return crop.tobytes("png")
    finally:
        crop = None


def scale_factors(doc, page_index, box):
    """(sx, sy) mapping XML image coords -> embedded image pixels for a page."""
    pix = embed_pixmap(doc, page_index)
    iw = box["iw"]
    ih = box["ih"]
    sx = pix.width / iw if iw else 0.0
    sy = pix.height / ih if ih else 0.0
    return sx, sy


# ---------------------------------------------------------------------------
# Reference / outputs loading
# ---------------------------------------------------------------------------

def read_document_lines(doc_id):
    """List of {page, region_id, ordre, texte} rows in document order across the
    document's PAGE XML files. Reuses region boxes in XML order."""
    rows = []
    page = 0
    while os.path.exists(page_xml_path(doc_id, page)):
        for ordre, b in enumerate(region_boxes(page_xml_path(doc_id, page))):
            rows.append({
                "page": page, "region_id": b["id"], "ordre": ordre,
                "texte": b["text"],
            })
        page += 1
    return rows


def _line_png_content(doc_id, page, ordre, region_id):
    path = os.path.join(_crop_dir(doc_id), f"p{page:02d}_{ordre:02d}_{region_id}.png")
    with open(path, "rb") as fh:
        return fh.read()


def line_png_b64(doc_id, page, ordre, region_id):
    return base64.b64encode(_line_png_content(doc_id, page, ordre, region_id)).decode()


def crop_lines(doc_id, lines):
    """Crop every line to outputs/lines/<doc_id>/pNN_<ordre>_<region_id>.png.
    Keyed on (page, ordre) because a page can have multiple TextRegions sharing
    the same id — ordre disambiguates."""
    crop_dir = _crop_dir(doc_id)
    os.makedirs(crop_dir, exist_ok=True)
    from eval_ocr_htr.data import pdf_path
    doc = fitz.open(pdf_path(doc_id))
    count = 0
    for page in sorted({r["page"] for r in lines}):
        src = embed_pixmap(doc, page)
        page_lines = [r for r in lines if r["page"] == page]
        regions = region_boxes(page_xml_path(doc_id, page))
        region_by_ordre = {}
        for ordre, b in enumerate(regions):
            region_by_ordre[ordre] = b
        sx, sy = 0.0, 0.0
        if regions:
            sx, sy = scale_factors(doc, page, regions[0])
        for r in page_lines:
            b = region_by_ordre.get(r["ordre"])
            if b is None:
                continue
            png = crop_line_png(src, b, sx, sy)
            if png is None:
                log(f"crop failed p{page:02d} ordre {r['ordre']} {r['region_id']}")
                continue
            path = os.path.join(
                crop_dir, f"p{page:02d}_{r['ordre']:02d}_{r['region_id']}.png")
            with open(path, "wb") as fh:
                fh.write(png)
            count += 1
        src = None
    doc.close()
    return count


# ---------------------------------------------------------------------------
# Inference (Albert API)
# ---------------------------------------------------------------------------

def raw_line_path(doc_id, model, page, ordre):
    return os.path.join(_raw_lines_dir(doc_id), f"{model}__p{page:02d}_{ordre:02d}.json")


def call_line(base_url, key, model, doc_id, page, ordre, region_id):
    """One-line inference. The whole-document model via /v1/ocr (image_url
    document), per-page models via /v1/chat/completions. Returns (status, dict)."""
    b64 = line_png_b64(doc_id, page, ordre, region_id)
    if model == MODEL_OCR:
        payload = {
            "model": model,
            "document": {"type": "image_url",
                         "image_url": f"data:image/png;base64,{b64}"},
        }
        r = request_with_retry(base_url, "POST", "/v1/ocr", key, json=payload)
    else:
        payload = {
            "model": model, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": LINE_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
        }
        r = request_with_retry(base_url, "POST", "/v1/chat/completions", key, json=payload)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {}


def line_output(model, payload):
    """Extract output text from a one-line API payload."""
    if model == MODEL_OCR:
        pages = payload.get("pages", [])
        return "\n".join(_get(p, "markdown", default="") for p in pages)
    return _get(payload, "choices", 0, "message", "content", default="") or ""


def run_inference(base_url, key, doc_id, lines, probe=True):
    """Transcribe each line per model, with per-line cache and per-call latency.
    probe=True: 1 line x 3 models. Returns count of API calls made (0 if all cached)."""
    raw_dir = _raw_lines_dir(doc_id)
    os.makedirs(raw_dir, exist_ok=True)
    crop_dir = _crop_dir(doc_id)
    if not os.path.isdir(crop_dir):
        crop_lines(doc_id, lines)
    call_count = 0
    targets = [lines[0]] if probe else lines
    for row in targets:
        page = row["page"]
        ordre = row["ordre"]
        rid = row["region_id"]
        for model in ALL_MODELS:
            rp = raw_line_path(doc_id, model, page, ordre)
            if os.path.exists(rp):
                log(f"cache {model} p{page:02d}_{ordre:02d} (API skip)")
                continue
            t0 = time.time()
            status, j = call_line(base_url, key, model, doc_id, page, ordre, rid)
            dur = time.time() - t0
            call_count += 1
            out = line_output(model, j) if status == 200 else ""
            rec = {
                "model": model, "page": page, "region_id": rid,
                "ordre": ordre,
                "status_code": status, "duration_s": round(dur, 2),
                "output": out,
                "error": None if status == 200 else _get(j, "error", default=None),
            }
            os.makedirs(os.path.dirname(rp), exist_ok=True)
            with open(rp, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=2)
            log(f"p{page:02d}_{ordre:02d} {model} status {status} duration {dur:.1f}s")
    return call_count


def read_line_raw(doc_id, model, page, ordre):
    p = raw_line_path(doc_id, model, page, ordre)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def read_latency(doc_id, model, page, ordre):
    rec = read_line_raw(doc_id, model, page, ordre)
    return None if rec is None else rec.get("duration_s")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_line(doc_id, metric_map):
    """metric_map[(model,page,rid)] -> metrics. Per-model micro-average and
    median per line. Builds per-(model,page) sums too for the page bootstrap."""
    agg = {}
    page_cer = {}  # (model,page) -> (Ser,Der,ler,Ner) normalized char edits
    for model in ALL_MODELS:
        rows = [m for (mm, *_), m in metric_map.items() if mm == model]
        if not rows:
            continue
        S = sum(r["S_norm"] for r in rows)
        D = sum(r["D_norm"] for r in rows)
        I = sum(r["I_norm"] for r in rows)
        N = sum(r["N_norm"] for r in rows)
        Ss = sum(r["S_strict"] for r in rows)
        Ds = sum(r["D_strict"] for r in rows)
        Is = sum(r["I_strict"] for r in rows)
        Ns = sum(r["N_strict"] for r in rows)
        Sw = sum(r["S_word"] for r in rows)
        Dw = sum(r["D_word"] for r in rows)
        Iw = sum(r["I_word"] for r in rows)
        Nw = sum(r["N_word"] for r in rows)
        agg[model] = {
            "n_lines": len(rows),
            "cer_norm_micro": (S + D + I) / N if N else None,
            "wer_micro": (Sw + Dw + Iw) / Nw if Nw else None,
            "cer_strict_micro": (Ss + Ds + Is) / Ns if Ns else None,
            "ser_norm": sum(r["ser_norm"] for r in rows) / len(rows),
            "ser_strict": sum(r["ser_strict"] for r in rows) / len(rows),
            "cer_norm_median": _median([r["cer_norm"] for r in rows]),
            "wer_median": _median([r["wer"] for r in rows]),
            "cer_strict_median": _median([r["cer_strict"] for r in rows]),
            "empty": sum(r["empty"] for r in rows),
            "too_long": sum(r["too_long"] for r in rows),
            "signals": sum(1 for r in rows if r["signals"]),
            "multi_line": sum(r["multi_line"] for r in rows),
            "median_latency": _median([
                read_latency(doc_id, mm, pp, rid)
                for (mm, pp, rid) in metric_map if mm == model
            ]),
        }
        by_page = {}
        for (mm, pp, rid), r in metric_map.items():
            if mm != model:
                continue
            b = by_page.setdefault(pp, [0, 0, 0, 0])
            b[0] += r["S_norm"]; b[1] += r["D_norm"]
            b[2] += r["I_norm"]; b[3] += r["N_norm"]
        for page, (pS, pD, pI, pN) in by_page.items():
            page_cer[(model, page)] = (pS, pD, pI, pN)
    return agg, page_cer


def bootstrap(page_cer, include_insertions, metric_name):
    """Paired bootstrap by page. page_cer[(model,page)] = (S,D,I,N) where S,D,I,N
    are normalized char edits micro-summed over the page's lines.

    Resample the pages with replacement (same seed across calls), recompute each
    model's micro CER over the sampled pages, track who beats whom and the CER
    difference. include_insertions=False compares CER reference (S+D)/N (ignores
    added text); True compares CER classic (S+D+I)/N (accounts for insertions).

    Returns per-model-pair win share and 95% CI of the difference."""
    pages = sorted({p for (_m, p) in page_cer})
    models = [m for m in ALL_MODELS if any(m == mm for (mm, _) in page_cer)]
    rng = random.Random(BOOTSTRAP_SEED)
    pairs = [(a, b) for a in models for b in models if a != b]
    wins = {p: 0.0 for p in pairs}
    diffs = {p: [] for p in pairs}
    cer_base = {}
    for m in models:
        totS = sum(v[0] for (mm, _p), v in page_cer.items() if mm == m)
        totD = sum(v[1] for (mm, _p), v in page_cer.items() if mm == m)
        totI = sum(v[2] for (mm, _p), v in page_cer.items() if mm == m)
        totN = sum(v[3] for (mm, _p), v in page_cer.items() if mm == m)
        num = (totS + totD) if not include_insertions else (totS + totD + totI)
        cer_base[m] = num / totN if totN else None
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [rng.choice(pages) for _ in pages]
        cer = {}
        for m in models:
            S = D = I = N = 0
            for pg in sample:
                if (m, pg) in page_cer:
                    pS, pD, pI, pN = page_cer[(m, pg)]
                    S += pS; D += pD; I += pI; N += pN
            num = (S + D) if not include_insertions else (S + D + I)
            cer[m] = num / N if N else None
        for a, b in pairs:
            if cer[a] is None or cer[b] is None:
                continue
            if cer[a] < cer[b]:
                wins[(a, b)] += 1
            diffs[(a, b)].append(cer[a] - cer[b])
    out = {}
    for a, b in pairs:
        share = wins[(a, b)] / BOOTSTRAP_DRAWS
        d = diffs[(a, b)]
        lo = sorted(d)[int(0.025 * len(d))]
        hi = sorted(d)[int(0.975 * len(d))]
        out[f"{a}__vs__{b}"] = {
            "share_a_beats_b": round(share, 4),
            "diff_a_minus_b_ic95": [round(lo, 4), round(hi, 4)],
            "cer_a": cer_base[a], "cer_b": cer_base[b],
        }
    return {"metric": metric_name, "pairs": out}


def norm_lev(s):
    return normalize(clean(s))


def concat_page_metrics(doc_id):
    """For each page, concat the per-line outputs of a model in XML order and
    compare to the page reference. Returns per (model,page): cer_ref (S+D)/N,
    cer classic (S+D+I)/N normalized, and N."""
    lines = read_document_lines(doc_id)
    page_ref = {}
    for r in lines:
        page_ref.setdefault(r["page"], []).append((r["ordre"], r["texte"]))
    out = {}
    for model in ALL_MODELS:
        for page, page_lines in page_ref.items():
            ref = normalize(clean(" ".join(t for _o, t in sorted(page_lines))))
            parts = []
            for ordre, _t in sorted(page_lines):
                rec = read_line_raw(doc_id, model, page, ordre)
                parts.append(rec["output"] if rec else "")
            hyp = normalize(clean(" ".join(parts)))
            d, S, D, I = levenshtein_counts(ref, hyp)
            N = len(ref)
            out[(model, page)] = {
                "N": N, "S": S, "D": D, "I": I,
                "cer_ref_norm": (S + D) / N if N else None,
                "cer_classic_norm": (S + D + I) / N if N else None,
            }
    return out


def cost_metrics(agg, lines, extrapolate_pages, rate_per_min):
    """Median latency per line, median lines/page, extrapolation to
    extrapolate_pages pages (calls, sequential duration, duration at rate_per_min)."""
    per_page = {}
    for r in lines:
        per_page.setdefault(r["page"], 0)
        per_page[r["page"]] += 1
    med_lines_page = _median(list(per_page.values()))
    n_lines = len(lines)
    calls_total = extrapolate_pages * (med_lines_page or 0)
    out = {
        "n_lines_total": n_lines,
        "n_pages": len(per_page),
        "median_lines_per_page": med_lines_page,
        f"calls_for_{extrapolate_pages}_pages": calls_total,
        "extrapolation": {},
    }
    for model, a in agg.items():
        lat = a["median_latency"]
        calls = calls_total
        seq_s = calls * (lat or 0)
        at_rate_s = calls / (rate_per_min / 60.0)
        out["extrapolation"][model] = {
            "median_latency_sec": lat,
            "calls": calls,
            "sequential_duration_days": round(seq_s / 86400, 1),
            f"duration_at_{rate_per_min}_req_min_days": round(at_rate_s / 86400, 1),
        }
    return out


def compute_all(doc_id, extrapolate_pages, rate_per_min):
    """Compute all metrics from cache (no API call)."""
    lines = read_document_lines(doc_id)
    metric_map = {}
    for row in lines:
        page, ordre = row["page"], row["ordre"]
        for model in ALL_MODELS:
            rec = read_line_raw(doc_id, model, page, ordre)
            if rec is None or rec.get("status_code") != 200:
                continue
            metric_map[(model, page, ordre)] = line_metrics(row["texte"], rec["output"])
    agg, page_cer = per_line(doc_id, metric_map)
    boot_ref = bootstrap(page_cer, include_insertions=False,
                         metric_name="CER reference (S+D)/N, micro per page")
    boot_class = bootstrap(page_cer, include_insertions=True,
                           metric_name="CER classic (S+D+I)/N, micro per page")
    concat = concat_page_metrics(doc_id)
    cost = cost_metrics(agg, lines, extrapolate_pages, rate_per_min)
    return {
        "doc": doc_id,
        "per_line": {
            f"{m}__p{p:02d}__o{o:02d}": meta
            for (m, p, o), meta in sorted(metric_map.items())
        },
        "par_model": agg,
        "concat_page": {
            f"{m}__p{p:02d}": meta
            for (m, p), meta in sorted(concat.items())
        },
        "bootstrap_cer_ref": boot_ref,
        "bootstrap_cer_classic": boot_class,
        "cost": cost,
    }


def write_outputs(doc_id, extrapolate_pages, rate_per_min):
    res = compute_all(doc_id, extrapolate_pages, rate_per_min)
    json_path = os.path.join(albert.OUTPUTS_DIR, f"lines_{doc_id}.json")
    md_path = os.path.join(albert.OUTPUTS_DIR, f"lines_{doc_id}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    _write_md(doc_id, res, md_path)
    log(f"wrote {json_path} and {md_path}")
    return res


def fmt3(v):
    return f"{v:.3f}" if v is not None else "–"


def _write_md(doc_id, res, path):
    L = []
    n_lines = res["cost"]["n_lines_total"]
    L.append(f"# Line by line — {doc_id}")
    L.append("")
    L.append("Transcription of each line isolated from a PAGE XML segmentation, "
             "with full-page vs line-by-line comparison and paired bootstrap.")
    L.append("")
    L.append(f"## Per-model aggregates ({n_lines} lines)")
    L.append("")
    L.append("| model | CER norm | CER strict | WER norm | SER norm | SER strict "
             "| med. lat. (s) | empty | >2x | signals | multi-line |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in ALL_MODELS:
        a = res["par_model"].get(m)
        if not a:
            continue
        L.append(
            f"| {m} | {fmt3(a['cer_norm_micro'])} | {fmt3(a['cer_strict_micro'])} "
            f"| {fmt3(a['wer_micro'])} | {fmt3(a['ser_norm'])} | {fmt3(a['ser_strict'])} "
            f"| {fmt3(a['median_latency'])} | {a['empty']} | {a['too_long']} "
            f"| {a['signals']} | {a['multi_line']} |"
        )
    L.append("")
    L.append("SER = share of lines that are not identical (0 = perfect, 1 = nothing identical).")
    L.append("")
    L.append("## Full page vs line by line — reference CER (S+D)/N, per page")
    L.append("")
    L.append("| model | full page (med. page) | lines concat (med. page) | "
             "lines concat — classic CER (S+D+I)/N |")
    L.append("|---|---|---|---|")
    conc = {}
    concat_classic = {}
    for m in ALL_MODELS:
        pk = [v for k, v in res["concat_page"].items() if k.startswith(f"{m}__")]
        conc[m] = _median([v["cer_ref_norm"] for v in pk])
        concat_classic[m] = _median([v["cer_classic_norm"] for v in pk])
    for m in ALL_MODELS:
        L.append(
            f"| {m} | {fmt3(None)} | {fmt3(conc.get(m))} "
            f"| {fmt3(concat_classic.get(m))} |"
        )
    L.append("")
    L.append("## Paired bootstrap per page")
    L.append("")
    L.append("### CER reference (S+D)/N, lines concatenated per page: ignores added text")
    L.append("")
    L.append("| pair (A vs B) | share of draws where A beats B | CER A - CER B (IC95) |")
    L.append("|---|---|---|")
    for pair, v in res["bootstrap_cer_ref"]["pairs"].items():
        L.append(f"| {pair.replace('__vs__', ' vs ')} | {v['share_a_beats_b']:.3f} "
                 f"| [{v['diff_a_minus_b_ic95'][0]:.3f}, "
                 f"{v['diff_a_minus_b_ic95'][1]:.3f}] |")
    L.append("")
    L.append("### CER classic (S+D+I)/N, line by line, micro-average over the sampled pages")
    L.append("")
    L.append("| pair (A vs B) | share of draws where A beats B | CER A - CER B (IC95) |")
    L.append("|---|---|---|")
    for pair, v in res["bootstrap_cer_classic"]["pairs"].items():
        L.append(f"| {pair.replace('__vs__', ' vs ')} | {v['share_a_beats_b']:.3f} "
                 f"| [{v['diff_a_minus_b_ic95'][0]:.3f}, "
                 f"{v['diff_a_minus_b_ic95'][1]:.3f}] |")
    L.append("")
    L.append("## Cost / extrapolation")
    L.append("")
    c = res["cost"]
    L.append(f"- total lines = {c['n_lines_total']}; pages = {c['n_pages']}; "
             f"median lines/page = {c['median_lines_per_page']}")
    for m, e in c["extrapolation"].items():
        L.append(f"- {m}: median latency = {fmt3(e['median_latency_sec'])}s, "
                 f"calls = {e['calls']}, "
                 f"sequential = {e['sequential_duration_days']}d, ")
    L.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def print_report(doc_id, res):
    print("==", "Per-model aggregates", "==")
    print(f"{'model':24} {'CER_norm':9} {'CER_str':9} {'WER':7} {'SER_n':7} "
          f"{'SER_s':7} {'lat':7}")
    for m in ALL_MODELS:
        a = res["par_model"].get(m)
        if not a:
            continue
        print(f"{m:24} {fmt3(a['cer_norm_micro']):9} {fmt3(a['cer_strict_micro']):9} "
              f"{fmt3(a['wer_micro']):7} {fmt3(a['ser_norm']):7} "
              f"{fmt3(a['ser_strict']):7} {fmt3(a['median_latency']):7}")
    print()
    for name, key in (("CER reference (S+D)/N", "bootstrap_cer_ref"),
                      ("CER classic (S+D+I)/N", "bootstrap_cer_classic")):
        print()
        print(f"== Paired bootstrap per page — share where A beats B — {name} ==")
        for pair, v in res[key]["pairs"].items():
            a, b = pair.split("__vs__")
            print(f"{a} vs {b}: share={v['share_a_beats_b']:.3f} diff={v['diff_a_minus_b_ic95']}")
    print()
    print("==", "Cost / extrapolation", "==")
    c = res["cost"]
    print(f"total lines={c['n_lines_total']} pages={c['n_pages']} "
          f"median lines/page={c['median_lines_per_page']} "
          f"calls={list(c['extrapolation'].values())[0]['calls'] if c['extrapolation'] else 0}")
    for m, e in c["extrapolation"].items():
        print(f"{m}: med_latency={e['median_latency_sec']}s "
              f"sequential={e['sequential_duration_days']}d ")


def run(doc_id, base_url, key, mode, probe, extrapolate_pages, rate_per_min):
    """Orchestrate a lines subcommand (probe|run|report) for a document."""
    import sys

    from eval_ocr_htr.data import exists
    if not exists(doc_id):
        sys.exit(f"PDF not found: {doc_id}")
    lines = read_document_lines(doc_id)
    if not lines:
        sys.exit(f"No PAGE XML lines found for document {doc_id}")
    if mode == "report":
        res = write_outputs(doc_id, extrapolate_pages, rate_per_min)
        print_report(doc_id, res)
        return
    key = albert.require_api_key(key)
    crop_lines(doc_id, lines)
    c = run_inference(base_url, key, doc_id, lines, probe=probe)
    what = "probe" if probe else "run"
    log(f"{what} API calls made: {c}")
    res = write_outputs(doc_id, extrapolate_pages, rate_per_min)
    print_report(doc_id, res)
