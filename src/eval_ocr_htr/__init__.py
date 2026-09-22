"""eval-ocr-htr — benchmark OCR/HTR models on scanned handwritten documents.

Evaluates vision models served by Albert API (SecNumCloud) against per-page or
per-line reference transcriptions. All inference results are cached on disk so
reports can be regenerated without re-paying for API calls.

Privacy by design: no transcribed or reference text is ever printed, logged or
included in generated reports — only ids, page/line numbers, lengths, counts and
metrics.

CLI (subcommands, English):
    eval-ocr-htr models
    eval-ocr-htr probe [--doc ID] [--pages N-M] [--models a,b,c]
    eval-ocr-htr run   [--doc ID] [--pages N-M] [--models a,b,c]
    eval-ocr-htr report [--models a,b,c]
    eval-ocr-htr lines probe|run|report [--doc ID]
"""

import argparse
import sys

from eval_ocr_htr import albert
from eval_ocr_htr.albert import (
    ALL_MODELS, DEFAULT_EXTRAPOLATE_PAGES, DEFAULT_RATE_PER_MIN, MODELS_PER_PAGE,
    MODEL_OCR, config, ensure_dirs, log, require_api_key,
)
from eval_ocr_htr.data import list_documents


def _parse_interval(spec, low, high):
    """--pages 0-4 -> range(0,5). In any mode, clamped to [low, high)."""
    if not spec:
        return range(low, high)
    if "-" in spec:
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
    else:
        lo = hi = int(spec)
    lo = max(low, lo)
    hi = min(high - 1, hi)
    if hi < lo:
        return range(0)
    return range(lo, hi + 1)


def _resolve_models(spec):
    """Return the models to run from a comma-separated list, or all default ones."""
    if not spec:
        return ALL_MODELS
    wanted = [m.strip() for m in spec.split(",") if m.strip()]
    invalid = [m for m in wanted if m not in ALL_MODELS]
    if invalid:
        sys.exit(f"unknown model(s): {', '.join(invalid)}")
    return wanted


def _print_models(base_url, key):
    albert.available_models(base_url, key)
    print("Default models:")
    for m in ALL_MODELS:
        print(f"  {m}")
    print("Per-page vision models:")
    for m in MODELS_PER_PAGE:
        print(f"  {m} (per-page)")
    print(f"Whole-document OCR endpoint: {MODEL_OCR}")


def _process_doc(base_url, key, doc_id, pages, models, ctx, probe):
    from eval_ocr_htr import page
    for model in models:
        if model == MODEL_OCR:
            page.process_whole_document(base_url, key, ctx, doc_id, pages)
        else:
            page.process_per_page(base_url, key, ctx, doc_id, model, pages)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="eval-ocr-htr",
        description=(
            "Benchmark OCR/HTR models (Albert API) on scanned handwritten "
            "documents."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_models = sub.add_parser("models", help="list the default benchmark models")
    p_models.add_argument("--base-url", help="override ALBERT_BASE_URL")
    p_models.add_argument("--api-key", help="override ALBERT_API_KEY")

    for name in ("probe", "run"):
        p = sub.add_parser(name, help=f"{'validate the chain on one page' if name == 'probe' else 'run a full benchmark'}")
        p.add_argument("--doc", default=None, help="document id (PDF stem); default: first/processed docs")
        p.add_argument("--pages", metavar="N-M", default=None,
                       help="page interval (ex: 0-4)")
        p.add_argument("--models", default=None, help=f"comma-separated model ids (default: all {len(ALL_MODELS)})")
        p.add_argument("--base-url", help="override ALBERT_BASE_URL")
        p.add_argument("--api-key", help="override ALBERT_API_KEY")

    p_report = sub.add_parser("report", help="regenerate reports from cache (no API call)")
    p_report.add_argument("--models", default=None, help="comma-separated model ids (default: all)")
    p_report.add_argument("--extrapolate-pages", type=int, default=DEFAULT_EXTRAPOLATE_PAGES,
                          help=f"target page count for linear extrapolation (default: {DEFAULT_EXTRAPOLATE_PAGES})")
    p_report.add_argument("--base-url", help="override ALBERT_BASE_URL")
    p_report.add_argument("--api-key", help="override ALBERT_API_KEY")

    p_lines = sub.add_parser("lines", help="line-by-line mode on a PAGE XML segmentation")
    p_lines.add_argument("subcommand", choices=["probe", "run", "report"],
                         help="probe: 1 line x models; run: all lines; report: metrics from cache")
    p_lines.add_argument("--doc", default=None, help="document id (PDF stem)")
    p_lines.add_argument("--extrapolate-pages", type=int, default=DEFAULT_EXTRAPOLATE_PAGES,
                         help=f"target page count (default: {DEFAULT_EXTRAPOLATE_PAGES})")
    p_lines.add_argument("--rate-limit", type=int, default=DEFAULT_RATE_PER_MIN,
                         help=f"requests/minute for duration extrapolation (default: {DEFAULT_RATE_PER_MIN})")
    p_lines.add_argument("--base-url", help="override ALBERT_BASE_URL")
    p_lines.add_argument("--api-key", help="override ALBERT_API_KEY")

    args = parser.parse_args(argv)
    ensure_dirs()
    base_url, api_key = config(args)

    if args.cmd == "models":
        _print_models(base_url, api_key)
        return

    if args.cmd == "lines":
        from eval_ocr_htr import lines
        docs = list_documents()
        if not docs:
            sys.exit("No PDF found under data/documents/")
        doc_id = args.doc or docs[0]
        lines.run(
            doc_id, base_url, api_key, args.subcommand,
            probe=(args.subcommand == "probe"),
            extrapolate_pages=args.extrapolate_pages,
            rate_per_min=args.rate_limit,
        )
        return

    docs = list_documents()
    if not docs:
        sys.exit("No PDF found under data/documents/")
    doc_id = args.doc or docs[0]
    models = _resolve_models(args.models)

    if args.cmd == "report":
        from eval_ocr_htr import report
        report.write_outputs(docs, args.extrapolate_pages)
        log("Reports regenerated from cache (no API call).")
        return

    key = require_api_key(api_key)
    ctx = {"cost_cumul": 0.0}

    from eval_ocr_htr import page
    n = page.page_count(doc_id)
    interval = list(_parse_interval(args.pages, 0, n) if args.pages else range(0, n))

    if args.cmd == "probe":
        page.run_probe(base_url, key, doc_id, interval if interval else [0], ctx)
    else:  # run
        page.run_document(base_url, key, doc_id, interval, ctx)
        from eval_ocr_htr import report
        try:
            report.write_outputs(docs, args.extrapolate_pages)
        except Exception as e:
            log(f"report regeneration failed: {e}")
        log("Run complete.")


if __name__ == "__main__":
    main()
