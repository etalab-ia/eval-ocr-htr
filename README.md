# eval-ocr-htr

Benchmark OCR/HTR (handwritten text recognition) models served by the **Albert API**
(SecNumCloud) on scanned handwritten documents. It measures transcription quality
(CER / WER), latency, cost and environmental impact, both on whole pages and line
by line, and turns the results into Markdown reports you can reproduce offline from
an on-disk cache.

This project ships no document corpus, no ground truth and no results — only the
tooling.

## Scope

Two complementary evaluation modes:

- **Full-page** (`run`): each page of a PDF is transcribed by two kinds of models —
  a whole-document OCR endpoint (one request per document) and per-page vision
  models (one request per page) — then compared page by page to an optional
  reference transcription (`pNN.txt`).
- **Line-by-line** (`lines run`): each `TextRegion` of a `PAGE XML` segmentation is
  treated as a handwritten line, cropped from the embedded PDF image at its native
  resolution, transcribed individually, and compared to the per-line reference
  carried in the same `PAGE XML`.

## Getting started

```bash
# create a virtualenv and install the package (+ dev deps)
uv sync

# configure credentials (never committed)
cp .env.example .env        # then fill ALBERT_BASE_URL / ALBERT_API_KEY
```

Requires Python ≥ 3.12. The only runtime dependencies are `requests` and `pymupdf`.

## Data layout

```
data/
├── documents/<doc_id>.pdf     PDF scans (one file per document)
├── ground_truth/<doc_id>/pNN.txt   optional page reference transcription
├── ground_truth/<doc_id>/pNN.xml   PAGE XML per page (required for line mode)
├── manifest.csv               optional; doc_id,category
└── outputs/                   API response cache + generated Markdown/JSON reports
```

There is deliberately **no mapping file**: ground truth is associated page by page
by the presence of `pNN.txt` / `pNN.xml` files.

## Running a benchmark

```bash
eval-ocr-htr models                          # list the default benchmark models
eval-ocr-htr probe  --doc <id> --pages 0-4   # validate the chain on a few pages
eval-ocr-htr run    --doc <id> --pages 0-4   # full benchmark, cached & resumable
eval-ocr-htr report                          # regenerate reports from cache only

eval-ocr-htr lines probe --doc <id>          # 1 line x all models
eval-ocr-htr lines run   --doc <id>          # all lines
eval-ocr-htr lines report --doc <id>         # metrics from cache only
```

Options: `--doc` document id, `--pages N-M` page interval, `--models a,b,c`
model subset, `--extrapolate-pages` target count for linear cost/duration
extrapolation, `--rate-limit` requests/minute for the duration extrapolation.

Every API response is cached on disk under `data/outputs/raw`, so interrupted
runs resume where they stopped and reports (`report`, `lines report`) are
regenerated **without re-paying for API calls**.

## Metrics

- **CER** (character error rate) via Levenshtein edit distance, computed on a
  normalized reference (lowercase, accent/punctuation stripped, whitespace
  collapsed) or on a strictly space-normalized reference, reported both without
  insertions (`(S+D)/N`, ignores added text — "surplus") and with insertions
  (`(S+D+I)/N`).
- **WER / SER** at the line level, plus invention indicators (empty, >2× longer,
  multi-line).
- **Word recall** — share of reference words present in the output.
- **Degenerate / unusable signals** — repetition, absurd number runs,
  illegible dominance — flagged per page/line.
- **Cost, impact, latency** — normalized per page, plus a **linear extrapolation**
  to a target page count (a simple multiplication from the sample actually
  processed, not a platform measurement).
- **Paired bootstrap per page** in line mode to compare models on the same pages.

> Note: the tool **never outputs reference or transcribed text** into console,
> logs or reports — only identifiers, page/line numbers, lengths, counts and
> metrics. The extreme case of a transcription is gated to [0, 1] per metric.

## Data privacy by design

The documents are only ever sent to the Albert API (SecNumCloud). Nothing — no
handwritten text, no reference, no PDF page — is embedded in the generated
reports. This keeps the results shareable without exposing the source content.

## Related

Companion evaluation infra by the same team: [etalab-ia/eval-transcript](https://github.com/etalab-ia/eval-transcript).

## License

MIT — see [LICENSE](LICENSE).
