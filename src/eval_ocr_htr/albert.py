"""Albert API client, configuration and on-disk cache.

Centralizes the environment / .env loading, the HTTP client with retries, and the
raw-response cache that makes the benchmark resumable and lets reports be
regenerated without re-paying for API calls.

Privacy by design: this module never prints or logs any transcribed or reference
text — only identifiers (ids, page/line numbers), lengths, counts and metrics.
The documents themselves never leave the machine except to Albert API (SecNumCloud).
"""

import json
import os
import re
import sys
import time
from datetime import datetime

import requests

DEFAULT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
DEFAULT_RATE_PER_MIN = 100
DEFAULT_EXTRAPOLATE_PAGES = 100_000

# Whole-document OCR endpoint (one request per document).
MODEL_OCR = "mistral-ocr-2512"
# Per-page vision models (one request per page).
MODELS_PER_PAGE = ["gemma-4-31b-it", "lightonocr-2-1b"]
ALL_MODELS = [MODEL_OCR] + MODELS_PER_PAGE

# The transcription prompt stays in French because the target documents are
# handwritten French pages. It is corpus-agnostic.
HTR_PROMPT = (
    "Transcris fidèlement le texte manuscrit de la page en français en conservant "
    "l'orthographe et la ponctuation d'origine y compris les fautes, restitue la mise "
    "en page en markdown simple, ne commente rien, ne résume rien, n'invente rien, "
    "écris [illisible] pour un passage indéchiffrable, et ne produis rien d'autre "
    "que la transcription. N'utilise aucune balise HTML, aucune image markdown, "
    "aucun bloc de code. Ne restitue pas les lignes de pointillés ou de tirets des "
    "champs laissés vides dans le formulaire pré-imprimé : un champ vide se note par "
    "rien du tout."
)

OCR_DPI = 200
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5
RETRY_BASE_DELAY = 2.0
REQUEST_TIMEOUT = 180
NEAR_EMPTY_CHARS = 20

DATA_DIR = os.environ.get("EVAL_OCR_HTR_DATA_DIR", "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
GROUND_TRUTH_DIR = os.path.join(DATA_DIR, "ground_truth")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")
RAW_DIR = os.path.join(OUTPUTS_DIR, "raw")


# ---------------------------------------------------------------------------
# Configuration / environment
# ---------------------------------------------------------------------------

def _load_dotenv(path=".env"):
    """Load KEY=VALUE lines from a .env file, without overriding variables that
    are already set in the process environment."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def config(args=None):
    """Return (base_url, api_key). Reads ALBERT_BASE_URL / ALBERT_API_KEY from
    the environment (after loading ./.env), with optional CLI overrides."""
    _load_dotenv()
    base_url = os.environ.get("ALBERT_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    api_key = os.environ.get("ALBERT_API_KEY", "")
    if args is not None:
        if getattr(args, "base_url", None):
            base_url = args.base_url.rstrip("/")
        if getattr(args, "api_key", None):
            api_key = args.api_key
    return base_url, api_key


def require_api_key(api_key):
    if not api_key:
        sys.exit("ALBERT_API_KEY is not set in the environment")
    return api_key


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dirs() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _get(d, *keys, default=None):
    """Safely traverse a mixed dict/list path, returning default on any miss."""
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and isinstance(k, int) and -len(cur) <= k < len(cur):
            cur = cur[k]
        else:
            return default
    return cur


# ---------------------------------------------------------------------------
# Albert API interactions
# ---------------------------------------------------------------------------

def health_models(base_url, key: str) -> dict:
    """Return {model_id: status} from GET /health/models."""
    try:
        r = requests.get(
            f"{base_url}/health/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        if r.status_code != 200:
            log(f"health/models replied {r.status_code}, no model filter applied")
            return {}
        data = r.json().get("data", [])
        return {m.get("id"): m.get("status") for m in data if m.get("id")}
    except Exception as e:
        log(f"health/models unavailable ({e}), no filter applied")
        return {}


def available_models(base_url, key: str) -> list:
    """Models with status 'green'. Log skipped ones."""
    health = health_models(base_url, key)
    available = []
    for m in ALL_MODELS:
        status = health.get(m)
        if status != "green":
            log(f"model {m} skipped (status='{status}', not green)")
            continue
        available.append(m)
    return available


def request_with_retry(base_url, method: str, path: str, key: str, **kw):
    """HTTP request against Albert API with retries on 429/503 and backoff.

    Returns the requests.Response. Host is always derived from base_url, so the
    Authorization header and the endpoint stay server-scoped (no arbitrary URLs)."""
    headers = kw.pop("headers", {})
    headers["Authorization"] = f"Bearer {key}"
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        r = requests.request(
            method, f"{base_url}{path}", headers=headers,
            timeout=REQUEST_TIMEOUT, **kw,
        )
        if r.status_code in (429, 503) and attempt < MAX_RETRIES:
            delay = RETRY_BASE_DELAY * (RETRY_BACKOFF ** (attempt - 1))
            log(f"  retry {attempt}/{MAX_RETRIES} code {r.status_code}, backoff {delay:.0f}s")
            time.sleep(delay)
            last = r
            continue
        return r
    return last  # should not happen (attempt<MAX handled above)


# ---------------------------------------------------------------------------
# Raw response cache
# ---------------------------------------------------------------------------

def raw_path(model: str, page: int) -> str:
    return os.path.join(RAW_DIR, f"{model}__p{page:02d}.json")


def raw_path_doc(model: str) -> str:
    return os.path.join(RAW_DIR, f"{model}__doc.json")


def write_raw(record: dict, model: str, page: int) -> None:
    with open(raw_path(model, page), "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)


def write_raw_doc(record: dict, model: str) -> None:
    with open(raw_path_doc(model), "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)


def read_raw(model: str, page: int):
    p = raw_path(model, page)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def read_raw_doc(model: str):
    p = raw_path_doc(model)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Output cleaning / degenerate signals (shared by page & line modes)
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    """Strip non-content noise from a transcription for display and metrics only.
    Order: HTML tags, markdown images, runs of 4+ dots (blank pre-printed form
    fields), markdown emphasis chars, then normalize spaces and drop empty lines."""
    t = text or ""
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"\.{4,}", "", t)
    t = re.sub(r"[*_`#>]", "", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return "\n".join(ln for ln in t.splitlines() if ln.strip())


def degenerate_signals(text: str) -> list:
    """Return the list of unusable signals triggered by a transcription,
    regardless of output length. Empty list => usable page. Signals:
    - 'repetition'         : a same non-empty line appears 5+ times, OR unique /
                             non-empty lines ratio < 0.5 over >= 6 lines
    - 'absurd_number_run'  : a run of 15+ digits
    - 'illegible_dominance': [illisible] marks > 50% of whitespace-separated words
    """
    t = text or ""
    signals = []
    from collections import Counter
    nonempty = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if nonempty:
        counts = Counter(nonempty)
        if max(counts.values()) >= 5:
            signals.append("repetition")
        elif len(nonempty) >= 6 and len(counts) / len(nonempty) < 0.5:
            signals.append("repetition")
    if re.search(r"\d{15,}", t):
        signals.append("absurd_number_run")
    words = re.findall(r"\S+", t)
    if words and words.count("[illisible]") / len(words) > 0.5:
        signals.append("illegible_dominance")
    return signals


def is_near_empty(text: str) -> bool:
    """A transcription is 'near-empty' when it carries almost no content
    (typical of a blank scan page)."""
    return len((text or "").strip()) <= NEAR_EMPTY_CHARS
