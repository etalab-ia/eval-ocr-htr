"""Metrics for the OCR/HTR benchmark.

Provides the edit-distance based metrics: CER and WER, with and without
insertions, SER (line with at least one error), plus alignment / word-recall
statistics and the shared text normalization.

Only ids, page/line numbers, lengths, counts and metrics ever leave this module —
never any transcribed or reference text.
"""

import re
import statistics
from collections import Counter

from eval_ocr_htr.albert import clean, degenerate_signals


# ---------------------------------------------------------------------------
# Reference normalization
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Shared normalization for both reference and model outputs: lowercase,
    collapse whitespace, drop punctuation. Diacritics are kept (an accent error
    counts as a character error)."""
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)  # drop punctuation (keep letters/digits/space)
    return t


def normalize_strict(text: str) -> str:
    """Normalize spacing only, preserving case and punctuation. Used for the
    'strict' CER variant."""
    t = text or ""
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------

def levenshtein_counts(ref, hyp):
    """Minimal edit distance with backtrace.

    Returns (distance, S, D, I):
      S = substitutions, D = deletions (ref chars missing), I = insertions
    (extra hyp chars). O(len(ref)*len(hyp)) time and space, stdlib only.
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return dp[n][m], S, D, I


def alignment_stats(ref, hyp):
    """From an optimal character alignment, count:
      M  ref chars matched (equal),
      S  ref chars substituted,
      D  ref chars deleted (absent from hyp),
      I  hyp chars inserted (absent from ref).
    Returns dict with shares: ref_aligned = (M+S)/N (share of the reference that
    is present, possibly via substitution, in the output) and out_not_aligned =
    I/len(hyp) (share of the output that has no counterpart in the reference)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    i, j = n, m
    M = S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] and ref[i - 1] == hyp[j - 1]:
            M += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    N = len(ref)
    L = len(hyp)
    return {
        "M": M, "S": S, "D": D, "I": I,
        "ref_aligned": (M + S) / N if N else None,
        "out_not_aligned": I / L if L else None,
    }


def word_recall(ref_words, hyp_words):
    """Fraction of reference words (multiset, order-insensitive) found in the
    hypothesis word multiset. Controls the reading-order bias."""
    if not ref_words:
        return 1.0
    r = Counter(ref_words)
    h = Counter(hyp_words)
    found = 0
    for w, c in r.items():
        found += min(c, h.get(w, 0))
    return found / sum(r.values())


def cer_without_insertions(S, D, N):
    return (S + D) / N if N else None


def cer_with_insertions(S, D, I, N):
    return (S + D + I) / N if N else None


def word_tokens(s):
    return normalize(clean(s)).split()


# ---------------------------------------------------------------------------
# Per-page metrics
# ---------------------------------------------------------------------------

def page_metrics(ref_raw, out_raw):
    """Compute all metrics for one page given raw reference and model output.

    Returns a dict with counts and metrics. Never contains text.
    """
    ref_clean = clean(ref_raw)
    out_clean = clean(out_raw)
    ref_norm = normalize(ref_clean)
    out_norm = normalize(out_clean)
    ref_strict = normalize_strict(ref_clean)
    out_strict = normalize_strict(out_clean)

    N_norm = len(ref_norm)
    N_strict = len(ref_strict)

    d_n, S_n, D_n, I_n = levenshtein_counts(ref_norm, out_norm)
    d_s, S_s, D_s, I_s = levenshtein_counts(ref_strict, out_strict)

    ref_words = ref_norm.split()
    out_words = out_norm.split()

    return {
        "N_norm": N_norm,
        "N_strict": N_strict,
        "levenshtein_min_norm": d_n,
        "levenshtein_min_strict": d_s,
        "S_norm": S_n, "D_norm": D_n, "I_norm": I_n,
        "S_strict": S_s, "D_strict": D_s, "I_strict": I_s,
        "cer_ref_norm": cer_without_insertions(S_n, D_n, N_norm),
        "cer_raw_norm": cer_with_insertions(S_n, D_n, I_n, N_norm),
        "surplus_norm": (I_n / N_norm) if N_norm else None,
        "cer_ref_strict": cer_without_insertions(S_s, D_s, N_strict),
        "cer_raw_strict": cer_with_insertions(S_s, D_s, I_s, N_strict),
        "surplus_strict": (I_s / N_strict) if N_strict else None,
        "word_recall": word_recall(ref_words, out_words),
        "len_ref": len(ref_raw),
        "len_out": len(out_raw),
        "signals": degenerate_signals(out_raw),
    }


# ---------------------------------------------------------------------------
# Per-line metrics
# ---------------------------------------------------------------------------

def line_metrics(ref_raw, out_raw):
    """Per-line metrics, never contains text. CER classic (S+D+I)/N, WER, SER,
    normalized and strict + invention indicators."""
    out = out_raw or ""
    ref_norm = normalize(clean(ref_raw))
    out_norm = normalize(clean(out))
    ref_strict = normalize_strict(clean(ref_raw))
    out_strict = normalize_strict(clean(out))

    Nn = len(ref_norm)
    Ns = len(ref_strict)

    d_n, S_n, D_n, I_n = levenshtein_counts(ref_norm, out_norm)
    d_s, S_s, D_s, I_s = levenshtein_counts(ref_strict, out_strict)

    rw = word_tokens(ref_raw)
    ow = word_tokens(out_raw)
    d_w, S_w, D_w, I_w = levenshtein_counts(rw, ow)
    Nw = len(rw)

    signals = degenerate_signals(out)

    empty_out = len(clean(out)) == 0
    long_out = len(out) > 2 * max(len(ref_raw), 1)
    multiline = len([l for l in out.splitlines() if l.strip()]) > 1

    return {
        "N_norm": Nn, "S_norm": S_n, "D_norm": D_n, "I_norm": I_n,
        "cer_norm": cer_with_insertions(S_n, D_n, I_n, Nn),
        "N_strict": Ns, "S_strict": S_s, "D_strict": D_s, "I_strict": I_s,
        "cer_strict": cer_with_insertions(S_s, D_s, I_s, Ns),
        "N_word": Nw, "S_word": S_w, "D_word": D_w, "I_word": I_w,
        "wer": (S_w + D_w + I_w) / Nw if Nw else None,
        "ser_norm": int(ref_norm != out_norm),
        "ser_strict": int(ref_strict != out_strict),
        "len_ref": len(ref_raw), "len_out": len(out_raw),
        "signals": signals, "empty": empty_out, "too_long": long_out,
        "multi_line": multiline,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else 0.0
