"""Unit tests for the metrics module (CER, Levenshtein, word recall,
normalization). All fixtures are synthetic strings.
"""

from eval_ocr_htr.metrics import (
    cer_with_insertions,
    cer_without_insertions,
    levenshtein_counts,
    normalize,
    word_recall,
    alignment_stats,
    page_metrics,
)


def test_normalize_lowercases_and_collapses():
    assert normalize("  Bonjour   Monde  ") == " bonjour monde "


def test_normalize_keeps_diacritics():
    # accent errors count as character errors: accents are preserved
    assert normalize("É") == "é"
    assert normalize("Héllo Wörld") == "héllo wörld"
    assert normalize("Héllo Wörld") != normalize("Hello World")


def test_normalize_strips_punctuation():
    assert normalize("Salut, toi !") == "salut toi "


def test_levenshtein_counts_empty():
    assert levenshtein_counts("", "") == (0, 0, 0, 0)


def test_levenshtein_counts_pure_insertion():
    # return tuple is (distance, S, D, I); inserting 'x' -> I=1
    d, S, D, I = levenshtein_counts("abc", "xabc")
    assert (d, S, D, I) == (1, 0, 0, 1)


def test_levenshtein_counts_pure_deletion():
    d, S, D, I = levenshtein_counts("xabc", "abc")
    assert (d, S, D, I) == (1, 0, 1, 0)


def test_levenshtein_counts_substitution():
    d, S, D, I = levenshtein_counts("cat", "cot")
    assert (d, S, D, I) == (1, 1, 0, 0)


def test_alignment_stats_insertions():
    a = alignment_stats("aple", "apple")
    # 'p' in the output has no counterpart -> one insertion
    assert a["I"] == 1


def test_cer_without_insertions_ignores_insertions():
    # N=5, an insertion I=1 doesn't raise the "ref" CER
    assert cer_without_insertions(S=0, D=0, N=5) == 0.0
    # one deletion among 3 ref chars
    assert cer_without_insertions(S=0, D=1, N=3) == 1 / 3


def test_cer_without_insertions_substitution():
    assert cer_without_insertions(S=1, D=0, N=3) == 1 / 3


def test_cer_with_insertions_counts_insertions():
    assert cer_with_insertions(S=0, D=0, I=1, N=5) == 1 / 5


def test_word_recall():
    recall = word_recall(["le", "chat", "dort"], ["le", "chat", "mange"])
    assert recall == 2 / 3


def test_page_metrics_no_text():
    m = page_metrics("Le chat dort.", "Le chat dort.")
    assert "text" not in m and "output" not in m
    assert abs(m["cer_ref_norm"] - 0.0) < 1e-9
