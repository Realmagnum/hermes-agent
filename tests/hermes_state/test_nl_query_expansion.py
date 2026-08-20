"""Tests for the NL query expansion (L1) in hermes_state_search.

Covers stopword stripping, Russian morphology prefixes, and the
progressive fallback (AND → OR → trigram) without needing a live DB:
the expansion/morph helpers are pure functions on SessionSearchMixin.
"""

import pytest

from hermes_state_search import SessionSearchMixin


class _Dummy(SessionSearchMixin):
    """Minimal host exposing just the static/instance helpers under test."""

    pass


DUMMY = _Dummy()


@pytest.mark.parametrize(
    "tok,expected",
    [
        # Russian: vowel-ending words drop 1 char
        ("серверы", "сервер*"),
        ("конфиге", "конфиг*"),
        ("прописаны", "прописан*"),
        # Russian: consonant-ending words drop 2 chars (plural endings)
        ("алиасов", "алиас*"),
        ("роутера", "роутер*"),
        ("проверки", "проверк*"),
        # 4-char stems: kept whole (never cut below 4 chars)
        ("роль", "роль*"),
        ("порт", "порт*"),
        # Latin: full word + wildcard
        ("config", "config*"),
        ("tavily", "tavily*"),
        # Short tokens unchanged
        ("ssh", "ssh"),
        ("LXC", "LXC"),
    ],
)
def test_morph_prefix(tok, expected):
    assert SessionSearchMixin._morph_prefix(tok) == expected


def test_expand_strips_stopwords_and_builds_variants():
    exp = DUMMY._expand_nl_query(
        "Сколько алиасов в ssh-конфиге и какие серверы там прописаны?"
    )
    assert exp is not None
    # stopwords «сколько», «в», «и», «какие», «там» gone
    assert "сколько" not in exp["and"]
    assert "в " not in exp["and"]
    # morphology prefixes present
    assert "алиас*" in exp["and"]
    assert "сервер*" in exp["and"]
    # three variants, strictly loosening
    assert len(exp["and"].split(" AND ")) == len(exp["or"].split(" OR "))
    assert exp["bare"] == "алиасов ssh конфиге серверы прописаны"


def test_expand_splits_separator_tokens():
    exp = DUMMY._expand_nl_query('роутер для подсети "10.10.20.0/24"')
    assert exp is not None
    # quoted IP is split, not kept as an exact phrase
    assert "10.10.20.0/24" not in exp["and"]
    # «роутер» is present as a prefixed term («роуте*» matches роутер/роутеры)
    assert any("роуте" in p for p in exp["and"].split(" AND "))
    # stopword «для» is gone, «подсети» → «подсет*»
    assert "для" not in exp["and"]
    assert "подсет*" in exp["and"]


def test_expand_returns_none_for_insufficient_terms():
    assert DUMMY._expand_nl_query("в и на") is None
    assert DUMMY._expand_nl_query("привет") is None
    assert DUMMY._expand_nl_query("") is None


def test_expand_keeps_quoted_phrases_without_separators():
    exp = DUMMY._expand_nl_query('точная "фраза для поиска" настройка')
    assert exp is not None
    # the pure-word phrase survives verbatim in the AND/OR variants
    assert '"фраза для поиска"' in exp["and"]
    assert "настройк*" in exp["and"]
