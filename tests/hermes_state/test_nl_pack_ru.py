"""Tests for the Russian language pack (pure data on the shared mechanism).

Russian is one consumer of the language-agnostic expansion mechanism —
these tests mirror test_nl_lang_packs.py and add nothing to the engine.
"""

import json

import pytest

from hermes_state import SessionDB
from hermes_state_search import _NL_LANG_PACKS, SessionSearchMixin


def _ru_kwargs() -> dict:
    p = _NL_LANG_PACKS["ru"]
    return dict(
        suffixes=p["suffixes"], endings=p["endings"], vowels=p["vowels"],
        min_stem=p["min_stem"], trailing_vowel_drop=p["trailing_vowel_drop"],
        fallback=p["fallback"],
    )


class TestRuDetection:
    def test_script_detection(self):
        assert SessionSearchMixin._detect_lang("сколько алиасов в ssh-конфиге") == "ru"

    def test_cyrillic_beats_affinity_ties(self):
        # mixed-script query: Cyrillic wins by script before affinity runs
        assert SessionSearchMixin._detect_lang("проверить nginx конфиг") == "ru"


class TestRuMorphology:
    @pytest.mark.parametrize("tok,expected", [
        ("алиасов", "алиас*"),      # 2-char ending table
        ("серверы", "сервер*"),     # trailing vowel drop
        ("роутера", "роутер*"),
        ("конфигов", "конфиг*"),
    ])
    def test_frequent_inflections(self, tok, expected):
        assert SessionSearchMixin._morph_prefix(tok, **_ru_kwargs()) == expected

    def test_short_token_untouched(self):
        assert SessionSearchMixin._morph_prefix("ssh", **_ru_kwargs()) == "ssh"


class TestRuExpansion:
    @pytest.fixture()
    def host(self):
        return object.__new__(SessionSearchMixin)

    def test_natural_question_expansion(self, host):
        out = host._expand_nl_query("сколько алиасов в ssh-конфиге и какие серверы там прописаны?")
        assert out is not None
        # stopwords «сколько», «в», «какие», «там» stripped
        for sw in ("сколько", "какие", "там"):
            assert sw not in out["bare"].split()
        assert "алиас*" in out["and"]
        assert "сервер*" in out["and"]
        # separated compound keeps its parts
        assert "конфи" in out["bare"] or "конфиг" in out["bare"]


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("sess-ru", source="cli")
    d.append_message(
        session_id="sess-ru", role="user",
        content="сколько алиасов в ssh-конфиге прописано?",
    )
    d.append_message(
        session_id="sess-ru", role="assistant",
        content="в конфиге три алиаса: pve1, pve2 и бастион",
    )
    yield d
    d.close()


class TestRuE2E:
    def test_inflected_question_finds_session(self, db):
        rows = db.search_messages(
            "найди в истории сколько алиасов в ssh-конфиге было"
        )
        assert rows, "inflected RU question must reach the alias session"
        blob = json.dumps(rows, ensure_ascii=False).lower()
        assert "алиас" in blob

    def test_genuinely_absent_stays_empty(self, db):
        assert db.search_messages("квантовые единороги и рецепты") == []
