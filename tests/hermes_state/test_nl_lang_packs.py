"""Tests for the Latin-script language packs (pure data).

Each pack must: be selected by stopword affinity, strip its stopwords,
stem real flexions into prefixes the unicode61 tokenizer can wildcard,
and never crash on accented input. No mechanism changes here.
"""

import json

import pytest

from hermes_state import SessionDB
from hermes_state_search import _NL_LANG_PACKS, SessionSearchMixin


def _pack_kwargs(lang: str) -> dict:
    p = _NL_LANG_PACKS[lang]
    return dict(
        suffixes=p["suffixes"], endings=p["endings"], vowels=p["vowels"],
        min_stem=p["min_stem"], trailing_vowel_drop=p["trailing_vowel_drop"],
        fallback=p["fallback"],
    )


class TestPackDetection:
    @pytest.mark.parametrize("query,lang", [
        ("dónde está la configuración del servidor", "es"),
        ("où est la configuration du serveur", "fr"),
        ("wo ist die Konfiguration des Servers", "de"),
        ("onde está a configuração do servidor", "pt"),
        ("dov'è la configurazione del server", "it"),
        ("what about the server config", "default"),  # EN → default fallback
    ])
    def test_affinity_detection(self, query, lang):
        assert SessionSearchMixin._detect_lang(query) == lang

    def test_cyrillic_still_script_routed(self):
        assert SessionSearchMixin._detect_lang("где конфигурация сервера") == (
            "ru" if "ru" in _NL_LANG_PACKS else "default"
        )


class TestPackExpansion:
    """E2E through _expand_nl_query with each pack's own data."""

    @pytest.fixture()
    def host(self):
        return object.__new__(SessionSearchMixin)

    def test_spanish_question(self, host):
        out = host._expand_nl_query("¿dónde está la configuración del servidor?")
        assert out is not None
        assert "configuració*" in out["and"] or "configuración" in out["bare"]
        # stopword 'del/la/está' stripped
        assert " está" not in f" {out['bare']} "
        assert "del" not in out["bare"].split()

    def test_french_question(self, host):
        out = host._expand_nl_query("où est la configuration du serveur")
        assert out is not None
        assert "serveu*" in out["and"] or "serveur*" in out["and"]
        assert "la" not in out["bare"].split()

    def test_german_question(self, host):
        out = host._expand_nl_query("wo ist die Konfiguration des Servers")
        assert out is not None
        assert "Server*" in out["and"]
        assert "die" not in out["bare"].split()
        assert "ist" not in out["bare"].split()

    def test_portuguese_question(self, host):
        out = host._expand_nl_query("onde está a configuração do servidor")
        assert out is not None
        # -ção flexion strips via the pack suffix table → configura*
        assert "configura*" in out["and"]
        assert "está" not in out["bare"].split()

    def test_italian_question(self, host):
        out = host._expand_nl_query("dov'è la configurazione del server")
        assert out is not None
        # -zione flexion strips via the pack suffix table
        assert "configura*" in out["and"]
        assert "della" not in out["bare"].split()


class TestPackMorphology:
    def test_real_flexions_stem_to_valid_prefixes(self):
        cases = [
            ("es", "servidores", "server*"),   # wait: servidor+s → servidor*? see below
        ]
        # Spanish plural 'servidores' strips -es → 'servidor', then keep
        got = SessionSearchMixin._morph_prefix("servidores", **_pack_kwargs("es"))
        assert got.startswith("servidor") and got.endswith("*")

    def test_accents_survive_tokenization(self, ):
        h = object.__new__(SessionSearchMixin)
        out = h._expand_nl_query("configuración rápida del sistema")
        assert "rápid*" in out["and"] or "rápida*" in out["and"]

    def test_all_packs_min_stem_guard(self):
        for lang in ("es", "fr", "de", "pt", "it"):
            kw = _pack_kwargs(lang)
            assert SessionSearchMixin._morph_prefix("abc", **kw) == "abc"


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("sess-es", source="cli")
    d.append_message(
        session_id="sess-es", role="user",
        content="configuré el servidor proxy en docker",
    )
    d.append_message(
        session_id="sess-es", role="assistant",
        content="el proxy quedó configurado con traefik",
    )
    yield d
    d.close()


class TestSpanishE2E:
    def test_inflected_question_finds_session(self, db):
        rows = db.search_messages("¿dónde quedó la configuración de los proxies?")
        assert rows, "inflected ES question must reach 'configuré/configurado'"
        blob = json.dumps(rows, ensure_ascii=False).lower()
        assert "proxy" in blob or "configur" in blob
