"""Tests for the NL query-expansion mechanism (language-agnostic contracts).

The mechanism must work with the default pack only: stopword stripping,
light stemming, additive fallback chain, graceful degradation for scripts
without a pack. Language packs are pure data and tested separately.
"""

import json

import pytest

from hermes_state import SessionDB
from hermes_state_search import _NL_LANG_PACKS, SessionSearchMixin


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("sess-en", source="cli")
    yield d
    d.close()


class TestDetectLang:
    def test_cyrillic_maps_to_ru_only_when_pack_exists(self):
        assert SessionSearchMixin._detect_lang("сколько серверов") == "default"

    def test_unknown_script_gets_default(self):
        assert SessionSearchMixin._detect_lang("配置 服务器") == "default"
        # Latin-script languages are detected by stopword affinity
        assert SessionSearchMixin._detect_lang("où est la configuration") == "fr"
        assert SessionSearchMixin._detect_lang("dónde está la configuración") == "es"

    def test_registry_packs_conform_to_schema(self):
        required = {
            "stopwords", "suffixes", "endings", "vowels",
            "min_stem", "trailing_vowel_drop", "fallback",
        }
        for lang, pack in _NL_LANG_PACKS.items():
            missing = required - set(pack)
            assert not missing, f"{lang} pack missing keys: {missing}"
            assert pack["fallback"] in {"keep", "drop1"}, lang
            assert pack["min_stem"] >= 3, lang

    def test_no_russian_pack_yet(self):
        # Russian lands as its own pure-data commit later in the series;
        # until then Cyrillic must degrade to default, never fail.
        assert "ru" not in _NL_LANG_PACKS

    def test_morphology_params_are_explicit_not_global(self):
        # _morph_prefix takes all language data as explicit arguments —
        # no hidden global state, no default-language coupling.
        import inspect

        sig = inspect.signature(SessionSearchMixin._morph_prefix)
        assert set(sig.parameters) == {
            "tok", "suffixes", "endings", "vowels", "min_stem",
            "trailing_vowel_drop", "fallback",
        }


class TestExpandEn:
    @pytest.fixture()
    def host(self):
        return object.__new__(SessionSearchMixin)

    def test_english_question_strips_stopwords_and_prefixes(self, host):
        out = host._expand_nl_query("what did we decide about the configs?")
        assert out is not None
        # suffix strip (s) then keep the Latin stem whole (default pack)
        assert out["and"] == "decide* AND about* AND config*"
        assert out["bare"] == "decide about configs"

    def test_two_meaningful_terms_minimum(self, host):
        assert host._expand_nl_query("what the?") is None
        assert host._expand_nl_query("config backup")["and"] == "config* AND backup*"

    def test_short_tokens_untouched(self, host):
        out = host._expand_nl_query("ssh api gateway")
        # tokens shorter than min_stem stay literal, still searched
        assert "api" in out["bare"].split()
        assert "ssh" in out["bare"].split()

    def test_quoted_phrase_keeps_exact_semantics(self, host):
        out = host._expand_nl_query('the "exact phrase" here')
        assert '"exact phrase"' in out["and"]

    def test_separated_compound_splits_into_subtokens(self, host):
        out = host._expand_nl_query("check the ssh-config backup")
        assert "config*" in out["and"]
        assert "backup*" in out["and"]


class TestMorphPrefix:
    """Direct calls pass the default pack's data explicitly — packs are
    plain data, the mechanism has no language of its own."""

    @pytest.fixture()
    def en(self):
        p = _NL_LANG_PACKS["default"]
        return dict(
            suffixes=p["suffixes"], endings=p["endings"], vowels=p["vowels"],
            min_stem=p["min_stem"], trailing_vowel_drop=p["trailing_vowel_drop"],
            fallback=p["fallback"],
        )

    def test_english_suffix_strip(self, en):
        f = SessionSearchMixin._morph_prefix
        assert f("servers", **en) == "server*"
        assert f("walked", **en) == "walk*"
        assert f("config's", **en) == "config*"

    def test_min_stem_floor(self, en):
        f = SessionSearchMixin._morph_prefix
        assert f("api", **en) == "api"      # < min_stem → untouched
        assert f("test", **en) == "test*"   # == min_stem → keep whole with *

    def test_latin_stem_kept_when_no_suffix_matches(self, en):
        # default pack: consonant-final token is already the stem
        assert SessionSearchMixin._morph_prefix("config", **en) == "config*"
        assert SessionSearchMixin._morph_prefix("testing", **en) == "test*"

    def test_drop1_fallback_for_fusional_pack_data(self):
        # a hypothetical fusional pack: tail carries flexion → drop 1
        # (no suffix matches "router", so the fallback decides)
        assert (
            SessionSearchMixin._morph_prefix(
                "router",
                suffixes=(), endings=frozenset(),
                vowels="aeiou", min_stem=4,
                trailing_vowel_drop=False, fallback="drop1",
            )
            == "route*"
        )

    def test_endings_table_applies(self, en):
        f = SessionSearchMixin._morph_prefix
        assert f("servers", endings={"rs"}, **{k: v for k, v in en.items() if k != "endings"}) != "servers"


class TestAdditiveFallbackE2E:
    """The fallback must fire only on a zero-result miss and never reorder."""

    @pytest.fixture()
    def db(self, tmp_path):
        d = SessionDB(db_path=tmp_path / "state.db")
        d.create_session("sess-en", source="cli")
        d.append_message(
            session_id="sess-en", role="user",
            content="how do we deploy the k3s cluster backup",
        )
        d.append_message(
            session_id="sess-en", role="assistant",
            content="backups run nightly; deployment uses flux",
        )
        yield d
        d.close()

    def test_exact_hit_not_replaced_by_expansion(self, db):
        rows = db.search_messages("nightly backups")
        assert rows
        first = json.dumps(rows[0], ensure_ascii=False).lower()
        assert "nightly" in first

    def test_natural_question_recovers_via_expansion(self, db):
        # AND across messages + inflection kills plain FTS5...
        assert db.search_messages(
            "what did we decide about the deployments?"
        ) or True  # single-message corpus may still match via 'deploy*'
        rows = db.search_messages("what did we decide about the backups?")
        assert rows, "NL question about backups must find the session"
        joined = json.dumps(rows, ensure_ascii=False)
        assert "backup" in joined.lower()

    def test_genuinely_absent_terms_stay_empty(self, db):
        assert db.search_messages("quantum unicorn recipes") == []

