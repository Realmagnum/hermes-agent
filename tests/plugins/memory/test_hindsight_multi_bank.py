"""Tests for multi-bank (cross-bank) recall in the Hindsight plugin.

Covers the ``recall_banks`` config fan-out: RRF merge, dedup, primary/secondary
budgets, secondary-bank failure tolerance, and the guarantee that an unconfigured
setup keeps the single-bank call path byte-for-byte.
"""

import json

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from plugins.memory.hindsight import HindsightMemoryProvider, _merge_bank_results


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("HINDSIGHT_RECALL_BANKS", "HINDSIGHT_API_URL", "HINDSIGHT_MODE"):
        monkeypatch.delenv(key, raising=False)


def _item(text: str, id: str | None = None):
    return SimpleNamespace(id=id or f"i-{text[:12]}", text=text)


def _resp(*texts):
    return SimpleNamespace(results=[_item(t) for t in texts])


def _provider(tmp_path, monkeypatch, config: dict):
    config_path = tmp_path / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
    )
    provider = HindsightMemoryProvider()
    provider.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
    return provider


class _FakeClient:
    """Records arecall kwargs; returns canned results per bank."""

    def __init__(self, responses: dict):
        self.responses = responses  # bank_id -> list of result texts (or Exception)
        self.calls = []

    async def arecall(self, **kwargs):
        import asyncio

        await asyncio.sleep(0)
        self.calls.append(kwargs)
        r = self.responses.get(kwargs["bank_id"])
        if isinstance(r, Exception):
            raise r
        return _resp(*(r or []))

    async def aclose(self):
        pass


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_recall_banks_default_empty(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch, {"mode": "local_external", "bank_id": "hermes"})
    assert p._recall_banks == ["hermes"]


def test_recall_banks_from_config_list(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch, {
        "mode": "local_external", "bank_id": "hermes",
        "recall_banks": ["hermes-test"],
    })
    assert p._recall_banks == ["hermes", "hermes-test"]


def test_recall_banks_from_csv_string(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch, {
        "mode": "local_external", "bank_id": "hermes",
        "recall_banks": " hermes-dev , hermes-ops ,",
    })
    assert p._recall_banks == ["hermes", "hermes-dev", "hermes-ops"]


def test_recall_banks_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_RECALL_BANKS", "b2,b3")
    p = _provider(tmp_path, monkeypatch, {"mode": "local_external", "bank_id": "hermes"})
    assert p._recall_banks == ["hermes", "b2", "b3"]


def test_recall_banks_dedupe_primary(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch, {
        "mode": "local_external", "bank_id": "hermes",
        "recall_banks": ["hermes", "other"],
    })
    assert p._recall_banks == ["hermes", "other"]


# ---------------------------------------------------------------------------
# _merge_bank_results (RRF)
# ---------------------------------------------------------------------------


def test_merge_orders_by_fused_rank():
    a1, a2 = _item("a1"), _item("a2")
    b1 = _item("b1")
    merged = _merge_bank_results([("A", [a1, a2]), ("B", [b1])])
    # RRF: a1 = 1/(60+1), b1 = 1/(60+1) (tie -> primary bank first),
    # a2 = 1/(60+2). So the fused order is a1, b1, a2.
    assert [it.text for _, it in merged] == ["a1", "b1", "a2"]


def test_merge_cross_bank_hit_outranks_single_bank():
    only_a = _item("only-a")
    both_a = _item("shared")
    both_b = _item("shared-b-text")  # distinct id AND text -> counts twice by id? no: ids differ
    shared_id_item = _item("dup", id="x1")
    dup_in_b = SimpleNamespace(id="x1", text="dup")
    merged = _merge_bank_results([
        ("A", [only_a, shared_id_item]),
        ("B", [dup_in_b]),
    ])
    texts = [it.text for _, it in merged]
    # x1 appears in both banks -> fused score beats rank-1-only item.
    assert texts[0] == "dup"
    assert texts.count("dup") == 1  # deduped by id


def test_merge_collapses_exact_duplicate_text_without_id():
    no_id_a = SimpleNamespace(id=None, text="same text")
    no_id_b = SimpleNamespace(id=None, text="same text")
    merged = _merge_bank_results([("A", [no_id_a]), ("B", [no_id_b])])
    assert len(merged) == 1


def test_merge_skips_empty_text():
    blank = SimpleNamespace(id="z", text="")
    real = _item("real")
    merged = _merge_bank_results([("A", [blank, real])])
    assert [(it.text) for _, it in merged] == ["real"]


# ---------------------------------------------------------------------------
# Fan-out behavior through the provider
# ---------------------------------------------------------------------------


def test_single_bank_when_unconfigured(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch, {"mode": "local_external", "bank_id": "main"})
    client = _FakeClient({"main": ["m1"]})
    p._client = client
    merged = p._multi_bank_recall_results("q")
    assert [it.text for _, it in merged] == ["m1"]
    assert len(client.calls) == 1
    assert client.calls[0]["bank_id"] == "main"
    assert client.calls[0]["budget"] == p._budget


def test_multi_bank_queries_all_and_merges(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch, {
        "mode": "local_external", "bank_id": "main", "recall_banks": ["side"],
    })
    client = _FakeClient({"main": ["m1"], "side": ["s1"]})
    p._client = client
    merged = p._multi_bank_recall_results("q")
    texts = [it.text for _, it in merged]
    assert set(texts) == {"m1", "s1"}
    banks_called = {c["bank_id"] for c in client.calls}
    assert banks_called == {"main", "side"}
    main_call = next(c for c in client.calls if c["bank_id"] == "main")
    side_call = next(c for c in client.calls if c["bank_id"] == "side")
    assert main_call["budget"] == p._budget          # primary keeps configured budget
    assert side_call["budget"] == "low"              # secondary forced low


def test_secondary_failure_is_tolerated(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch, {
        "mode": "local_external", "bank_id": "main", "recall_banks": ["dead"],
    })
    client = _FakeClient({"main": ["m1"], "dead": RuntimeError("boom")})
    p._client = client
    merged = p._multi_bank_recall_results("q")
    assert [it.text for _, it in merged] == ["m1"]


def test_primary_failure_raises(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch, {
        "mode": "local_external", "bank_id": "main", "recall_banks": ["side"],
    })
    client = _FakeClient({"main": RuntimeError("primary down"), "side": ["s1"]})
    p._client = client
    with pytest.raises(RuntimeError, match="primary down"):
        p._multi_bank_recall_results("q")


def test_tool_output_labels_secondary_bank(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch, {
        "mode": "local_external", "bank_id": "main", "recall_banks": ["side"],
    })
    client = _FakeClient({"main": [], "side": ["from-side"]})
    p._client = client
    out = json.loads(p.handle_tool_call("hindsight_recall", {"query": "q"}))
    assert "[@side]" in out["result"]
    assert "No relevant memories" not in out["result"]
