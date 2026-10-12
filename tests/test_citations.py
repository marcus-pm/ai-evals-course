"""Citation guard (RESP-1), tested offline with stand-in run items and a fake
rewrite (no LLM, no API keys)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent import citations


def _call(call_id: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_call_item", call_id=call_id, tool_name=name)


def _output(call_id: str, output: object) -> SimpleNamespace:
    return SimpleNamespace(type="tool_call_output_item", call_id=call_id, output=output)


SEARCH = {
    "ok": True,
    "results": [
        {"policy_id": "cw-escalations", "score": 2.5},
        {"policy_id": "cw-account-security", "score": 9.5},
        {"policy_id": "cw-cancellations", "score": 0.0},
    ],
}


def _run(reply: str, ids: list[str], rewritten: str) -> tuple[tuple[str, str], list[str]]:
    requests: list[str] = []

    async def rewrite(request: str) -> str:
        requests.append(request)
        return rewritten

    return asyncio.run(citations.enforce_citation(reply, ids, rewrite)), requests


def test_policy_ids_get_policy_first_then_matched_search_by_score() -> None:
    items = [
        _call("a", "search_help_center"),
        _output("a", SEARCH),
        _call("b", "get_policy"),
        _output("b", {"ok": True, "policy_id": "cw-refunds", "body": "..."}),
    ]
    assert citations.policy_ids_used(items) == [
        "cw-refunds",
        "cw-account-security",
        "cw-escalations",
    ]


def test_policy_ids_ignore_failures_unmatched_search_and_other_tools() -> None:
    items = [
        _call("a", "get_policy"),
        _output("a", {"ok": False, "error": "not_found"}),
        _call("b", "search_help_center"),
        _output("b", {"ok": True, "results": [{"policy_id": "cw-disputes", "score": 0.0}]}),
        _call("c", "get_order"),
        _output("c", {"ok": True, "order": {"order_id": 4127}}),
    ]
    assert citations.policy_ids_used(items) == []


def test_policy_ids_parse_string_outputs() -> None:
    items = [_call("a", "get_policy"), _output("a", str({"ok": True, "policy_id": "cw-returns"}))]
    assert citations.policy_ids_used(items) == ["cw-returns"]


def test_cites_any_matches_whole_ids_only() -> None:
    assert citations.cites_any("See (cw-returns).", ["cw-returns"])
    assert not citations.cites_any("See cw-returns-extra.", ["cw-returns"])
    assert not citations.cites_any("No citation here.", ["cw-returns"])


def test_no_policy_used_or_already_cited_needs_no_rewrite() -> None:
    (result, requests) = _run("Hello.", [], "unused")
    assert result == ("Hello.", "not_needed") and requests == []
    (result, requests) = _run("Use settings (cw-account-security).", ["cw-account-security"], "x")
    assert result[1] == "cited" and requests == []


def test_missing_citation_is_rewritten_once() -> None:
    (result, requests) = _run(
        "Use account settings.", ["cw-account-security"], "Use settings (cw-account-security)."
    )
    assert result == ("Use settings (cw-account-security).", "rewritten")
    assert len(requests) == 1 and "cw-account-security" in requests[0]


def test_failed_rewrite_appends_the_first_id() -> None:
    (result, requests) = _run(
        "Use account settings.", ["cw-account-security", "cw-escalations"], "Still no id."
    )
    assert result == ("Use account settings.\n\n(Source: cw-account-security)", "appended")
    assert len(requests) == 1
