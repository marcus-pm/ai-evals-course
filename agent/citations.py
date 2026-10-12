"""Citation guard for RESP-1: a reply that uses a policy must cite its id.

The prompt asks the model to cite policy ids, but a prompt rule is not
enforcement. This guard runs after each turn, outside the model:

1. Collect the policy ids this turn's tools returned: every id from a
   successful get_policy, then search_help_center results that matched
   (score > 0), best match first.
2. If the reply contains none of those ids, ask the model once to rewrite
   the reply with the id.
3. If the rewrite still cites none, append "(Source: <id>)" with the first
   collected id.

It is not an SDK output guardrail, so the Module 1 agent keeps no guardrails
and the Module 4 guard holes in agent/guards.py stay untouched. Callers (the
CLI today, the server later) run it on each turn's result.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from agents.items import RunItem


def policy_ids_used(new_items: list[RunItem]) -> list[str]:
    """Policy ids returned by this turn's tools, get_policy first, no duplicates."""
    tool_names = {
        item.call_id: item.tool_name
        for item in new_items
        if item.type == "tool_call_item"
    }
    fetched: list[str] = []
    searched: list[tuple[float, str]] = []
    for item in new_items:
        if item.type != "tool_call_output_item":
            continue
        output = _as_dict(item.output)
        if not output or output.get("ok") is not True:
            continue
        name = tool_names.get(item.call_id)
        if name == "get_policy" and output.get("policy_id"):
            fetched.append(output["policy_id"])
        elif name == "search_help_center":
            for hit in output.get("results", []):
                if hit.get("score", 0) > 0 and hit.get("policy_id"):
                    searched.append((hit["score"], hit["policy_id"]))
    ordered = fetched + [pid for _, pid in sorted(searched, key=lambda s: -s[0])]
    return list(dict.fromkeys(ordered))


def cites_any(reply: str, policy_ids: list[str]) -> bool:
    """True when the reply contains at least one of the ids as a whole token."""
    return any(
        re.search(rf"(?<![\w-]){re.escape(pid)}(?![\w-])", reply) for pid in policy_ids
    )


def rewrite_request(policy_ids: list[str]) -> str:
    """The one-time instruction sent to the model when a citation is missing."""
    ids = ", ".join(policy_ids)
    return (
        "Citation check: your last reply used Cartwheel policy information but did "
        f"not cite a policy id. Rewrite that reply for the user with the relevant "
        f"policy id in parentheses (available ids: {ids}). Do not call tools. "
        "Reply with the rewritten message only."
    )


async def enforce_citation(
    reply: str,
    policy_ids: list[str],
    rewrite: Callable[[str], Awaitable[str]],
) -> tuple[str, str]:
    """Return (final reply, outcome).

    Outcome is "not_needed" (no policy used), "cited" (already cited),
    "rewritten" (the model's rewrite cites an id), or "appended" (the source
    line was added).
    """
    if not policy_ids:
        return reply, "not_needed"
    if cites_any(reply, policy_ids):
        return reply, "cited"
    rewritten = await rewrite(rewrite_request(policy_ids))
    if cites_any(rewritten, policy_ids):
        return rewritten, "rewritten"
    return f"{reply.rstrip()}\n\n(Source: {policy_ids[0]})", "appended"


def _as_dict(output: Any) -> dict[str, Any] | None:
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                value = parse(output)
            except (ValueError, SyntaxError):
                continue
            if isinstance(value, dict):
                return value
    return None
