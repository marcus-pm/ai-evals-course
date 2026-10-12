"""HW1 additional tool: validate_offer, tested offline against the seeded
data-quality defects (no LLM, no API keys)."""

from __future__ import annotations

from agent import tools
from agent.agent import TOOLS_BY_ROLE
from agent.auth import AuthContext

SHOPPER_1 = AuthContext(user_id=1, role="shopper")


def _offers(result: dict) -> dict[int, dict]:
    assert result["ok"] is True
    return {offer["product_id"]: offer for offer in result["offers"]}


def test_valid_product_has_no_issues(world: dict) -> None:
    offers = _offers(tools.validate_offer(SHOPPER_1, [1]))
    assert offers[1] == {"product_id": 1, "valid": True, "issues": []}


def test_seeded_defects_are_invalid(world: dict) -> None:
    offers = _offers(tools.validate_offer(SHOPPER_1, [3, 4]))
    assert offers[3]["valid"] is False and offers[3]["issues"] == ["missing_title"]
    assert offers[4]["valid"] is False and offers[4]["issues"] == ["nonpositive_price"]


def test_duplicate_title_is_not_an_issue(world: dict) -> None:
    # Products 1 and 2 share a title; ids keep them distinct.
    offers = _offers(tools.validate_offer(SHOPPER_1, [1, 2]))
    assert offers[1]["valid"] is True and offers[2]["valid"] is True


def test_unknown_product_and_request_order(world: dict) -> None:
    result = tools.validate_offer(SHOPPER_1, [999_999, 4, 1])
    assert [o["product_id"] for o in result["offers"]] == [999_999, 4, 1]
    assert result["offers"][0] == {"product_id": 999_999, "valid": False, "issues": ["not_found"]}


def test_rejects_empty_and_oversized_lists(world: dict) -> None:
    assert tools.validate_offer(SHOPPER_1, [])["error"] == "invalid_argument"
    too_many = list(range(1, tools.MAX_SEARCH_LIMIT + 2))
    assert tools.validate_offer(SHOPPER_1, too_many)["error"] == "invalid_argument"


def test_registered_for_every_role() -> None:
    for role, role_tools in TOOLS_BY_ROLE.items():
        assert "validate_offer" in [tool.name for tool in role_tools], role
