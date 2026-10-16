import pytest

from app.business import qualify

GOOD = {"service": "kitchen", "zip_code": "78704", "timeline_weeks": 6, "budget_usd": 40_000,
        "decision_maker": True}


def test_complete_good_lead_qualifies():
    assert qualify(GOOD).status == "qualified"


@pytest.mark.parametrize(("change", "reason"), [
    ({"service": "other", "other_service": "roof"}, "service not offered (roof)"),
    ({"zip_code": "73301"}, "outside service area"),
    ({"zip_code": "90210"}, "outside service area"),
    ({"timeline_weeks": 30}, "start date too far out"),
    ({"budget_usd": 20_000}, "below $25,000 minimum"),
    ({"service": "bathroom", "budget_usd": 9_000}, "below $12,000 minimum"),
    ({"service": "kitchen_and_bathroom", "budget_usd": 30_000}, "below $35,000"),
    ({"decision_maker": False}, "not the property owner"),
])
def test_each_rule_disqualifies(change, reason):
    verdict = qualify({**GOOD, **change})
    assert verdict.status == "disqualified"
    assert reason in verdict.reason


def test_bathroom_budget_uses_its_own_minimum():
    assert qualify({**GOOD, "service": "bathroom", "budget_usd": 12_000}).status == "qualified"


def test_missing_facts_listed_in_asking_order():
    verdict = qualify({"budget_usd": 50_000})
    assert verdict.status == "incomplete"
    assert verdict.missing == ("service", "zip_code", "timeline_weeks", "decision_maker")


def test_a_known_bad_fact_wins_over_missing_ones():
    assert qualify({"zip_code": "10001"}).status == "disqualified"


def test_budget_cannot_be_judged_before_the_service_is_known():
    assert qualify({"budget_usd": 5_000}).status == "incomplete"
