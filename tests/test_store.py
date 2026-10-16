from datetime import datetime

import pytest

from app.ports import SlotUnavailable


def test_free_slots_are_future_weekdays_on_different_days(make_deps):
    cal = make_deps().calendar
    slots = cal.find_free_slots(3)
    assert len(slots) == 3
    assert len({s.start.date() for s in slots}) == 3
    assert all(s.start > datetime(2026, 10, 19, 9) and s.start.weekday() < 5 for s in slots)


def test_seeded_busy_slots_are_never_offered(make_deps):
    cal = make_deps().calendar
    taken = {a["start"] for a in cal.appointments(include_existing=True)}
    assert taken, "the demo calendar should start partly booked"
    offered = {s.start.isoformat() for s in cal.find_free_slots(50)}
    assert not offered & taken


@pytest.mark.parametrize(("part", "check"), [("morning", lambda h: h < 12), ("afternoon", lambda h: h >= 12)])
def test_part_of_day_filter(make_deps, part, check):
    slots = make_deps().calendar.find_free_slots(5, part_of_day=part)
    assert slots
    assert all(check(s.start.hour) for s in slots)


def test_weekday_filter(make_deps):
    slots = make_deps().calendar.find_free_slots(3, weekdays=["Thursday"])
    assert slots
    assert {s.start.strftime("%A") for s in slots} == {"Thursday"}


def test_double_booking_is_refused_by_the_database(make_deps):
    cal = make_deps().calendar
    slot = cal.find_free_slots(1)[0]
    cal.book(slot.start, "c1", "Ann", "5125550101")
    with pytest.raises(SlotUnavailable, match="already booked"):
        cal.book(slot.start, "c2", "Bob", "5125550102")
    assert slot.start not in [s.start for s in cal.find_free_slots(50)]


def test_cancelled_slot_can_be_booked_again(make_deps):
    cal = make_deps().calendar
    slot = cal.find_free_slots(1)[0]
    appt = cal.book(slot.start, "c1", "Ann", "5125550101")
    cal.cancel(appt.id)
    assert cal.book(slot.start, "c2", "Bob", "5125550102").id != appt.id


@pytest.mark.parametrize("start", [datetime(2026, 10, 20, 10, 0), datetime(2026, 10, 24, 9, 0),
                                   datetime(2026, 10, 1, 9, 0)])
def test_off_grid_weekend_or_past_times_are_refused(make_deps, start):
    with pytest.raises(SlotUnavailable, match="not a bookable"):
        make_deps().calendar.book(start, "c1", "Ann", "5125550101")


def test_crm_upsert_merges_and_rejects_unknown_fields(make_deps):
    crm = make_deps().crm
    crm.upsert_lead("c1", "web", {"service": "kitchen", "decision_maker": True})
    lead = crm.upsert_lead("c1", "web", {"budget_usd": 30_000, "status": "in_progress"})
    assert (lead["service"], lead["budget_usd"], lead["decision_maker"]) == ("kitchen", 30_000, True)
    with pytest.raises(ValueError, match="unknown lead fields"):
        crm.upsert_lead("c1", "web", {"password": "x"})


def test_one_open_handoff_per_conversation(make_deps):
    handoffs = make_deps().handoffs
    first = handoffs.open("c1", "complaint", "s")
    assert handoffs.open("c1", "again", "s") == first
    handoffs.resolve(first)
    assert handoffs.active("c1") is None
