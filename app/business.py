"""The fictional business the agent works for, and its qualification rules.

Brightline Renovations remodels kitchens and bathrooms in the Austin, TX area. The
rules below are the whole sales policy: the graph agent enforces them in code, the
baseline agent only reads them in its prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

NAME = "Brightline Renovations"
ASSISTANT_NAME = "Ava"

SERVICES = ("kitchen", "bathroom", "kitchen_and_bathroom")
SERVICE_LABELS = {"kitchen": "kitchen remodel", "bathroom": "bathroom remodel",
                  "kitchen_and_bathroom": "kitchen and bathroom remodel"}
MIN_BUDGET_USD = {"kitchen": 25_000, "bathroom": 12_000, "kitchen_and_bathroom": 35_000}
MAX_START_WEEKS = 12  # we book consultations for projects that start within ~3 months
SERVICE_ZIP_PREFIXES = ("786", "787")  # Austin metro (Travis, Williamson, Hays counties)

# Free in-home design consultations: 90 minutes, weekdays.
SLOT_TIMES = (time(9, 0), time(11, 0), time(13, 0), time(15, 0))
SLOT_MINUTES = 90
BOOKING_HORIZON_DAYS = 21

FACTS = f"""\
{NAME} -- kitchen and bathroom remodeling in the Austin, TX metro area (ZIP codes 786xx and 787xx).
- Services: full kitchen remodels, full bathroom remodels, or both. We do NOT do roofing, \
additions, pools, landscaping, HVAC, or small handyman repairs.
- Typical project cost: kitchens from $25,000, bathrooms from $12,000, both from $35,000.
- First step: a free 90-minute in-home design consultation with a project designer, \
weekdays 9:00-16:30. No obligation.
- Licensed and insured; 5-year workmanship warranty; financing available through a partner lender.
- Typical timeline: kitchens 6-10 weeks of work, bathrooms 3-5 weeks, after design approval.
- Office: Mon-Fri 8:00-17:00, phone (512) 555-0142. Exact prices, discounts and contract \
questions are handled by the team, not the assistant."""


@dataclass(frozen=True)
class Verdict:
    status: str  # "qualified" | "disqualified" | "incomplete"
    reason: str = ""
    missing: tuple[str, ...] = ()


QUALIFY_ORDER = ("service", "zip_code", "timeline_weeks", "budget_usd", "decision_maker")


def qualify(lead: dict) -> Verdict:
    """Apply the sales policy to what we know so far. Pure function, easy to test.

    A known fact that breaks a rule disqualifies at once; otherwise every missing
    fact is listed in the order the agent should ask for it.
    """
    service = lead.get("service")
    if service == "other":
        return Verdict("disqualified", f"service not offered ({lead.get('other_service') or 'other'})")
    zip_code = lead.get("zip_code")
    if zip_code and not str(zip_code).startswith(SERVICE_ZIP_PREFIXES):
        return Verdict("disqualified", f"outside service area (ZIP {zip_code})")
    weeks = lead.get("timeline_weeks")
    if weeks is not None and weeks > MAX_START_WEEKS:
        return Verdict("disqualified", f"start date too far out ({weeks} weeks)")
    budget = lead.get("budget_usd")
    if budget is not None and service in MIN_BUDGET_USD and budget < MIN_BUDGET_USD[service]:
        return Verdict("disqualified",
                       f"budget ${budget:,} below ${MIN_BUDGET_USD[service]:,} minimum for {SERVICE_LABELS[service]}")
    if lead.get("decision_maker") is False:
        return Verdict("disqualified", "not the property owner / decision maker")

    missing = tuple(f for f in QUALIFY_ORDER if lead.get(f) is None)
    if missing:
        return Verdict("incomplete", missing=missing)
    return Verdict("qualified")
