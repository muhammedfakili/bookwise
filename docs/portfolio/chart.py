"""Portfolio image: measured results, graph agent vs baseline, read from eval/runs/*.json.

    uv run python docs/portfolio/chart.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "eval" / "runs"
OUT = ROOT / "docs" / "portfolio" / "05-measured-results.png"
CRM = "CRM fields right\n(booked leads)"


def runs(agent: str) -> list[dict]:
    found = sorted(RUNS.glob(f"{agent}-r*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in found if "partial" not in p.name]


def verdict(r: dict) -> str | None:
    """Same rule as eval.run_eval.Result.verdict."""
    if r["booked_start"] or r["lead_status"] in ("qualified", "booked"):
        return "qualified"
    if "disqualified" in r["decisions"] or r["lead_status"] == "disqualified":
        return "disqualified"
    return None


def pooled(agent: str) -> dict:
    """Pool every scored conversation of every run (so 2 runs of 30 = 60 conversations)."""
    results = [r for run in runs(agent) for r in run["results"] if not r["error"]]
    should_book = [r for r in results if r["expected"] == "booked"]
    qual_cases = [r for r in results if r["expected"] in ("booked", "disqualified")]
    predicted = [r for r in results if r["handoff_reason"]]  # any handoff = work for the team
    expected_h = [r for r in results if r["expected"] == "handoff"]
    booked = [r for r in should_book if r["booked_start"]]
    return {
        "n": len(results), "runs": len(runs(agent)),
        "Correct outcome": 100 * sum(r["outcome"] == r["expected"] for r in results) / len(results),
        "Qualified correctly": 100 * sum(verdict(r) == ("qualified" if r["expected"] == "booked" else "disqualified")
                                         for r in qual_cases) / len(qual_cases),
        "Booked when it should": 100 * sum(r["outcome"] == "booked" for r in should_book) / len(should_book),
        "Handoff precision": 100 * sum(r["expected"] == "handoff" for r in predicted) / max(len(predicted), 1),
        "Handoff recall": 100 * sum(bool(r["handoff_reason"]) for r in expected_h) / len(expected_h),
        # Of the leads it booked: how many CRM fields hold exactly what the customer said.
        CRM: 100 * sum(r["crm_fields_ok"] for r in booked) / sum(r["crm_fields_total"] for r in booked),
        "wrong": sum(1 for r in results if r["booked_start"] and (r["expected"] != "booked" or r["slot_ok"] is False)),
    }


graph, base = pooled("graph"), pooled("baseline")
metrics = ["Correct outcome", "Qualified correctly", "Booked when it should", "Handoff precision",
           "Handoff recall", CRM]

BG, FG, MUTED = "#0b0d10", "#f2f6fa", "#8fa0b4"
BLUE, GREY = "#2f7fd4", "#5d6b7c"
fig, ax = plt.subplots(figsize=(15.68, 6.98), dpi=100)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
width = 0.38
for i, m in enumerate(metrics):
    for offset, data, color in ((-width / 2, base, GREY), (width / 2, graph, BLUE)):
        v = data[m]
        ax.bar(i + offset, v, width * 0.94, color=color)
        ax.text(i + offset, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", color=FG, fontsize=16, fontweight="bold")
ax.set_xticks(range(len(metrics)), metrics, color=FG, fontsize=13)
ax.set_ylim(0, 115)
ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=11)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
ax.bar(0, 0, color=GREY, label=f"One prompt + free tool calling  ·  {base['wrong']} wrong bookings")
ax.bar(0, 0, color=BLUE, label=f"LeadPilot: LangGraph, rules in code  ·  {graph['wrong']} wrong bookings")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.13), frameon=False, labelcolor=FG, fontsize=13, ncol=2)
fig.suptitle("30 simulated customers, same model, two agent designs", color=FG, fontsize=22, fontweight="bold",
             x=0.06, ha="left", y=0.96)
fig.text(0.06, 0.875, f"{graph['n']} conversations per agent · 12 should book, 10 should be declined, 8 should "
         "reach a person · agent: Gemma 4 26B, customers: Gemini 3.5 Flash-Lite (both free tier)",
         color=MUTED, fontsize=12, ha="left")
fig.subplots_adjust(top=0.78, bottom=0.1, left=0.06, right=0.98)
fig.savefig(OUT, facecolor=BG)
print("saved", OUT, {"graph": graph, "baseline": base})
