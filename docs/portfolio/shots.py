"""Portfolio images for LeadPilot: cover + screens of the running app (after demo_seed.py).

Needs the API and UI running locally (see demo_seed.py for the API command).
Run:  uv run --with playwright python docs/portfolio/shots.py
"""

import json
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "portfolio"
URL = "http://localhost:8501"
W, H = 1568, 698  # same size as the other portfolio screenshots
IDS = json.loads((OUT / "demo_ids.json").read_text(encoding="utf-8"))

HIDE = """
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {display:none !important;}
"""


def settle(page: Page, ms: int = 3000) -> None:
    page.wait_for_timeout(ms)
    page.add_style_tag(content=HIDE)


def scroll_to_bottom(page: Page) -> None:
    page.mouse.move(500, 400)
    page.mouse.wheel(0, 5000)
    page.wait_for_timeout(800)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")

        cover = browser.new_page(viewport={"width": 1000, "height": 800})
        cover.goto((OUT / "cover.html").as_uri())
        cover.locator(".card").screenshot(path=str(OUT / "00-cover.png"))

        page = browser.new_context(viewport={"width": W, "height": H}, color_scheme="dark").new_page()

        # 1. a qualified lead booked: the chat and what the agent wrote to the CRM
        page.goto(f"{URL}/?c={IDS['booked']}")
        page.get_by_text("Consultation booked").wait_for(timeout=60_000)
        settle(page)
        scroll_to_bottom(page)
        page.screenshot(path=str(OUT / "01-qualified-and-booked.png"))

        # 2. outside the service area: declined politely, never offered a slot
        page.goto(f"{URL}/?c={IDS['declined']}")
        page.get_by_text("What the agent knows").wait_for(timeout=60_000)
        settle(page)
        page.screenshot(path=str(OUT / "02-declined-outside-area.png"))

        # 3. a complaint handed to a person, who answers from the team inbox
        page.goto(f"{URL}/admin")
        page.get_by_text("Waiting for a person").wait_for(timeout=60_000)
        page.get_by_text("Conversation").first.click()
        settle(page, 2000)
        page.screenshot(path=str(OUT / "03-handoff-team-inbox.png"))

        # 4. every lead with its status and the reason, as the CRM sees it
        page.get_by_text("Leads", exact=True).click()
        settle(page, 2000)
        page.screenshot(path=str(OUT / "04-leads-crm.png"))

        browser.close()
    print("saved to", OUT)


if __name__ == "__main__":
    main()
