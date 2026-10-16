"""LeadPilot screens: the customer chat and the team's admin view.

    uv run streamlit run ui/app.py     (needs the API: uv run uvicorn app.api:app --port 8000)
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# `streamlit run ui/app.py` puts ui/ on the path; the pages import `ui.client`, so add the root too.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="LeadPilot", page_icon=":material/support_agent:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/chat.py", title="Customer chat", icon=":material/chat:", default=True),
        st.Page("app_pages/admin.py", title="Team inbox", icon=":material/inbox:"),
    ],
    position="top",
)
page.run()
