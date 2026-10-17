"""The team's side: handoff queue (reply and give back to the bot), leads, bookings, WhatsApp outbox."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.client import ApiClient, ApiError

api = ApiClient()

st.title("Team inbox", anchor=False)
try:
    leads = api.admin("leads")
    appointments = api.admin("appointments")
    handoffs = api.admin("handoffs")
    outbox = api.admin("whatsapp/outbox")
except ApiError as exc:
    st.error(str(exc), title="The API is not answering")
    st.stop()

open_handoffs = [h for h in handoffs if h["status"] == "open"]
with st.container(horizontal=True):
    st.metric("Leads", len(leads), border=True)
    st.metric("Qualified or booked", sum(lead["status"] in ("qualified", "booked") for lead in leads), border=True)
    st.metric("Consultations booked", len(appointments), border=True)
    st.metric("Waiting for a person", len(open_handoffs), border=True)

view = st.segmented_control("View", ["Handoffs", "Leads", "Bookings", "WhatsApp outbox"], default="Handoffs",
                            label_visibility="collapsed")

if view == "Handoffs":
    if not handoffs:
        st.caption("No handoffs yet. The assistant hands over complaints, angry customers, requests for a person, "
                   "quotes, discounts, contracts, reschedules, and conversations it can't follow.")
    for h in handoffs:
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.badge(h["status"], color="red" if h["status"] == "open" else "gray")
                st.markdown(f"**{h['reason']}** · `{h['conversation_id']}` · {h['created'].replace('T', ' ')}")
            st.caption(h["summary"])
            with st.expander("Conversation"):
                for m in api.conversation(h["conversation_id"])["messages"]:
                    st.markdown(f"**{m['role'].replace('_', ' ')}:** {m['text']}")
            if h["status"] == "open":
                with st.form(f"reply-{h['id']}", clear_on_submit=True, border=False):
                    text = st.text_input("Reply as a team member", key=f"text-{h['id']}")
                    with st.container(horizontal=True):
                        send = st.form_submit_button("Send reply", icon=":material/send:")
                        done = st.form_submit_button("Give back to the assistant", icon=":material/smart_toy:")
                if send and text.strip():
                    api.reply(h["id"], text)
                    st.toast("Reply sent")
                    st.rerun()
                if done:
                    api.resolve(h["id"])
                    st.rerun()

elif view == "Leads":
    if leads:
        df = pd.DataFrame(leads)[["updated", "conversation_id", "channel", "status", "status_reason", "name", "phone",
                                  "service", "zip_code", "budget_usd", "timeline_weeks", "decision_maker"]]
        df["decision_maker"] = df["decision_maker"].map({1: True, 0: False})  # SQLite stores booleans as 0/1
        st.dataframe(df, hide_index=True, alt="All leads", column_config={
            "updated": "Updated", "conversation_id": "Conversation", "channel": "Channel", "status": "Status",
            "status_reason": "Reason", "name": "Name", "phone": "Phone", "service": "Project", "zip_code": "ZIP",
            "budget_usd": st.column_config.NumberColumn("Budget", format="$%d"),
            "timeline_weeks": st.column_config.NumberColumn("Start (weeks)"),
            "decision_maker": st.column_config.CheckboxColumn("Owner"),
        })
    else:
        st.caption("No leads yet.")

elif view == "Bookings":
    if appointments:
        st.dataframe(pd.DataFrame(appointments)[["start", "end", "name", "phone", "conversation_id", "status"]],
                     hide_index=True, alt="Booked consultations")
    else:
        st.caption("No consultations booked yet.")

else:
    st.caption("Replies the mock WhatsApp channel would have POSTed to the Cloud API (/messages).")
    st.json(outbox[-10:] if outbox else [])
