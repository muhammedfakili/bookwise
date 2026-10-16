"""The customer's side: a web chat, with a live view of what the agent knows and did."""

from __future__ import annotations

import streamlit as st

from ui.client import ApiClient, ApiError

STATUS_COLORS = {"new": "gray", "in_progress": "blue", "qualified": "violet", "booked": "green",
                 "disqualified": "orange", "handoff": "red"}
FIELDS = [("service", "Project"), ("zip_code", "ZIP"), ("timeline_weeks", "Start (weeks)"),
          ("budget_usd", "Budget"), ("decision_maker", "Owner / decides"), ("name", "Name"), ("phone", "Phone")]

api = ApiClient()
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
    st.session_state.last_turn = None

chat_col, info_col = st.columns([3, 2], gap="large")

with info_col, st.container(horizontal=True, vertical_alignment="center"):
    st.subheader("What the agent knows", anchor=False)
    if st.button("New chat", icon=":material/add_comment:"):
        st.session_state.conversation_id = None
        st.session_state.last_turn = None
        st.rerun()

conversation = None
if st.session_state.conversation_id:
    try:
        conversation = api.conversation(st.session_state.conversation_id)
    except ApiError as exc:
        st.error(str(exc))

with chat_col:
    st.title("Brightline Renovations", anchor=False)
    st.caption("Kitchen & bathroom remodeling, Austin TX. Chat with Ava, our assistant.")
    messages = conversation["messages"] if conversation else []
    if not messages:
        with st.chat_message("assistant", avatar=":material/support_agent:"):
            st.write("Hi! Tell me about the project you have in mind.")
    for m in messages:
        if m["role"] == "customer":
            st.chat_message("user").write(m["text"])
        elif m["role"] == "human_agent":
            st.chat_message("Team", avatar=":material/person:").write(f"**Team member:** {m['text']}")
        else:
            st.chat_message("assistant", avatar=":material/support_agent:").write(m["text"])
    if conversation and conversation["mode"] == "human":
        st.info("A team member has taken over this conversation. The assistant is paused.",
                icon=":material/headset_mic:")

prompt = st.chat_input("Type your message", submit_mode="disable")
if prompt:
    with chat_col:
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant", avatar=":material/support_agent:"), st.spinner("Ava is typing..."):
            try:
                turn = api.chat(prompt, st.session_state.conversation_id)
            except ApiError as exc:
                st.error(str(exc))
                st.stop()
    st.session_state.conversation_id = turn["conversation_id"]
    st.session_state.last_turn = turn
    st.rerun()

with info_col:
    lead = (conversation or {}).get("lead")
    if not lead:
        st.caption("Nothing yet. Start chatting: the agent fills this in as it learns about the project.")
    else:
        status = lead["status"]
        st.badge(status.replace("_", " "), color=STATUS_COLORS.get(status, "gray"))
        if lead.get("status_reason"):
            st.caption(lead["status_reason"])
        rows = []
        for key, label in FIELDS:
            value = lead.get(key)
            if key == "budget_usd" and value is not None:
                value = f"${value:,}"
            if key == "decision_maker" and value is not None:
                value = "yes" if value else "no"
            rows.append({"Field": label, "Value": "—" if value is None else str(value)})
        st.dataframe(rows, hide_index=True, alt="Lead fields collected so far")

        for appt in conversation.get("appointments", []):
            st.success(f"Consultation booked: {appt['start'].replace('T', ' ')[:16]}", icon=":material/event:")
        for h in conversation.get("handoffs", []):
            st.warning(f"Handoff ({h['status']}): {h['reason']}", icon=":material/headset_mic:")

        last = st.session_state.last_turn
        if last and last.get("events"):
            st.markdown(f"**Last turn** · action `{last['action']}` · {last['llm_calls']} model calls")
            for event in last["events"]:
                st.status(event, state="complete", type="step")
