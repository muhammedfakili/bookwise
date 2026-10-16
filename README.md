# LeadPilot — a sales chat agent that qualifies leads, books consultations and knows when to call a human

A prospect writes in the web chat (or on WhatsApp). The agent asks the qualifying questions a good salesperson would ask (what project, where, when, what budget, who decides), declines politely when the lead doesn't fit, books a free consultation when it does, writes everything to the CRM, and hands the conversation to a person when it should: complaints, angry customers, discount or contract questions, reschedules, or when it can't follow.

![Qualified and booked](docs/portfolio/01-qualified-and-booked.png)

> The business, Brightline Renovations (kitchen and bathroom remodeling, Austin TX), is fictional. Calendar, CRM and WhatsApp are mocks behind clean interfaces, so Google Calendar, HubSpot or the WhatsApp Cloud API can be plugged in without touching the agent.

## Results

<!-- RESULTS -->

## How it works

The main design choice: **the LLM reads and writes, code decides.** A LangGraph state machine runs once per customer message:

```
customer message (web chat / WhatsApp webhook)
   │
   ▼
understand ── LLM with structured output: intent, sentiment, confidence and every fact
              in the message (service, ZIP, budget, start date, owner?, name, phone, chosen time)
   │
   ▼
decide ─────── plain Python, unit tested:
   │           1. handoff rules (complaint, asks for a person, quote/discount/contract/warranty,
   │              reschedule, angry, two unclear messages in a row)
   │           2. sales policy (app/business.py): service offered? ZIP in area? start within
   │              12 weeks? budget ≥ minimum for that service? owner who decides?
   │           3. next missing fact, or contact details, or a time
   ▼
one action ─── ask │ offer_slots │ book │ disqualify │ handoff │ follow_up
   │           (only these nodes touch the calendar, the CRM and the handoff queue)
   ▼
respond ────── LLM writes the message, told exactly what to say
```

So the model can't book a lead that failed the rules, invent a free slot, or skip a handoff, however the customer phrases things (prompt injection included). The database refuses double bookings on its own (a partial unique index). The graph state — facts so far, slots on the table, retry counters — is checkpointed in SQLite per conversation, so a restart doesn't lose a half-qualified lead.

For comparison, the repo also has the way most booking-agent demos are built: **one prompt with the rules + free tool calling** (`app/agents/baseline_agent.py`). It uses the same calendar, CRM and handoff tools and the same model; only the design differs. The evaluation runs both against the same simulated customers.

| Part | File |
|---|---|
| Business rules (the whole sales policy, pure function) | `app/business.py` |
| LangGraph agent: nodes, `decide()`, slot matching | `app/agents/graph_agent.py` |
| Baseline agent: one prompt + native tool calling | `app/agents/baseline_agent.py` |
| Ports (Calendar, CRM, Handoff, Channel) | `app/ports.py` |
| SQLite implementations + transcripts | `app/store.py` |
| Web chat / WhatsApp entry point, human takeover | `app/service.py`, `app/channels.py` |
| HTTP API (FastAPI) | `app/api.py` |
| Chat screen + team inbox (Streamlit, talks only to the API) | `ui/` |
| LLM providers: Gemini, Claude, NVIDIA NIM | `app/llm.py` |
| Simulated-customer evaluation | `eval/personas.py`, `eval/run_eval.py` |

### Human handoff

When the agent hands off, it opens a ticket with the reason and what it knows, tells the customer a person will reply in the same chat, and **stops answering**. The team sees the queue in the inbox, replies inside the conversation (on WhatsApp the reply goes out through the channel), and gives it back to the bot when done.

![Handoff in the team inbox](docs/portfolio/03-handoff-team-inbox.png)

### WhatsApp-ready

`POST /webhook/whatsapp` accepts the WhatsApp Cloud API webhook format (and `GET` answers Meta's verification handshake). Replies go through a `ChannelPort`: the mock keeps the exact JSON it would POST to `/{phone-number-id}/messages`; `WhatsAppCloud` in `app/channels.py` sends it for real once a Meta app, phone number id and token exist (not tested against Meta here).

## Run it

You need Docker and one free API key: Gemini (https://aistudio.google.com/apikey, 500 requests/day) or NVIDIA NIM (https://build.nvidia.com, no daily cap).

```bash
echo GOOGLE_API_KEY=your-key > .env          # or: LLM_PROVIDER=nvidia + NVIDIA_API_KEY=...
docker compose up -d --build
```

Open http://localhost:8501 (customer chat + team inbox). API docs: http://localhost:8000/docs.

Without Docker (Python 3.12 + [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
uv run uvicorn app.api:app --port 8000
uv run streamlit run ui/app.py
```

Settings (`.env`): `LLM_PROVIDER` (`gemini` | `nvidia` | `anthropic`), `LLM_MODEL`, `AGENT` (`graph` | `baseline`), `DB_PATH`, `CLOCK` (freeze time for demos), `APP_API_KEY` (protects the API; the UI sends it), `WHATSAPP_VERIFY_TOKEN`.

```bash
curl -X POST localhost:8000/chat -H "content-type: application/json" \
     -d '{"message": "Hi, we want a new kitchen in 78704"}'
```

### Reproduce the numbers

```bash
uv run python -m eval.run_eval --agent graph --tag r1
uv run python -m eval.run_eval --agent baseline --tag r1
uv run python -m eval.run_eval --agent graph --tag r2      # a second, independent run
uv run python docs/portfolio/chart.py
uv run pytest                                              # no API key needed
```

<!-- LIMITS -->
