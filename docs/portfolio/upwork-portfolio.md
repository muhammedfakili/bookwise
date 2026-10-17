# Upwork portfolio entry — LeadPilot

Same layout as the other projects: title, role, 5 skills, one paragraph (600-character limit), cover + images.

**Title** (62 characters)
LeadPilot — AI Sales Chat Agent: Qualify Leads, Book, Hand Off

**Role**
AI Engineer

**Skills**
AI Agent Development · LangChain · Python · FastAPI · Chatbot Development

**Description** (480 / 600 characters)
A chat agent for a remodeling company (web chat, WhatsApp-ready): asks the qualifying questions, declines leads that don't fit, books a free consultation, writes the CRM and hands complaints or discount requests to a person. Built with LangGraph: the model reads, code decides. Tested on 30 simulated customers against a one-prompt tool-calling agent: 28/30 vs 30/30 correct, 0 wrong bookings each, booked leads' CRM data 97% vs 79% complete. FastAPI, Streamlit, Docker, 98 tests.

**Images, in this order**
| File | Shows |
|---|---|
| `00-cover.png` | Cover: black background, robot, "AI SALES / CHAT AGENT: / QUALIFY & BOOK" |
| `01-lead-card-booked.png` | The chat and what the agent collected: project, ZIP, start, budget, owner, name, phone; status booked |
| `02-booking-afternoon.png` | Customer prefers afternoons: new times offered, Thursday 3 PM booked and confirmed |
| `03-declined-outside-area.png` | Dallas ZIP: declined politely, reason saved, no slot offered |
| `04-handoff-team-inbox.png` | A complaint handed to a person; the team member answers inside the conversation |
| `05-measured-results.png` | Measured results: graph agent vs one-prompt tool-calling agent on 30 simulated customers |
| `06-leads-crm.png` | Every lead with status and reason, as the CRM sees it |

Every number comes from `eval/results/graph-r1.md` and `eval/results/baseline-r1.md`.
Images are reproducible: `demo_seed.py` + `shots.py` (Playwright against the running app) and `chart.py` in this folder.
The business (Brightline Renovations) and all customers are fictional.
