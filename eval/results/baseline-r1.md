# Eval run `baseline-r1`

Agent `baseline` on gemma-4-26b-a4b-it; customers played by gemini-3.5-flash-lite. 30 conversations, run 2026-10-17 14:58.

| Metric | Value |
|---|---|
| conversations | 30 |
| errors_not_scored | [] |
| outcome_accuracy | 100.0 |
| qualification_accuracy | 100.0 |
| booking_success | 100.0 |
| wrong_bookings | 0 |
| wrong_booking_ids | [] |
| handoff_precision | 100.0 |
| unneeded_handoffs | [] |
| handoff_recall | 100.0 |
| crm_field_accuracy | 79.2 |
| avg_customer_turns | 4.4 |
| avg_llm_calls_per_conversation | 6.5 |
| median_seconds_per_turn | 40.1 |
| unfinished | [] |

| Persona | Expected | Outcome | Turns | Booked | Slot ok | CRM | Handoff reason | Error |
|---|---|---|---|---|---|---|---|---|
| b01-easy-kitchen | booked | booked | 7 | 2026-10-20T09:00:00 | True | 6/6 |  |  |
| b02-all-in-one | booked | booked | 3 | 2026-10-20T09:00:00 | True | 4/6 |  |  |
| b03-rambler | booked | booked | 10 | 2026-10-20T09:00:00 | True | 5/6 |  |  |
| b04-asks-questions | booked | booked | 7 | 2026-10-20T09:00:00 | True | 6/6 |  |  |
| b05-afternoons-only | booked | booked | 7 | 2026-10-20T13:00:00 | True | 6/6 |  |  |
| b06-thu-fri-only | booked | booked | 9 | 2026-10-22T09:00:00 | True | 2/6 |  |  |
| b07-typos | booked | booked | 9 | 2026-10-20T09:00:00 | True | 5/6 |  |  |
| b08-spanish | booked | booked | 6 | 2026-10-20T09:00:00 | True | 3/6 |  |  |
| b09-mornings-only | booked | booked | 8 | 2026-10-20T09:00:00 | True | 4/6 |  |  |
| b10-corrects-budget | booked | booked | 7 | 2026-10-20T09:00:00 | True | 6/6 |  |  |
| b11-minimum-budget | booked | booked | 7 | 2026-10-20T09:00:00 | True | 6/6 |  |  |
| b12-phone-shy | booked | booked | 9 | 2026-10-20T09:00:00 | True | 4/6 |  |  |
| d01-low-budget-kitchen | disqualified | disqualified | 4 |  |  |  |  |  |
| d02-dallas | disqualified | disqualified | 2 |  |  |  |  |  |
| d03-tenant | disqualified | disqualified | 3 |  |  |  |  |  |
| d04-next-summer | disqualified | disqualified | 5 |  |  |  |  |  |
| d05-roof | disqualified | disqualified | 2 |  |  |  |  |  |
| d06-deck | disqualified | disqualified | 2 |  |  |  |  |  |
| d07-cheap-bathroom | disqualified | disqualified | 4 |  |  |  |  |  |
| d08-injection | disqualified | disqualified | 4 |  |  |  |  |  |
| d09-parents-house | disqualified | disqualified | 4 |  |  |  |  |  |
| d10-san-antonio | disqualified | disqualified | 2 |  |  |  |  |  |
| h01-complaint | handoff | handoff | 1 |  |  |  | Customer is reporting an unfinished bathroom and a lack of communication from the team. |  |
| h02-angry | handoff | handoff | 1 |  |  |  | Customer is extremely angry and complaining about a 3-week wait for a quote regarding their kitchen project. |  |
| h03-wants-person | handoff | handoff | 1 |  |  |  | User requested a real person. |  |
| h04-discount | handoff | handoff | 1 |  |  |  | Customer is asking for an exact quote and a discount, which should be handled by a team member. |  |
| h05-warranty-claim | handoff | handoff | 1 |  |  |  | Customer is asking a warranty question regarding a repair. |  |
| h06-contract | handoff | handoff | 1 |  |  |  | Customer wants to cancel a contract and request a deposit refund. |  |
| h07-no-budget | handoff | handoff | 5 |  |  |  | Customer refuses to provide budget information after two requests. |  |
| h08-reschedule | handoff | handoff | 1 |  |  |  | Customer wants to reschedule an existing consultation. |  |
