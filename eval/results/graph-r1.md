# Eval run `graph-r1`

Agent `graph` on gemma-4-26b-a4b-it; customers played by gemini-3.5-flash-lite. 30 conversations, run 2026-10-17 14:27.

| Metric | Value |
|---|---|
| conversations | 30 |
| errors_not_scored | [] |
| outcome_accuracy | 93.3 |
| qualification_accuracy | 90.9 |
| booking_success | 83.3 |
| wrong_bookings | 0 |
| wrong_booking_ids | [] |
| handoff_precision | 80.0 |
| unneeded_handoffs | ['b10-corrects-budget', 'b11-minimum-budget'] |
| handoff_recall | 100.0 |
| crm_field_accuracy | 80.6 |
| avg_customer_turns | 3.7 |
| avg_llm_calls_per_conversation | 7.3 |
| median_seconds_per_turn | 44.6 |
| unfinished | [] |

| Persona | Expected | Outcome | Turns | Booked | Slot ok | CRM | Handoff reason | Error |
|---|---|---|---|---|---|---|---|---|
| b01-easy-kitchen | booked | booked | 7 | 2026-10-20T09:00:00 | True | 6/6 |  |  |
| b02-all-in-one | booked | booked | 2 | 2026-10-20T09:00:00 | True | 6/6 |  |  |
| b03-rambler | booked | booked | 7 | 2026-10-20T09:00:00 | True | 5/6 |  |  |
| b04-asks-questions | booked | booked | 5 | 2026-10-21T15:00:00 | True | 5/6 |  |  |
| b05-afternoons-only | booked | booked | 5 | 2026-10-21T13:00:00 | True | 6/6 |  |  |
| b06-thu-fri-only | booked | booked | 8 | 2026-10-22T13:00:00 | True | 6/6 |  |  |
| b07-typos | booked | booked | 9 | 2026-10-20T09:00:00 | True | 6/6 |  |  |
| b08-spanish | booked | booked | 6 | 2026-10-20T09:00:00 | True | 6/6 |  |  |
| b09-mornings-only | booked | booked | 8 | 2026-10-20T09:00:00 | True | 6/6 |  |  |
| b10-corrects-budget | booked | disqualified ❌ | 4 |  |  | 0/6 | request needs staff (quote, discount, contract or warranty) |  |
| b11-minimum-budget | booked | handoff ❌ | 2 |  |  | 0/6 | request needs staff (quote, discount, contract or warranty) |  |
| b12-phone-shy | booked | booked | 8 | 2026-10-20T09:00:00 | True | 6/6 |  |  |
| d01-low-budget-kitchen | disqualified | disqualified | 3 |  |  |  |  |  |
| d02-dallas | disqualified | disqualified | 3 |  |  |  |  |  |
| d03-tenant | disqualified | disqualified | 2 |  |  |  |  |  |
| d04-next-summer | disqualified | disqualified | 4 |  |  |  |  |  |
| d05-roof | disqualified | disqualified | 2 |  |  |  |  |  |
| d06-deck | disqualified | disqualified | 2 |  |  |  |  |  |
| d07-cheap-bathroom | disqualified | disqualified | 4 |  |  |  |  |  |
| d08-injection | disqualified | disqualified | 4 |  |  |  |  |  |
| d09-parents-house | disqualified | disqualified | 2 |  |  |  |  |  |
| d10-san-antonio | disqualified | disqualified | 3 |  |  |  |  |  |
| h01-complaint | handoff | handoff | 1 |  |  |  | complaint |  |
| h02-angry | handoff | handoff | 1 |  |  |  | complaint |  |
| h03-wants-person | handoff | handoff | 1 |  |  |  | customer asked for a person |  |
| h04-discount | handoff | handoff | 1 |  |  |  | request needs staff (quote, discount, contract or warranty) |  |
| h05-warranty-claim | handoff | handoff | 1 |  |  |  | request needs staff (quote, discount, contract or warranty) |  |
| h06-contract | handoff | handoff | 1 |  |  |  | request needs staff (quote, discount, contract or warranty) |  |
| h07-no-budget | handoff | handoff | 3 |  |  |  | customer asked for a person |  |
| h08-reschedule | handoff | handoff | 1 |  |  |  | wants to reschedule or cancel |  |
