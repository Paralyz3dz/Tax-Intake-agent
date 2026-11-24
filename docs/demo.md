# Final demo script (11/30 – 12/1, about 10 minutes)

Fictional clients only. The web part runs on the public server (`docs/deploy.md`); a local run is the backup.

## Before the demo

```powershell
python -m pytest tests -q                       # everything green
```

- Server: open `https://<DOMAIN>/healthz` (expects `"status":"ok"` and `"guide_passages":154`). Have the invite
  code and a preparer account ready.
- Open tabs: the site's `/interview` (signed in with the invite code), `/preparer` in a private window (signed
  in as the preparer), and Claude Desktop with the `tax-intake` server (see `docs/mcp.md`).
- MCP interviews are saved where the MCP server runs. To show one in `/preparer`, run the web server locally
  next to Claude Desktop (`python -m uvicorn src.app.api:app --reload`), or point both at the same
  `TAX_AGENT_INTERVIEW_DATABASE_URL`.

## 1. The problem (1 min)

A junior assistant interviews clients for a preparer. They miss follow-ups, and every gap costs the preparer
review time. The tool is a digital clone of the preparer's interview: it does not file and gives no final advice.

## 2. Web interview (3 min) — `/interview`

1. Interview type **Full tax intake**; topic "Self-employment and gig income"; background:
   `W-2 job, drives Uber on weekends, two kids, travels to craft fairs`.
2. Point at "Asked now because the client mentioned 'drives'": trigger words reorder follow-ups.
3. Open a question's "Why ask" and the official passage with its citation.
4. Ask a tax question in the reply box (e.g. "can I deduct my phone?"): the answer cites retrieved passages
   and the interview repeats its question.
5. Answer "mixed" for travel: an escalation appears in the sidebar.
6. Type an unclear answer ("maybe?"): the engine refuses it and asks again.
7. Complete the interview and select **Submit to preparer** on the summary.

## 3. Student: Form 8843 (2 min) — `/interview`

1. **Home**, then Interview type **International student: Form 8843**: only six questions.
2. Answer no, no, F visa, yes, no, no. Select **Draft Form 8843** in the summary.
3. Enter the entry date August 15, 2024 and one trip (left December 20, 2024, back January 10, 2025):
   line 4a shows 356 days for 2025, to confirm.
4. Download the review copy (marked, with a checklist). Point out that names and dates typed here are not
   saved and never reach a model.

## 4. The user's own AI runs the interview (2 min) — Claude Desktop

1. Ask: *"Start a tax intake interview. The client is an F-1 graduate student who works as a TA."*
2. Show that Claude asks one question at a time and declines to give tax advice.
3. Point out the residency questions in the Pub 4012 / Form 8843 order.
4. Explain: no API key of ours; the engine checks every answer the model sends.

## 5. Preparer review (1 min) — `/preparer`

1. Interviews wait under "Waiting for review"; MCP interviews carry the badge "AI assistant".
2. Open one: what needs the preparer, questions and answers, documents, reminders, the full record.
3. Download the review PDF, then approve with a note (the decision records who made it).

## 6. Why trust it (1 min)

- Playbook checked against Form 13614-C (51/51), Direct File knockouts (102/102), the Pub 4012 scope chart,
  and 9 replayed IRS-based and fictional cases (`docs/validation.md`).
- Every question cites the IRS / FTB publication or instructions it follows.
- When unsure, the tool escalates; the preparer decides.

## Backup plan

- Server unreachable: run locally (`python -m uvicorn src.app.api:app --reload`, `TAX_AGENT_MODE=local`).
- No knowledge base: `TAX_AGENT_RETRIEVAL=off`; the interview, Form 8843 and review still work, and interview
  questions keep their prebuilt passages.
- Claude Desktop not available: use the ChatGPT desktop app, or show a finished MCP interview already in
  the local `/preparer`.
