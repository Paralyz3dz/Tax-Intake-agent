# Deploying the web service

How to run Tax Agent as a public website on one Linux server. The local setup in the README
(`uvicorn` on 127.0.0.1, `TAX_AGENT_MODE=local`) stays the same for development.

## What runs where

| Part | Needed for | Where it runs |
|---|---|---|
| `app` container | Pages, interview engine, preparer review | the server |
| Prebuilt guide passages (`data/processed/guide_evidence.json`) | Official text under each interview question | baked into the image |
| LLM provider (`LLM_*`) | Free-text answers, tax questions mid-interview, document reading, chat page | OpenAI, or any OpenAI-compatible server |
| Knowledge base (PostgreSQL + pgvector, `DATABASE_URL`) | Live retrieval: chat page, tax questions mid-interview | the existing RDS instance |
| Ollama (`--profile retrieval`) | Query embeddings for live retrieval | the server (about 1 GB of RAM) |
| Caddy (`--profile https`) | HTTPS certificate, security headers | the server |

Without the `retrieval` profile the interview still works completely, with passages, because
they are prebuilt. Only the chat page and tax questions asked mid-interview need live
retrieval; without it they answer "currently unavailable".

## Before the first deployment (on your own machine)

1. Build the guide passages. This needs the knowledge base, Ollama and the LLM key, which the
   development setup already has:

   ```powershell
   python -m src.workflow.guide_evidence build    # about 15 minutes, a few cents of model calls
   python -m src.workflow.guide_evidence check    # should report every question, e.g. 154/154
   ```

   Commit the file. Rebuild it whenever playbook questions or the knowledge base change; `check`
   reports how many questions are missing (those fall back to live retrieval).

2. Make the secrets for the server's `.env`:

   ```powershell
   python -m src.app.auth secret     # -> TAX_AGENT_SESSION_SECRET
   python -m src.app.auth hash       # -> the hash in TAX_AGENT_PREPARERS=alice:<hash>
   ```

## Server

Any VM with Docker works, for example AWS Lightsail or EC2 with Ubuntu 24.04:
2 GB of RAM without Ollama, 4 GB with it. Open ports 22, 80 and 443.
If the knowledge base is on RDS, allow the server's IP address in the RDS security group.

```bash
curl -fsSL https://get.docker.com | sh
git clone https://github.com/Paralyz3dz/EE542-Finalproject.git tax-agent && cd tax-agent
cp .env.example .env && nano .env
```

Settings in `.env` for a public site:

```dotenv
TAX_AGENT_MODE=public
TAX_AGENT_SESSION_SECRET=<from auth secret>
TAX_AGENT_PREPARERS=alice:<hash>
TAX_AGENT_INVITE_CODES=demo-2026-12
LLM_API_KEY=<key>
TAX_AGENT_LLM_CALLS_PER_VISITOR=200
TAX_AGENT_LLM_CALLS_PER_DAY=2000
TAX_AGENT_RETENTION_DAYS=30
DOMAIN=203-0-113-7.sslip.io        # the server's IP with dashes, or your own domain
DATABASE_URL=<RDS URL>             # only with --profile retrieval
```

No domain? `sslip.io` turns an IP address into a host name a certificate can be issued for:
`203-0-113-7.sslip.io` points to `203.0.113.7`.

Start it:

```bash
docker compose --profile https up -d --build                        # interview only
docker compose --profile https --profile retrieval up -d --build    # + chat page and tax questions
curl https://$DOMAIN/healthz
```

The app refuses to start in public mode without a session secret of at least 32 characters
and at least one preparer account; `docker compose logs app` shows why.

## Who can use what (public mode)

| Visitor | Gets in with | Can use |
|---|---|---|
| Client / assistant | an invite code | `/interview`, `/` (chat) |
| Preparer | a `TAX_AGENT_PREPARERS` account | everything, including `/preparer` |

- Remove a code from `TAX_AGENT_INVITE_CODES` and restart to sign out everyone who used it.
- Sessions last 12 hours. Ten failed sign-ins from one address block it for 15 minutes.
- Preparer decisions record who made them (`"by"` in the interview history).

## Spending caps

Every model call made with the site's key counts against two daily limits (UTC days):
`TAX_AGENT_LLM_CALLS_PER_VISITOR` per browser and `TAX_AGENT_LLM_CALLS_PER_DAY` for the whole
site. When a limit is reached, interviews keep working: free-text answers are refused with
"enter the value directly", as when the model is switched off. The chat page answers 429.

The caps count model calls, not messages. Every call is counted: the interview's note reader,
tax questions, document reading, the reranking step of live retrieval, and each call the chat
agent makes. Exact answers (yes/no, amounts, options) and prebuilt guide passages use none.

| What the visitor does | Model calls |
|---|---|
| Exact answer, or a question whose passages are prebuilt | 0 |
| Free-text answer on the interview page | up to 2 (tell answer from question, read the answer) |
| Tax question mid-interview | about 3 (classify, rerank, answer) |
| One chat message | usually 3 (decide to search, rerank, answer) |

So 200 calls per visitor is about 100 free-text answers or 60 chat messages. When the budget
runs out during live retrieval, passages come back unranked instead of failing.
The MCP server's process counts its reranking calls against the same caps.
Also set a monthly budget in the provider's dashboard as a last line of defence.

To avoid the site's key entirely, point `LLM_BASE_URL` at a self-hosted model, or set
`TAX_AGENT_LLM=off` (exact answers only).

## Data

- Interviews are stored in SQLite on the `interviews` volume. For a managed database, set
  `TAX_AGENT_INTERVIEW_DATABASE_URL`.
- `TAX_AGENT_RETENTION_DAYS` deletes interviews not updated for that many days, at startup
  and every 6 hours.
- The sign-in page says fictional data only. Uploaded images are sent to the LLM provider.

## Operating

```bash
docker compose logs -f app                       # logs
git pull && docker compose up -d --build         # update
docker compose run --rm -v "$PWD:/backup" --entrypoint cp app \
  /data/interviews.sqlite3 /backup/              # back up interviews
```

## Limits of this setup

- **One worker process.** Chat sessions, staged uploads and the model-call counters are kept in
  memory. A restart clears chat sessions and resets the counters. Running several workers
  needs these moved to the database or Redis first (`src/app/limits.py`).
- **Accounts are environment settings.** Adding a preparer means editing `.env` and restarting.
  `src/app/auth.py` is the only place to change when moving to a users table or single sign-on.
