# Deploying the web service

How to run Tax Agent as a public website on one Linux server. The local setup in the README
(`uvicorn` on 127.0.0.1, `TAX_AGENT_MODE=local`) stays the same for development.

## What runs where

| Part | Needed for | Where it runs |
|---|---|---|
| `app` container | Pages, interview engine, preparer review | the server |
| Prebuilt guide passages (`data/processed/guide_evidence.json`) | Official text under each interview question | baked into the image |
| LLM provider (`LLM_*`) | Free-text answers, tax questions mid-interview, document reading, chat API | OpenAI, or any OpenAI-compatible server |
| Knowledge base (PostgreSQL + pgvector, `DATABASE_URL`) | Live retrieval: chat API, tax questions mid-interview | the existing RDS instance |
| Ollama (`--profile retrieval`) | Query embeddings for live retrieval | the server (about 1 GB of RAM) |
| Caddy (`--profile https`) | HTTPS certificate, security headers | the server |

Without the `retrieval` profile the interview still works completely, with passages, because
they are prebuilt. Only the chat API and tax questions asked mid-interview need live
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

Any VM with Docker works. These steps use AWS Lightsail, in the same region as the RDS
knowledge base (us-west-2):

1. Create an instance: platform **Linux operating system**, blueprint **Ubuntu 24.04 LTS**,
   plan **4 GB** (2 GB is enough without Ollama, i.e. without the chat API and tax questions).
2. **Networking** tab: attach a static IP, and add an IPv4 firewall rule for **HTTPS (443)**
   from any IPv4 address. SSH (22) and HTTP (80) are there by default.
3. Allow the static IP in the RDS security group: RDS > the database > Connectivity &
   security > VPC security group > Inbound rules > add **PostgreSQL (5432)** from `<static IP>/32`.
   Keep the existing rules.

Then, in the browser SSH window (**Connect using SSH**):

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu && newgrp docker      # run docker without sudo
git clone https://github.com/Paralyz3dz/Tax-Intake-agent.git tax-agent && cd tax-agent
nano .env && chmod 600 .env
```

Settings in `.env` for a public site:

```dotenv
TAX_AGENT_MODE=public
TAX_AGENT_SESSION_SECRET=<from auth secret>
TAX_AGENT_PREPARERS='alice:<hash>'
TAX_AGENT_INVITE_CODES=<a code people cannot guess>
TAX_AGENT_LLM_CALLS_PER_VISITOR=200
TAX_AGENT_LLM_CALLS_PER_DAY=3600   # about 110 per visitor; 3600 serves 30 people
TAX_AGENT_RETENTION_DAYS=30
DOMAIN=203-0-113-7.sslip.io        # the server's IP with dashes, or your own domain

LLM_API_KEY=<key>
DATABASE_URL='<RDS URL>'           # only with --profile retrieval
EMBEDDING_MODEL=<as in your local .env>
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
```

- **Keep the single quotes** around `TAX_AGENT_PREPARERS` and `DATABASE_URL`. A password hash
  contains `$`, and Docker Compose replaces `$name` in `.env` with a variable (usually empty)
  unless the value is single-quoted.
- `EMBEDDING_MODEL` and `OLLAMA_EMBEDDING_MODEL` must match what the knowledge base was built
  with, or live retrieval finds nothing.
- Interviews always go to the `interviews` volume (`/data`); docker-compose.yml sets the path.

No domain? `sslip.io` turns an IP address into a host name a certificate can be issued for:
`203-0-113-7.sslip.io` points to `203.0.113.7`.

Start it:

```bash
docker compose --profile https up -d --build                        # interview only
docker compose --profile https --profile retrieval up -d --build    # + chat API and tax questions
docker compose ps                                                   # app should become "healthy"
```

The app refuses to start in public mode without a session secret of at least 32 characters
and at least one preparer account; `docker compose logs app` shows why.

Check it:

```bash
curl https://$DOMAIN/healthz          # {"status":"ok","mode":"public",...,"guide_passages":154}
docker compose exec app python -c "from src.rag.retriever import retrieve; print(len(retrieve('standard deduction')['matches']))"
```

The second command should print `5`. A `ConnectionTimeout` means the RDS security group does
not allow the server's IP yet (step 3).

To stop paying for the server: a stopped Lightsail instance is still billed. Create a snapshot,
then delete the instance, and delete the static IP (an unattached static IP is billed too).

## Who can use what (public mode)

| Visitor | Gets in with | Can use |
|---|---|---|
| Client / assistant | an invite code | `/interview`, Form 8843 draft |
| Preparer | a `TAX_AGENT_PREPARERS` account | everything, including `/preparer` |

- Remove a code from `TAX_AGENT_INVITE_CODES` and restart to sign out everyone who used it.
- Sessions last 12 hours. Ten failed sign-ins from one address block it for 15 minutes.
- Preparer decisions record who made them (`"by"` in the interview history).

## Spending caps

Every model call made with the site's key counts against two daily limits (UTC days):
`TAX_AGENT_LLM_CALLS_PER_VISITOR` per browser and `TAX_AGENT_LLM_CALLS_PER_DAY` for the whole
site. When a limit is reached, interviews keep working: free-text answers are refused with
"enter the value directly", as when the model is switched off. The chat API answers 429.

The caps count model calls, not messages. Every call is counted: the interview's note reader,
tax questions, document reading, the reranking step of live retrieval, and each call the chat
agent makes. Exact answers (yes/no, amounts, options) and prebuilt guide passages use none.

| What the visitor does | Model calls |
|---|---|
| Exact answer, or a question whose passages are prebuilt | 0 |
| Free-text answer on the interview page | up to 2 (tell answer from question, read the answer) |
| Tax question mid-interview | 3 to 7 (classify, then up to 3 rounds of rerank + answer; a round searches again only when the passages leave a gap) |
| One chat message | usually 3 (decide to search, rerank, answer) |

So 200 calls per visitor is about 100 free-text answers, or at least 28 tax questions mid-interview. When the budget
runs out during live retrieval, passages come back unranked instead of failing.
The MCP server is a separate process with its own counters: it reads the same two limits but
does not share the web app's count. It has no per-browser visitors, so all MCP use shares one
visitor allowance; its only calls with the site's key are reranking in live retrieval, which
falls back to unranked passages when the allowance is spent.
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
