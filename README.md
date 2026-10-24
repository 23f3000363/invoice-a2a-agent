# Invoice Action Agent — A2A 1.0

This is a separate, self-contained FastAPI project for the invoice-agent assignment. It provides the Agent Card and the `/a2a` task endpoints, stores tasks in SQLite, reserves initial message IDs before the model call, caches decisions by canonical package content, and uses an atomic SQLite state transition for cancel-versus-result races.

## 1. Safe local setup (Windows PowerShell)

Extract this ZIP into a **new folder** such as `D:\GA_3\Q2_fixed`. Do not overwrite your existing project. Open PowerShell in that new folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

In `.env`, set `AIPIPE_TOKEN` to the AI Pipe token you already use. Keep the base URL and model as shown. Set `PUBLIC_BASE_URL=http://127.0.0.1:8000/a2a` for local testing. Never paste the token into chat or commit `.env` to GitHub. The ZIP intentionally excludes `.env`, virtual environments, and local databases.

Start the server:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Agent Card: `http://127.0.0.1:8000/.well-known/agent-card.json`
A2A base URL: `http://127.0.0.1:8000/a2a`

## 2. Check the route surface without using a grader check

In another PowerShell window in the same folder:

```powershell
python -c "from app.main import app; print('\n'.join(sorted(r.path+'  '+','.join(sorted(r.methods or [])) for r in app.routes)))"
```

You should see these application routes:

- `GET /.well-known/agent-card.json`
- `POST /a2a/message:send`
- `GET /a2a/tasks`
- `GET /a2a/tasks/{task_id}`
- `POST /a2a/tasks/{task_id}:cancel`
- `GET /health`

## 3. API contract

All A2A requests except the public Agent Card and `/health` require `Authorization: Bearer <token>` and `A2A-Version: 1.0`. POST requests require `Content-Type: application/a2a+json`. Every bearer token maps to an isolated principal; raw tokens are not stored. Use the exact public base URL, including `/a2a`, in the assignment submission.

The model proposes actions only. A receipt is created only after a continuation matches the stored task/context/batch/proposal and says `ACCEPTED`. This benchmark service records accepted synthetic actions; it does not initiate real payments.

## 4. Deploy only after local testing

For Render, connect the new folder's repository, set build command `pip install -r requirements.txt` and start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. Attach a persistent disk at `/opt/render/project/src/data` and set `DATABASE_PATH=/opt/render/project/src/data/invoice_a2a.db`. Set `PUBLIC_BASE_URL` to the exact URL you will submit, e.g. `https://YOUR-SERVICE.onrender.com/a2a`, and set `AIPIPE_TOKEN` in the host's secret environment settings. Do not deploy a placeholder URL.

## Known limitation

This code has local protocol/lifecycle smoke coverage, but hidden invoice decision accuracy cannot be proven without running the assignment's actual synthetic packages. In particular, evidence extraction depends on the model correctly interpreting the decisive paragraph. Do not spend a Check/Save until local tests and the Agent Card URL are verified.


## 5. Optional live AI connectivity test

After `.env` is configured, run `python tests/test_ai_connection.py`. This makes one small AI Pipe request and is not a grader Check/Save. The script initializes SQLite first, so the decision-cache table exists before any model call.
