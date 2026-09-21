# CMU Canvas Study Assistant

A local chat assistant that answers questions about your Canvas courses, assignments and
deadlines — "what's due this week?", "when is Homework 4 due?", "what am I enrolled in?" — and
that can pull up course files, showing PDFs, images and handouts right in the chat.

It runs entirely on your own machine: a Streamlit web UI in your browser, conversation history in a
local SQLite file, and **read-only** access to Canvas. The only data that leaves your machine goes
to the Canvas instance you configure and to the DeepSeek API that powers the chat.

- UI: [Streamlit](https://streamlit.io/)
- LLM: DeepSeek (`deepseek-flash`), with function/tool calling
- Data: Canvas LMS REST API at `https://canvas.cmu.edu` (configurable)
- History: SQLite (`canvas_assistant.db`)
- File text extraction: [pypdf](https://pypi.org/project/pypdf/)

## Read-only guarantee

This app can only *read* from Canvas. It can never submit an assignment, send a message, post to a
discussion, or change anything else in your account.

That is enforced in code, not just in the prompt. Every Canvas call goes through `ReadOnlySession`
in `app.py`, a `requests.Session` subclass that raises `ReadOnlyViolation` unless:

1. the HTTP method is `GET` — `POST`, `PUT`, `PATCH` and `DELETE` all raise, including through
   helpers like `session.post(...)`; and
2. the target is the Canvas origin you configured — requests to any other host raise, so your token
   cannot be sent somewhere else, and redirects are re-checked one hop at a time.

Both the high-level `request()` path and the low-level `send()` path are checked, so there is no
way to reach the network without passing the guard. The five tools exposed to the model are
read-only lookups; the model cannot invoke arbitrary endpoints, and unknown tool names are rejected.
`tests/test_app.py` covers all of this.

File downloads are the one place where traffic leaves the Canvas host, because Canvas answers a
file request with a redirect to its storage backend. Those downloads use a second session that is
still GET-only, must *start* at your Canvas origin, requires https, and strips the `Authorization`
header the moment the host changes — so your Canvas token is never sent to the storage host.
Downloaded files are held in memory for the preview and are never written to disk.

Your tokens are read from environment variables only. They are never written to the database, never
shown in the UI (the sidebar shows only "OK" or "MISSING"), and any secret that would otherwise
appear in an error message is scrubbed by `redact()`.

## Setup on Windows

You need Python 3.10 or newer ([python.org/downloads](https://www.python.org/downloads/); tick
"Add python.exe to PATH" during install).

Open PowerShell in the folder where you cloned this repo, then:

```powershell
# 1. Create a virtual environment
py -3 -m venv .venv

# 2. Activate it (PowerShell)
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your local config from the template
copy .env.example .env
notepad .env        # paste your tokens, save, close
```

If step 2 fails with "running scripts is disabled on this system", allow local scripts for your
user once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Using `cmd.exe` instead of PowerShell? Activate with `.venv\Scripts\activate.bat` and copy the
config with `copy .env.example .env`.

### Filling in `.env`

| Variable | Required | What it is |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | yes | DeepSeek API key, from [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys) |
| `CANVAS_API_TOKEN` | yes | Your Canvas personal access token (below) |
| `CANVAS_BASE_URL` | no | Canvas instance, defaults to `https://canvas.cmu.edu` |
| `DEEPSEEK_BASE_URL` | no | Defaults to `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | no | Defaults to `deepseek-flash` |
| `CANVAS_ASSISTANT_DB` | no | SQLite file path, defaults to `canvas_assistant.db` |

**Where to get a Canvas API token:** sign in at <https://canvas.cmu.edu>, then go to **Account →
Settings**, scroll to **Approved Integrations**, click **+ New Access Token**, give it a purpose
(for example "study assistant") and leave the expiry blank or set one you're comfortable with.
Click **Generate Token** and copy the value immediately — Canvas shows it only once. Paste it into
`.env` as `CANVAS_API_TOKEN`. If you lose it, delete the token in Canvas and generate a new one.

A Canvas token carries your full account access, so treat it like a password: keep it in `.env`
(which is git-ignored), don't paste it into chats or issues, and delete it in Canvas when you stop
using this app.

## Running it

With the virtual environment active:

```powershell
streamlit run app.py
```

Streamlit prints a local URL (`http://localhost:8501`) and usually opens it for you. Ask questions
in the chat box; when the assistant needs your real data it calls a Canvas tool, and you can expand
"Canvas lookup" to see exactly what it fetched. Press `Ctrl+C` in the terminal to stop the app.

Each new terminal session needs the virtual environment activated again
(`.\.venv\Scripts\Activate.ps1`) before `streamlit run app.py`.

### What it can answer

- "What courses am I in this semester?"
- "What's due in the next 10 days?"
- "When is the 15-213 midterm due?"
- "I have 6 hours tonight — what should I work on first?"
- "Show me the 21-241 syllabus."
- "Pull up the lecture 5 slides and tell me what's on them."

It cannot submit work, upload files or message anyone; ask it and it will tell you to do that in
Canvas yourself.

### Viewing course files

Ask for a file by name and the assistant searches your courses' Files, then displays it inline:
PDFs in a built-in viewer, images as images, text and code as text. Anything else — a `.pptx`, a
`.zip` — comes with a Download button instead. There is a download button on every preview.

For PDFs and text files the assistant also reads the contents (first ~50 pages, 20,000 characters),
so you can ask "what's the late policy in this syllabus?" rather than skimming it yourself. Files
larger than 25 MB are described but not fetched; open those in Canvas directly. Locked files stay
locked — the app respects whatever Canvas says you may see.

## Conversation history

Chats are stored in a local SQLite database (`canvas_assistant.db` next to `app.py` by default, or
wherever `CANVAS_ASSISTANT_DB` points). Nothing leaves your machine. Delete individual chats with
the ✕ in the sidebar, or delete the `.db` file to wipe everything. The file is git-ignored.

## Tests

No Canvas or DeepSeek credentials are needed — Canvas HTTP traffic is faked at the transport layer
and the UI runs through Streamlit's headless test harness.

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

## Troubleshooting

- **"Missing environment variables"** — `.env` is missing, empty, or in a different folder than the
  one you ran `streamlit run app.py` from. The app reads `.env` from the current directory.
- **"Canvas rejected the API token (401)"** — the token expired or was deleted. Generate a new one
  and update `.env`, then restart the app.
- **"DeepSeek rejected the API key (401)"** — check `DEEPSEEK_API_KEY`, and that the account has
  credit.
- **It can't find a file you know exists** — the assistant searches each course's Files area. If
  your instructor hid the Files tab, or the file was attached directly to a Page or assignment
  rather than uploaded to Files, it won't be listed. Naming the course ("in 15-213") narrows the
  search and usually helps.
- **Port already in use** — run `streamlit run app.py --server.port 8502`.
- **Changes to `.env` don't apply** — restart the app; environment variables are read at startup.

## macOS / Linux

Same steps, different activation:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## Layout

| File | Purpose |
| --- | --- |
| `app.py` | Everything: config, read-only Canvas client, file viewer, tool schemas, DeepSeek client, SQLite history, Streamlit UI |
| `requirements.txt` | Runtime dependencies |
| `requirements-dev.txt` | Test dependencies |
| `.env.example` | Template for your local `.env` |
| `tests/` | Unit tests and headless UI smoke tests |
