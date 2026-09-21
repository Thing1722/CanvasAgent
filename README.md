# CMU Canvas Study Assistant

A local chat assistant that answers questions about your Canvas courses, assignments and
deadlines — "what's due this week?", "when is Homework 4 due?", "what am I enrolled in?" — and
that can pull up course files and https links, showing PDFs, images, pages and math right in the chat.

It runs entirely on your own machine: a Streamlit web UI in your browser, conversation history in a
local SQLite file, and **read-only** access to Canvas. Data that leaves your machine goes to the
Canvas instance you configure, to the DeepSeek API that powers the chat, and — only when you ask
the assistant to open a link — to that public https URL. The Canvas token is never sent off the
Canvas origin (redirects included).

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
way to reach Canvas without passing the guard. The eight tools exposed to the model are
read-only lookups; the model cannot invoke arbitrary endpoints, and unknown tool names are rejected.
`tests/test_app.py` covers all of this.

Opening a non-Canvas https URL uses a **separate** GET-only session that never has the Canvas
token. Localhost, private RFC1918, link-local, and metadata addresses are blocked, as are `http`,
`file:`, and `javascript:` URLs. Redirects are followed one hop at a time and re-checked, so a
public page cannot bounce the request onto a private host. Only https is allowed: cleartext http
is an easy SSRF/mixed-content footgun, and Canvas itself already requires https.

File downloads are the one place where Canvas traffic leaves the Canvas host, because Canvas answers
a file request with a redirect to its storage backend. Those downloads use a second session that is
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
| `CANVAS_ASSISTANT_TZ` | no | IANA timezone for due dates and "today". Defaults to `America/New_York` (Pittsburgh / Eastern). Do not use a fixed UTC offset; EST/EDT follow the calendar. |

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

Due dates are shown in **Pittsburgh / Eastern time** (`America/New_York`) by default, even if the
computer itself is set to another zone (for example Beijing). Change it from the **Timezone**
select in the sidebar; the choice is remembered in the local database across reruns. You can also
set `CANVAS_ASSISTANT_TZ` in `.env`. Canvas still stores timestamps in UTC.

Each new terminal session needs the virtual environment activated again
(`.\.venv\Scripts\Activate.ps1`) before `streamlit run app.py`.

### What it can answer

- "What courses am I in this semester?"
- "What's due in the next 10 days?"
- "When is the 15-213 midterm due?"
- "I have 6 hours tonight — what should I work on first?"
- "Show me the 21-241 syllabus."
- "Pull up the lecture 5 slides and tell me what's on them."
- "What exactly does the Comparative Genre Analysis ask for, and how do I submit it?"
- "What did I turn in for Homework 4?" / "Did I upload the right PDF?"
- "Open this PDF: https://..."
- "What does this assignment formula mean?"

It cannot submit work, upload files or message anyone; ask it and it will tell you to do that in
Canvas yourself. JavaScript-heavy sites and Google Drive / login walls will show as HTML or an
error, not as a rendered app.

### Viewing course files

Ask for a file by name and the assistant searches your courses, then displays it inline: PDFs in a
built-in viewer, images as images, text and code as text. Anything else — a `.pptx`, a `.zip` —
comes with a Download button instead. There is a download button on every preview.

Paste an https link (a Canvas file URL, an assignment page, or an ordinary PDF/webpage) and the
assistant fetches it with `open_url`. Canvas file URLs reuse the same download path as `open_file`.
HTML is stripped to readable text (math included); navigation, scripts and footers are dropped
and the remaining text is grouped by heading or page so the assistant can quote original
sections. PDFs are shown with the same viewer.

Math in assignment prompts and in the assistant's replies is rendered with Streamlit's built-in
KaTeX support (`$...$` inline, `$$` on its own lines for display). Canvas MathJax (`\(...\)`,
`\[...\]`, math spans) is converted before display. Short `.tex` snippets are rendered; a full
paper is shown as source plus an excerpt.

Files are found in three places, because courses publish them differently:

1. the course **Files** area;
2. **Modules**, used automatically when a course hides its Files tab (common at CMU) or when Files
   comes back empty;
3. **attachments on an assignment**, which show up in the assignment's details.

If a course can't be searched at all — a hidden Files tab and no readable Modules — the assistant
says so instead of reporting "no files found".

For PDFs and text files the assistant also reads the contents (first ~50 pages, 20,000 characters),
so you can ask "what's the late policy in this syllabus?" rather than skimming it yourself. Files
larger than 25 MB are described but not fetched; open those in Canvas directly. Locked files stay
locked — the app respects whatever Canvas says you may see.

### Assignment details

Ask what an assignment actually requires and the assistant pulls the full record: the instructions
as written by the instructor (HTML stripped to readable text, with math kept), how it must be
submitted (file upload, text entry, a URL, on paper), which file extensions are allowed, how many
attempts you get, the rubric, any files attached to the prompt, other http(s) links in the prompt,
and whether you've submitted yet. Attached files and linked URLs can be opened from there.

Ask what you turned in and the assistant GETs only **your** submission (`/submissions/self`): text
entry, the URL you posted, uploaded files (openable in chat), grader/self comments, and your
grade/status. It cannot list classmates' work or submit/comment.

## Conversation history

Chats are stored in a local SQLite database (`canvas_assistant.db` next to `app.py` by default, or
wherever `CANVAS_ASSISTANT_DB` points). Nothing leaves your machine. Delete individual chats with
the ✕ in the sidebar, or delete the `.db` file to wipe everything. The file is git-ignored.

Opened files for the current chat appear in a **right-hand Files rail** next
to the transcript. The rail is a custom Streamlit component (`files_rail/`) that
paints one host on `document.body`, so it stays full-height while you scroll
the chat. Drag the double-line handle on the rail's left edge to resize it
(same `col-resize` cursor as the left Streamlit sidebar). Collapse or expand
it with the triangle on the middle of the right edge; a dropdown lists opened
files and selects the newest one automatically. Panel preview is HTML inside
the component (text, images, PDFs) with a Download button. The transcript
still shows previews inline. Streamlit has no native right sidebar, so this is
not a second `st.sidebar`.

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
- **It can't find a file you know exists** — the assistant searches Files, then Modules, and finds
  assignment attachments through the assignment itself. A file linked only from a Page or an
  Announcement still won't be found unless you paste the URL. Naming the course ("in 76-101")
  narrows the search, and asking about the assignment ("what's attached to the CGA?") reaches
  attachments directly.
- **A pasted link fails** — only public https URLs are fetched. Localhost, private IPs, and http
  are blocked on purpose. Sites that need a Google login or are mostly JavaScript will not look
  like they do in a browser; try a direct PDF link instead.
- **Math shows as raw TeX** — Streamlit 1.50+ renders `$...$` / `$$`. Reinstall with
  `pip install -r requirements.txt` if the UI is older.
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
