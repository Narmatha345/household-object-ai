# A2A agent: exposing the household detector to other agents

This guide covers the A2A (Agent-to-Agent) agent in `backend/app/a2a/`: what it is, how to run it
on Windows, and how to connect it to another agent (for example, a mentor's laptop) through
Tailscale. The README has a short version.

## 1. What A2A is (and is not)

- **The model** (YOLO26n, ONNX Runtime or PyTorch) finds objects in pixels. That is unchanged.
- **A2A** is a communication protocol. It defines how one agent *discovers* another (the
  **Agent Card** at `/.well-known/agent-card.json`) and *sends it work* (JSON-RPC methods such as
  `SendMessage`, which produce a **Task** with a lifecycle and **Artifacts** as results).

The A2A agent is only an adapter. It turns an incoming A2A message into a PIL image, calls the
same `DetectionRouter` that `/api/detect` uses, and turns the `DetectionResponse` into an A2A
artifact. For the same image it returns exactly what the web API returns.

```
Remote agent ──HTTPS/HTTP over Tailnet──► A2A server :9100 (separate process)
                                           ├─ GET  /.well-known/agent-card.json   public discovery
                                           ├─ POST /a2a   JSON-RPC (A2A 1.0 + 0.3)  bearer token
                                           │     └─ ObjectRecognitionExecutor
                                           │           ├─ validate image (image_input.decode_image)
                                           │           └─ DetectionRouter.detect  ◄── same pipeline as /api/detect
                                           │                 local YOLO → 70% gate → optional OpenAI fallback
                                           └─ GET  /health   operator check (not part of A2A)

Browser ──► Vite :5173 ──► FastAPI :8000 (/api/detect, unchanged, never exposed to the Tailnet)
```

| File | Role |
|---|---|
| `app/a2a/config.py` | `A2A_*` settings; refuses a network bind without a token |
| `app/a2a/agent_card.py` | Agent Card: skill, input/output modes, endpoint URL, security scheme |
| `app/a2a/executor.py` | Parses image parts, runs detection, publishes task status and artifacts |
| `app/a2a/security.py` | Bearer-token check, request size cap, access log (no bodies or tokens) |
| `app/a2a/server.py` | Builds the app with the official `a2a-sdk` (`DefaultRequestHandler`, in-memory task store) |
| `app/bootstrap.py` | Model loading shared by the web API and the A2A agent |
| `scripts/a2a_client.py` | Standalone test client (card, send, selftest, ask) |

SDK: [`a2a-sdk`](https://pypi.org/project/a2a-sdk/) 1.2.x (official A2A Python SDK), protocol
version **1.0** over the **JSON-RPC** binding. The same endpoint also accepts **v0.3** calls
(`message/send`), so agents built on the older SDK can call it too.

## 2. Protocol behavior

**Request.** Send a user message with exactly one image part carrying inline bytes:

```json
{"jsonrpc": "2.0", "id": 1, "method": "SendMessage",
 "params": {"message": {"messageId": "<uuid>", "role": "ROLE_USER",
   "parts": [{"raw": "<base64 JPEG/PNG/WEBP>", "mediaType": "image/jpeg", "filename": "room.jpg"}]}}}
```

Headers: `Authorization: Bearer <token>`, `A2A-Version: 1.0`, `Content-Type: application/json`.
An optional text part can be added and is ignored. `data:image/...;base64,` URLs are accepted.
Remote `http(s)` image URLs are **not** fetched, which avoids server-side request forgery.

**Lifecycle.** `SUBMITTED → WORKING → COMPLETED` or `FAILED`. `SendStreamingMessage` streams each
step over SSE. `GetTask` returns the stored task later. Cancellation is not supported, because
detection is one short blocking call.

**Successful result:** artifact `detections` with two parts:
- `application/json`: the `DetectionResponse`, i.e. `objects[]` (`label`, `confidence` 0-1,
  `box` [x1,y1,x2,y2]), `source` (`local`/`openai`), `fallback_used`, `confidence_threshold`, `timing`
- `text/plain`: a summary such as `Detected 2 object(s) [source: local]: table (87%), person (78%)`

**Errors.** Nothing is ever fabricated.

| Situation | Response |
|---|---|
| Missing/wrong token | HTTP 401 with `WWW-Authenticate: Bearer` |
| Body larger than ~13.4 MB (10 MB image as base64) | HTTP 413 |
| No image, two images, corrupt bytes, empty file, image > `MAX_UPLOAD_MB`, remote URL | JSON-RPC `-32602` InvalidParams, `error.data.code` = `missing_image`, `too_many_images`, `invalid_image`, `empty_upload`, `file_too_large`, `url_not_supported` |
| GIF/BMP/other format (by declared type or by sniffing the bytes) | JSON-RPC `-32005` ContentTypeNotSupported |
| Low confidence and no OpenAI key, OpenAI error, model failure | Task `TASK_STATE_FAILED`, reason in the status message (`fallback_not_configured`, `fallback_failed`, `internal_error`), **no artifact** |

The task history stores a placeholder instead of the image bytes, so images are not echoed back
or kept in memory.

## 3. Install and run on Windows

```powershell
cd D:\household-object-ai\backend
.venv\Scripts\activate
pip install -r requirements-dev.txt        # adds a2a-sdk[http-server]
python -m app.a2a                          # reads A2A_* from backend\.env
```

Expected log line:
`A2A agent ready: card=http://127.0.0.1:9100/.well-known/agent-card.json rpc=http://127.0.0.1:9100/a2a auth=bearer token`

**VS Code:** open the folder, select the interpreter `backend\.venv\Scripts\python.exe`
(Ctrl+Shift+P → *Python: Select Interpreter*), open a terminal, then `cd backend` and
`python -m app.a2a`. To debug, add a launch configuration with `"module": "app.a2a"` and
`"cwd": "${workspaceFolder}/backend"`.

The web app (port 8000) and the A2A agent (port 9100) are independent. Run either one or both.

### Settings (`backend/.env`)

| Variable | Default | Purpose |
|---|---|---|
| `A2A_HOST` | `127.0.0.1` | Bind address. Anything other than loopback **requires** `A2A_AUTH_TOKEN` (startup fails otherwise) |
| `A2A_PORT` | `9100` | Port |
| `A2A_PUBLIC_URL` | `http://127.0.0.1:9100` | Base URL advertised in the Agent Card. **Must be reachable from the remote machine** |
| `A2A_AUTH_TOKEN` | *(empty)* | Shared bearer token (at least 16 characters) |
| `A2A_REMOTE_URL` / `A2A_REMOTE_TOKEN` | — | Defaults for `scripts/a2a_client.py` |

Generate a token with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
`MAX_UPLOAD_MB`, `LOCAL_CONFIDENCE_THRESHOLD`, `MODEL_PATH` and `OPENAI_API_KEY` apply to the
agent exactly as they do to the web API.

## 4. Inspect the Agent Card and send an image (local)

```powershell
curl.exe http://127.0.0.1:9100/.well-known/agent-card.json
curl.exe http://127.0.0.1:9100/health

python scripts\a2a_client.py card                          # fetch + validate
python scripts\a2a_client.py send path\to\room.jpg          # labels + confidence + boxes
python scripts\a2a_client.py send path\to\room.jpg --stream # shows WORKING / COMPLETED events
python scripts\a2a_client.py selftest path\to\room.jpg      # full PASS/FAIL checklist
```

The client takes `--url` and `--token`. Without them it uses `A2A_REMOTE_URL` and
`A2A_REMOTE_TOKEN`, then `A2A_AUTH_TOKEN` from `backend\.env`.

`selftest` checks: Agent Card retrieval and validation (including "does the card advertise a
URL the caller can reach?"), a valid image completes with real labels and confidence scores,
GIF/corrupt/text-only/BMP inputs are rejected, and a request without the token is refused.

## 5. Connecting over Tailscale

### Security model

- **Tailnet membership is not authorization.** Anyone on the tailnet (or any device shared into
  it) can reach an open port. Every A2A request must carry `Authorization: Bearer <token>`.
  The Agent Card is public on purpose: it tells callers *how* to authenticate (an HTTP Bearer
  `securitySchemes` entry) but never contains the token.
- Give the token to your mentor over a private channel. To rotate it, change `A2A_AUTH_TOKEN`
  and restart the agent.
- **Never use `tailscale funnel`.** Funnel publishes the service to the public internet. Don't
  add router port forwarding, and don't open port 9100 to every network in Windows Firewall.
- The A2A spec expects HTTPS in production. Tailscale already encrypts traffic between devices
  (WireGuard). Option A adds real HTTPS on top. Option B is plain HTTP inside the encrypted
  tunnel, which is acceptable for a private dev test.

### Find your Tailnet address

```powershell
tailscale ip -4                     # e.g. 100.81.162.78
tailscale status                    # machine names; your mentor's machine must be listed
tailscale status --json | Select-String DNSName   # MagicDNS name, e.g. narmatha.tail2e5e01.ts.net
```

Your mentor's laptop must be able to see yours. Either it joins the same tailnet, or you share
the machine with them (admin console → Machines → `…` → Share). For two-way tests, they share
theirs with you too. If the tailnet uses a custom access policy, it must allow their device to
reach yours on TCP 443 (Option A) or 9100 (Option B). If the MagicDNS name does not resolve on a
shared device, use the `100.x.y.z` IP.

### Option A (recommended): `tailscale serve`, HTTPS, agent stays on localhost

1. One time: in the Tailscale admin console → **DNS**, enable **MagicDNS** and **HTTPS
   Certificates**.
2. In `backend\.env`, set
   ```
   A2A_HOST=127.0.0.1
   A2A_PUBLIC_URL=https://narmatha.tail2e5e01.ts.net
   A2A_AUTH_TOKEN=<your token>
   ```
3. Start the agent (`python -m app.a2a`), then in another terminal:
   ```powershell
   tailscale serve --bg 9100       # https://<name>.ts.net -> http://127.0.0.1:9100, tailnet only
   tailscale serve status
   ```
   To stop exposing it: `tailscale serve reset`.

No firewall rule is needed, because the agent never listens on a network interface.

### Option B: bind directly to the Tailscale IP (HTTP)

1. In `backend\.env`, set
   ```
   A2A_HOST=100.81.162.78
   A2A_PUBLIC_URL=http://100.81.162.78:9100
   A2A_AUTH_TOKEN=<your token>
   ```
2. Allow port 9100 **only from the Tailscale interface and address range**. Run this in an
   Administrator PowerShell:
   ```powershell
   New-NetFirewallRule -DisplayName "A2A agent (Tailscale only)" -Direction Inbound -Protocol TCP `
     -LocalPort 9100 -InterfaceAlias "Tailscale" -RemoteAddress 100.64.0.0/10 -Action Allow
   # remove later:  Remove-NetFirewallRule -DisplayName "A2A agent (Tailscale only)"
   ```
3. Start the agent. Tailscale must already be connected, or the bind fails.

### Test the cross-laptop connection

**On your laptop** (checks the exposure path, not yet another machine):
```powershell
python scripts\a2a_client.py --url https://narmatha.tail2e5e01.ts.net card
```

**On your mentor's laptop** (the real remote test). Copy `backend/scripts/a2a_client.py` and one
test photo to that laptop, then run:
```bash
pip install "a2a-sdk[http-server]>=1.2.3,<2" httpx pillow
export A2A_REMOTE_URL=https://narmatha.tail2e5e01.ts.net     # PowerShell: $env:A2A_REMOTE_URL="..."
export A2A_REMOTE_TOKEN=<token you shared>
python a2a_client.py card
python a2a_client.py send room.jpg
python a2a_client.py selftest room.jpg
```
Or the mentor's own agent can call you with any A2A client: resolve
`<A2A_REMOTE_URL>/.well-known/agent-card.json`, send `SendMessage` (or v0.3 `message/send`) with
the bearer header.

**Your laptop calling your mentor's agent** (the other direction):
```powershell
python scripts\a2a_client.py --url <mentor agent base URL> --token <mentor token> card
python scripts\a2a_client.py --url <mentor agent base URL> --token <mentor token> ask "hello from the household agent"
```

**Evidence that it worked:** your agent's console shows lines like
`A2A HTTP POST /a2a from 127.0.0.1 (forwarded for 100.x.y.z) -> 200` and
`A2A task=… state=completed … objects=[…]`, where `100.x.y.z` is the mentor's Tailscale IP.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `A2A_HOST=… is reachable from other machines, so A2A_AUTH_TOKEN must be set` | Set a token (at least 16 characters) or bind to `127.0.0.1` |
| Remote card fetch times out | Mentor's device can't reach yours: check `tailscale status` on both, machine sharing, ACLs; Option B: firewall rule, agent bound to the 100.x IP |
| Card loads remotely but `send` fails to connect | `A2A_PUBLIC_URL` still points to `127.0.0.1`; clients call the URL *in the card*. `a2a_client.py card` flags this |
| `tailscale serve` says HTTPS is not enabled | Enable HTTPS Certificates in the admin console (DNS page) |
| HTTP 401 | Missing or wrong `Authorization: Bearer` token |
| HTTP 413 / `file_too_large` | Image above `MAX_UPLOAD_MB` (default 10) |
| `-32005` content type not supported | Send JPEG, PNG or WEBP |
| Task FAILED `fallback_not_configured` | No object reached 70% confidence and `OPENAI_API_KEY` is empty. Use a clearer photo or configure the key |
| `[WinError 10048]` address in use | Something already uses port 9100: change `A2A_PORT` (and the URL) |
| First request slow (~5 s) | Model warm-up on CPU; later requests are faster |
| Mentor's SDK is v0.3 | Supported: the card includes v0.3 fields and `/a2a` accepts `message/send` |

## 7. Limitations

- Tasks live in memory (`InMemoryTaskStore`) and are lost on restart.
- One shared token, no per-caller identity or scopes. Fine for a two-person test; use OAuth2/OIDC
  or mTLS (both declarable in the Agent Card) for anything wider.
- No push notifications and no task cancellation.
- One image per request.
