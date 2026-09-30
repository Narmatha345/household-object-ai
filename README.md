# Household Object AI

**Local-first household object detection with an OpenAI fallback.**

You give it a photo (from the camera or an upload), and it identifies common
household objects. A local YOLO model always runs first. OpenAI is called
**only** when the local model can't give a confident answer. The goal is to
use the local model as much as possible and keep OpenAI API calls, and cost, low.

```
IMAGE → LOCAL YOLO → RELIABLE? ── YES → LOCAL RESULT   (no API call)
                              └── NO  → OPENAI FALLBACK
```

---

## 1. Project overview

- **Inputs:** browser camera capture, or photo upload (JPG, JPEG, PNG, WEBP).
- **Local model:** pretrained Ultralytics **YOLO26n**, loaded once at startup.
- **Confidence gate:** a local result counts only if a household object is
  detected with confidence ≥ `LOCAL_CONFIDENCE_THRESHOLD` (default 0.70).
- **Fallback:** OpenAI Responses API with image input, returning structured JSON.
- **Statistics:** total requests, local detections, OpenAI fallbacks and fallback rate.
- **Security:** the OpenAI key exists only on the backend.

## 2. Architecture

```
Camera / Upload → React frontend → FastAPI → Local YOLO → Confidence gate
                                                            ├─ reliable   → local result
                                                            └─ unreliable → OpenAI fallback
                                                                             → final result
```

Full details, including failure handling and the swappable interfaces, are in
[docs/architecture.md](docs/architecture.md).

## 3. Technology stack

| Area | Tech |
|---|---|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS 4, Axios, MediaDevices API |
| Backend | Python 3.10+, FastAPI, Uvicorn, python-multipart, Pillow, python-dotenv |
| Local ML | Ultralytics YOLO (YOLO26n pretrained on COCO) |
| Fallback | OpenAI Python SDK (Responses API, structured output) |
| Tests | pytest, FastAPI TestClient (OpenAI mocked) |

## 4. Folder structure

```
household-object-ai/
├── frontend/                 React + Vite app
│   └── src/
│       ├── components/       CameraCapture, ImageUpload, DetectionResult, LoadingState, StatsCard
│       ├── services/api.ts   Axios client + friendly error mapping
│       ├── types/            Shared TypeScript types
│       └── App.tsx
├── backend/
│   ├── app/
│   │   ├── api/routes.py     /api/health, /api/detect, /api/stats
│   │   ├── ml/               model_loader.py, detector.py (LocalObjectDetector)
│   │   ├── services/         local_detection, openai_fallback, detection_router, stats, image_input
│   │   ├── config.py         Environment-based settings
│   │   ├── errors.py         User-safe application errors
│   │   ├── schemas.py        Pydantic models
│   │   └── main.py           App factory + wiring
│   ├── models/               YOLO weights (git-ignored, auto-downloaded)
│   └── tests/
├── data/                     Dataset layout + dataset.yaml for future fine-tuning
└── docs/                     architecture.md, model-training.md
```

## 5. Local setup

Prerequisites: **Python 3.10+** and **Node.js 20+**.

```bash
git clone https://github.com/Narmatha345/household-object-ai.git
cd household-object-ai
```

## 6. Backend setup

```bash
cd backend
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements-dev.txt     # or requirements.txt without test tools
cp .env.example .env                    # Windows: copy .env.example .env
# then edit .env and set OPENAI_API_KEY
```

The first start downloads `yolo26n.pt` (~5 MB) into `backend/models/`.

## 7. Frontend setup

```bash
cd frontend
npm install
```

No frontend `.env` is needed for local development. Vite proxies `/api` to
`http://127.0.0.1:8000`.

## 8. Environment variables

**`backend/.env`** (secrets live here only, never commit it)

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | *(empty)* | Needed for the fallback. Without it, low-confidence images return a 503 |
| `LOCAL_CONFIDENCE_THRESHOLD` | `0.70` | Minimum local confidence to skip OpenAI |
| `MODEL_PATH` | `models/yolo26n.pt` | Local weights, relative to `backend/` |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Vision-capable model for the fallback |
| `OPENAI_TIMEOUT_SECONDS` | `30` | Timeout for the OpenAI request |
| `OPENAI_IMAGE_MAX_SIDE` | `1024` | Images are downscaled before upload to reduce cost |
| `HOUSEHOLD_LABELS` | household COCO classes | Comma-separated allowlist; `*` accepts every model class |
| `MAX_UPLOAD_MB` | `10` | Upload size limit |
| `CORS_ORIGINS` | `http://localhost:5173,…` | Allowed browser origins |

**`frontend/.env`** (optional, and never put secrets here)

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | *(empty)* | Backend URL when it isn't served through the Vite proxy |

## 9. How local-first detection works

1. The upload is checked for type, size and decodability, and EXIF rotation is applied.
2. `LocalObjectDetector` runs YOLO26n on the image. The model is loaded once
   at startup and reused for every request.
3. `LocalDetectionService` drops labels outside the household allowlist and
   keeps detections with confidence ≥ the threshold. For each label it keeps
   the highest-scoring box.
4. If anything remains, the result is **reliable**. It is returned with
   `source: "local"` and `fallback_used: false`, and **OpenAI is not called**.

```json
{ "source": "local", "objects": [{ "label": "chair", "confidence": 0.91 }], "fallback_used": false }
```

## 10. How the OpenAI fallback works

If no household object reaches the threshold (or the local model is unavailable):

1. The image is downscaled to at most 1024 px and JPEG-encoded.
2. It is sent to the OpenAI **Responses API** with a prompt that asks for the
   household objects, a short description and a reliability explanation. The
   answer comes back as schema-validated JSON.
3. Low-confidence local guesses are passed along as hints for the model to verify.
4. The response has `source: "openai"`, `fallback_used: true` and a
   `fallback_reason`.
5. If OpenAI fails, the API returns a clean `502` error. Details go only to the
   server logs.

## 11. How to run the application

Terminal 1, backend:

```bash
cd backend
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Terminal 2, frontend:

```bash
cd frontend
npm run dev
```

Open http://localhost:5173. The camera works on `localhost` or HTTPS.

Run the tests and the production build:

```bash
cd backend && python -m pytest
cd frontend && npm run build
```

## 12. API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | `{ "status": "ok", "local_model_loaded": true, "openai_fallback_configured": true }` |
| `POST` | `/api/detect` | multipart field `file`. Returns a `DetectionResponse` |
| `GET` | `/api/stats` | `{ total_requests, local_detections, openai_fallbacks, failed_requests, fallback_rate }` |

Interactive docs: http://localhost:8000/docs

Errors always look like `{ "detail": "<friendly message>", "code": "<error_code>" }`:
`unsupported_file_type` (415), `empty_upload` / `invalid_image` (400),
`file_too_large` (413), `fallback_failed` (502), `fallback_not_configured` (503).

Example:

```bash
curl -F "file=@living-room.jpg" http://localhost:8000/api/detect
```

## 13. Replacing the pretrained model with a custom household model

1. Label a household dataset in YOLO format under `data/processed/` and update
   `data/dataset.yaml`.
2. Fine-tune: `yolo detect train model=models/yolo26n.pt data=../data/dataset.yaml epochs=100`
3. Copy `runs/.../weights/best.pt` to `backend/models/household-yolo26n.pt`.
4. In `backend/.env`, set `MODEL_PATH=models/household-yolo26n.pt` and
   `HOUSEHOLD_LABELS=*`, then restart.

No frontend or routing changes are needed. To use something other than YOLO,
implement the `ObjectDetector` protocol in `app/ml/detector.py` and pass it to
`create_app(detector=...)`. The full guide is in
[docs/model-training.md](docs/model-training.md).

## 14. Future improvements

- Save fallback images and OpenAI labels as a review queue, and use them as
  training data (active learning).
- Cache fallback answers by image hash to avoid repeat API calls.
- Store stats persistently (SQLite) and show a history chart.
- Draw bounding boxes on the preview.
- Per-class thresholds.
- Export to ONNX/OpenVINO for faster CPU inference.
- Docker Compose for one-command startup, and CI running the tests and build.
- Authentication and rate limiting before any public deployment.

## License

[MIT](LICENSE)
