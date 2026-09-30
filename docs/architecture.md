# Architecture

Household Object AI is **local-first**: every image goes to a local YOLO model,
and OpenAI is called only when the local result fails a confidence gate. The goal
is to keep OpenAI API calls, and their cost, as low as possible.

## Request flow

```
 Camera / Upload
       │  image (JPEG / PNG / WEBP)
       ▼
 ┌──────────────┐   multipart/form-data   ┌──────────────────────────┐
 │  React + TS  │ ──────────────────────▶ │ FastAPI  POST /api/detect│
 │  (frontend)  │                         └────────────┬─────────────┘
 └──────────────┘                                      │ validate type, size, decode
        ▲                                              ▼
        │                                   ┌─────────────────────┐
        │                                   │  Local YOLO (26n)   │  always runs first
        │                                   └──────────┬──────────┘
        │                                              ▼
        │                                   ┌─────────────────────┐
        │                                   │  Confidence gate    │  household label AND
        │                                   │                     │  confidence ≥ threshold?
        │                                   └───┬─────────────┬───┘
        │                                  YES  │             │  NO
        │                                       ▼             ▼
        │                          ┌────────────────┐   ┌──────────────────┐
        │                          │ Local result   │   │ OpenAI fallback  │  Responses API,
        │                          │ (no API call)  │   │ (vision)         │  structured JSON
        │                          └───────┬────────┘   └────────┬─────────┘
        │                                  └─────────┬───────────┘
        │                                            ▼
        └──────────────────────────────── Final result + stats update
```

## Components

| Layer | File | Responsibility |
|---|---|---|
| API | `backend/app/api/routes.py` | HTTP endpoints, upload validation, runs routing in a worker thread |
| Input | `backend/app/services/image_input.py` | Content-type/extension check, size limit, safe Pillow decode, EXIF rotation |
| Model loading | `backend/app/ml/model_loader.py` | Loads YOLO weights once per process (cached, thread-safe) |
| Detector | `backend/app/ml/detector.py` | `ObjectDetector` protocol + `LocalObjectDetector` (YOLO) returning raw `(label, confidence)` pairs |
| Confidence gate | `backend/app/services/local_detection.py` | Applies the household-label allowlist and `LOCAL_CONFIDENCE_THRESHOLD`; decides *reliable* vs *unreliable* |
| Fallback | `backend/app/services/openai_fallback.py` | Downscales the image, calls the OpenAI Responses API, parses structured JSON |
| Router | `backend/app/services/detection_router.py` | Core logic: local → gate → (optional) fallback; records stats |
| Stats | `backend/app/services/stats.py` | Thread-safe in-memory counters |
| Wiring | `backend/app/main.py` | `create_app()` builds and injects all services, error handlers, CORS |

## The confidence gate

A local result is **reliable** when at least one detection:

1. has a label in the household allowlist (`HOUSEHOLD_LABELS`, defaults to the
   household-relevant COCO classes), and
2. has confidence ≥ `LOCAL_CONFIDENCE_THRESHOLD` (default `0.70`).

Only those detections are returned (best box per label, sorted by confidence).
Otherwise the result is unreliable. The weaker local guesses are sent to OpenAI
as *hints to verify*, and returned to the UI as `local_candidates`.

Raising the threshold increases accuracy but also fallback rate (cost). Lowering
it saves API calls but accepts more mistakes. Watch the fallback rate in the
stats card when tuning.

## Failure handling

| Situation | Behaviour |
|---|---|
| Unsupported type / empty / corrupt / too large | 415 / 400 / 400 / 413 with a friendly message; the model is never invoked |
| Local model failed to load at startup | Server still starts; `/api/health` reports `local_model_loaded: false`; requests go to the fallback |
| Local inference throws | Treated as unreliable → fallback |
| No `OPENAI_API_KEY` and local result unreliable | 503 `fallback_not_configured` |
| OpenAI error / timeout / bad response | 502 `fallback_failed`; the real error is logged server-side only |
| Any unexpected error | 500 with a generic message; no stack traces are returned |

## Swappable parts

- **Local model**: any class that implements `ObjectDetector`
  (`is_loaded` + `detect(PIL.Image) -> list[RawDetection]`) can be passed to
  `create_app(detector=...)`. For a fine-tuned YOLO model, just change `MODEL_PATH`.
- **Fallback**: any class that implements `FallbackDetector` can be passed to
  `create_app(fallback=...)`. The tests use this to mock OpenAI.

The frontend depends only on the `DetectionResponse` JSON shape, so neither swap
requires frontend changes.

## Security

- `OPENAI_API_KEY` lives only in `backend/.env` (git-ignored). The frontend has
  no secrets; in development, Vite proxies `/api` to the backend.
- Uploads are size-limited and decoded with Pillow's decompression-bomb guard.
- Error responses contain only a message and an error code.
