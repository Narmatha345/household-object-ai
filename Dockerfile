# Single image: builds the React UI, then serves it and the FastAPI API together.

# ---- Frontend build ----
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Empty base URL: the UI calls /api on the same origin that serves it.
ENV VITE_API_BASE_URL=""
RUN npm run build

# ---- Backend runtime ----
FROM python:3.12-slim

# Runtime libs needed by OpenCV, which ultralytics imports.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app/backend

# CPU-only PyTorch first, so ultralytics doesn't pull multi-GB CUDA wheels.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./
RUN pip install -r requirements.txt

COPY backend/app ./app

# Trained household model (15 classes), a copy of runs/household-yolo26n/weights/best.pt.
COPY backend/models/household-yolo26n.pt ./models/household-yolo26n.pt

COPY --from=frontend /frontend/dist /app/frontend/dist
ENV FRONTEND_DIST=/app/frontend/dist \
    MODEL_PATH=models/household-yolo26n.pt

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
