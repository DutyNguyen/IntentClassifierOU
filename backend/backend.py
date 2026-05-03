import os
import sys
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODEL_PATH = ROOT / "models" / "intent_classifier.pkl"
LABELS_PATH = ROOT / "models" / "labels.json"

APP_ENV = os.getenv("APP_ENV", "development").lower()
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "1200"))
API_KEY = os.getenv("API_KEY", "")


def _split_csv_env(value: str, default: list[str]) -> list[str]:
    if not value:
        return default
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items if items else default


ALLOWED_ORIGINS = _split_csv_env(
    os.getenv("ALLOWED_ORIGINS", ""),
    ["*"] if APP_ENV != "production" else [],
)
ALLOWED_HOSTS = _split_csv_env(
    os.getenv("ALLOWED_HOSTS", ""),
    ["*"] if APP_ENV != "production" else ["localhost", "127.0.0.1"],
)

_pipeline = None
_labels = None

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _load_model() -> None:
    global _pipeline, _labels
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Copy your trained model here."
        )
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels not found at {LABELS_PATH}. Copy labels.json here."
        )

    _pipeline = joblib.load(MODEL_PATH)
    with open(LABELS_PATH, encoding="utf-8") as f:
        _labels = json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    assert _labels is not None
    print(f"[Backend] Model loaded - {len(_labels)} intents.")
    yield


app = FastAPI(
    title="OU Intent Inference Engine",
    description="Intent classification API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=ALLOWED_HOSTS,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    allow_credentials=False,
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    intent: str
    confidence: float
    probabilities: dict[str, float]
    top5: list[dict]


def _preprocess(text: str) -> str:
    try:
        from nlp.preprocessor import VietnamesePreprocessor

        return VietnamesePreprocessor().process(text)
    except Exception:
        return text.lower().strip()


def _verify_api_key(x_api_key: str | None = Security(api_key_header)) -> str | None:
    """Verify API key from header. Swagger will show Authorize button."""
    if not API_KEY:
        return None  # No key required if not configured
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _: str | None = Security(_verify_api_key)):

    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty input")

    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Input too long. Max {MAX_INPUT_CHARS} characters",
        )

    processed = _preprocess(text)
    proba = _pipeline.predict_proba([processed])[0]
    classes = _pipeline.classes_

    prob_dict = {c: round(float(p), 6) for c, p in zip(classes, proba)}
    best_idx = int(np.argmax(proba))
    intent = classes[best_idx]
    confidence = float(proba[best_idx])

    top5 = sorted(
        [{"intent": c, "prob": round(float(p), 6)} for c, p in zip(classes, proba)],
        key=lambda x: x["prob"],
        reverse=True,
    )[:5]

    return PredictResponse(
        intent=intent,
        confidence=confidence,
        probabilities=prob_dict,
        top5=top5,
    )
