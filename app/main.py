"""Multimodal REST API; unavailable models return 503 rather than fake scores."""

import io
import logging
import os
import time
import warnings
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.schemas import ExplanationResponse, HealthResponse, ReportPredictionResponse
from pulmonary.explainability import explain
from pulmonary.inference import load_engine

LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


def create_app(engine_factory=None):
    @asynccontextmanager
    async def lifespan(application):
        application.state.engine = None
        application.state.request_count = 0
        application.state.error_count = 0
        application.state.total_latency_seconds = 0.0
        try:
            application.state.engine = (
                engine_factory()
                if engine_factory
                else load_engine(os.environ.get("MODEL_BUNDLE", "artifacts/team_bundle"))
            )
        except (OSError, ValueError, KeyError, RuntimeError):
            LOGGER.exception("Model bundle unavailable; readiness and inference will return 503")
        yield
        application.state.engine = None

    application = FastAPI(
        title="Pulmonary Review Assistant",
        version="0.1.0",
        description="Multimodal X-ray analysis with rule-based reports. Both radiograph and indication are required. Scores use weak labels and are not clinically validated. Clinician verification is required.",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def observe(request, call_next):
        started = time.perf_counter()
        length = request.headers.get("content-length")
        try:
            too_large = length is not None and int(length) > MAX_UPLOAD_BYTES + 64 * 1024
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
        if too_large:
            return JSONResponse(
                status_code=413, content={"detail": "Request exceeds the upload limit"}
            )
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        request.app.state.request_count += 1                                                                                                            
        request.app.state.error_count += int(response.status_code >= 400)
        request.app.state.total_latency_seconds += elapsed
        LOGGER.info(
            "%s %s status=%s latency_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed * 1000,
        )
        return response

    def engine_for(request):
        engine = request.app.state.engine
        if engine is None:
            raise HTTPException(
                503, "No verified model bundle is loaded. Train and export a bundle first."
            )
        return engine

    @application.get("/health", response_model=HealthResponse)                                                                                                                                                                                                                                                                                              
    def health(request: Request):
        engine = request.app.state.engine
        return {
            "status": "ready" if engine else "not_ready",
            "model_version": engine.manifest["model_version"] if engine else None,
        }

    @application.get("/ready", response_model=HealthResponse)
    def ready(request: Request):
        engine_for(request)
        return health(request)

    @application.get("/model-info")
    def model_info(request: Request):
        engine = engine_for(request)
        result = {
            key: engine.manifest[key]
            for key in [
                "model_version",
                "kind",
                "classes",
                "threshold",
                "preprocessing_version",
                "text_fields",
                "calibrated",
                "selection",
            ]
        }
        for key in ["evaluation_unit", "gradcam_target", "cohort_label_version"]:
            if key in engine.manifest:
                result[key] = engine.manifest[key]
        return result

    async def decode(image, indication, comparison, engine):
        if not indication.strip():
            raise HTTPException(422, "A nonempty clinical indication is required")
        if not engine.text(indication, comparison).strip():
            raise HTTPException(422, "The indication contains no usable text after cleaning")
        if (
            "comparison"
            in engine.manifest.get("required_text_fields", engine.manifest["text_fields"])
            and not comparison.strip()
        ):
            raise HTTPException(422, "This model also requires comparison text")
        if image.content_type not in {"image/png", "image/jpeg"}:
            raise HTTPException(415, "Upload a PNG or JPEG radiograph")
        content = await image.read(MAX_UPLOAD_BYTES + 1)
        await image.close()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image exceeds 10 MiB")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as source:
                    if source.format not in {"PNG", "JPEG"}:
                        raise ValueError("Decoded format is not PNG/JPEG")
                    if source.width * source.height > MAX_PIXELS:
                        raise ValueError("Image exceeds 20 million pixels")
                    source.load()
                    return source.copy()
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise HTTPException(422, "Unreadable or unsupported image") from exc

    @application.post("/predict", response_model=ReportPredictionResponse)
    async def predict(
        request: Request,
        image: Annotated[UploadFile, File()],
        indication: Annotated[str, Form(max_length=4000)],
        comparison: Annotated[str, Form(max_length=4000)] = "",
    ):
        engine = engine_for(request)
        decoded = await decode(image, indication, comparison, engine)
        result = await run_in_threadpool(explain, engine, decoded, indication, comparison)
        return {
            **result["prediction"],
            "predicted_class": result["predicted_class"],
            "generated_report": result["generated_report"],
            "gradcam_region": result["gradcam_region"],
        }

    @application.post("/explain", response_model=ExplanationResponse)
    async def explanation(
        request: Request,
        image: Annotated[UploadFile, File()],
        indication: Annotated[str, Form(max_length=4000)],
        comparison: Annotated[str, Form(max_length=4000)] = "",
    ):
        engine = engine_for(request)
        decoded = await decode(image, indication, comparison, engine)
        return await run_in_threadpool(explain, engine, decoded, indication, comparison)

    @application.get("/metrics")
    def request_metrics(request: Request):
        state = request.app.state
        return {
            "requests": state.request_count,
            "errors": state.error_count,
            "total_latency_seconds": state.total_latency_seconds,
            "mean_latency_seconds": state.total_latency_seconds / max(state.request_count, 1),
        }

    return application


app = create_app()
