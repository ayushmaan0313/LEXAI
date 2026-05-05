"""FastAPI server for LEXAI inference."""

import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io

from api.schemas import AnalysisResult, HealthResponse
from api.inference import InferencePipeline
from lexai.config import DEFAULT_CONFIG

PROJECT_ROOT = Path(__file__).parent.parent

app = FastAPI(
    title="LEXAI API",
    description="Explainable AI for Leukemia Detection",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

checkpoint_path = os.environ.get(
    "LEXAI_CHECKPOINT",
    str(PROJECT_ROOT / DEFAULT_CONFIG.server.model_checkpoint),
)
calibrated_path = os.environ.get(
    "LEXAI_CALIBRATED_CHECKPOINT",
    str(PROJECT_ROOT / DEFAULT_CONFIG.server.calibrated_checkpoint),
)
device = os.environ.get("LEXAI_DEVICE", DEFAULT_CONFIG.server.device)

actual_checkpoint = calibrated_path if Path(calibrated_path).exists() else checkpoint_path

pipeline = InferencePipeline(
    checkpoint_path=actual_checkpoint,
    config=DEFAULT_CONFIG,
    device=device,
)

WEB_DIST = PROJECT_ROOT / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIST / "assets")), name="assets")


@app.get("/", response_class=FileResponse)
async def serve_frontend():
    index_path = WEB_DIST / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "LEXAI API is running. Frontend not built — run `npm run build` in web/."}


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline.model_loaded,
        device=str(pipeline.device),
        version="0.2.0",
        num_classes=DEFAULT_CONFIG.data.num_classes,
        class_names=DEFAULT_CONFIG.data.class_names,
    )


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. "
                   f"Allowed: {allowed_types}",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = pipeline.analyze(image)
        return AnalysisResult(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("LEXAI_PORT", DEFAULT_CONFIG.server.port))
    uvicorn.run(
        "api.server:app",
        host=DEFAULT_CONFIG.server.host,
        port=port,
        reload=False,
    )
