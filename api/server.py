"""FastAPI server for LEXAI inference."""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import AnalysisResult, HealthResponse
from api.inference import InferencePipeline
from lexai.config import DEFAULT_CONFIG


# Initialize FastAPI
app = FastAPI(
    title="LEXAI API",
    description="Explainable AI for Leukemia Detection",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference pipeline
checkpoint_path = os.environ.get(
    "LEXAI_CHECKPOINT",
    str(PROJECT_ROOT / "checkpoints" / "best_model.pth"),
)
device = os.environ.get("LEXAI_DEVICE", "auto")

pipeline = InferencePipeline(
    checkpoint_path=checkpoint_path,
    config=DEFAULT_CONFIG,
    device=device,
)

# Serve static frontend files
WEB_DIR = PROJECT_ROOT / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the main dashboard page."""
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "LEXAI API is running. Frontend not found."}


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline.model_loaded,
        device=str(pipeline.device),
        version="0.1.0",
    )


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze a blood smear image for leukemia detection.

    Accepts: JPG, PNG, BMP, TIFF images
    Returns: Full analysis with classification, explainability, and uncertainty
    """
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. "
                   f"Allowed: {allowed_types}",
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run analysis
        result = pipeline.analyze(image)

        return AnalysisResult(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("LEXAI_PORT", 8000))
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
