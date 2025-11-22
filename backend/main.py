# main.py – Tippmaster Quantum Engine
# FastAPI központi indító modul

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yaml
import uvicorn
import os

# API routing
from backend.api.single import router as single_router
from backend.api.kombi import router as kombi_router
from backend.api.live import router as live_router
from backend.api.update import router as update_router
from backend.api.status import router as status_router
from backend.api.health import router as health_router
from backend.api.quantum import router as quantum_router  # új endpoint

# Pipeline inicializálás
from backend.pipeline.data_loader import DataLoader
from backend.pipeline.preprocess import Preprocessor
from backend.pipeline.model_runner import ModelRunner
from backend.pipeline.fusion import FusionCore
from backend.pipeline.tip_selector import TipSelector
from backend.pipeline.output_builder import OutputBuilder

# Utils
from backend.utils.logger import init_logger


# ------------------------------------------------------------------
# CONFIG LOAD
# ------------------------------------------------------------------

def load_config():
    config_path = os.path.join("config", "quantum_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# APP INITIALIZATION
# ------------------------------------------------------------------

app = FastAPI(
    title="Tippmaster Quantum Engine",
    description="Hybrid AI + ML sportfogadási motor",
    version="4.0"
)

# CORS (ha frontend csatlakozik)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Logger inicializálása
logger = init_logger()


# ------------------------------------------------------------------
# PIPELINE INITIALIZATION
# ------------------------------------------------------------------

config = load_config()

pipeline = {
    "data": DataLoader(config),
    "pre": Preprocessor(config),
    "runner": ModelRunner(config),
    "fusion": FusionCore(config),
    "tip": TipSelector(config),
    "out": OutputBuilder(config)
}

logger.info("Quantum Pipeline inicializálva.")


# ------------------------------------------------------------------
# API ROUTES
# ------------------------------------------------------------------

app.include_router(single_router, prefix="/single")
app.include_router(kombi_router, prefix="/kombi")
app.include_router(live_router, prefix="/live")
app.include_router(quantum_router, prefix="/quantum")  # ÚJ
app.include_router(update_router, prefix="/update")
app.include_router(status_router, prefix="/status")
app.include_router(health_router, prefix="/health")


# ------------------------------------------------------------------
# ROOT
# ------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "engine": "Tippmaster Quantum Engine",
        "version": "4.0",
        "status": "running",
        "pipelines": list(pipeline.keys())
    }


# ------------------------------------------------------------------
# RUN (DEV MODE)
# ------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
