# backend/main.py

import json
import sys
from pathlib import Path
from backend.utils.logger import get_logger

# Pipeline modul (HA tényleg master_pipeline helyett ensemble_pipeline van)
try:
    from backend.pipeline.ensemble_pipeline import EnsemblePipeline as MasterPipeline
except ImportError:
    from backend.pipeline.master_pipeline import MasterPipeline

logger = get_logger()


# ----------------------------------------------------------------------
# CONFIG LOADER
# ----------------------------------------------------------------------
def load_config(path="config.json"):
    config_path = Path(path)

    if not config_path.exists():
        logger.error(f"[MAIN] config.json nem található: {config_path.resolve()}")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception(f"[MAIN] Config betöltése sikertelen: {e}")
        return {}


# ----------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------
def main():
    logger.info("[MAIN] Rendszer indul...")

    # CONFIG BETÖLTÉS
    config = load_config()
    if not config:
        logger.warning("[MAIN] Üres config -> alapértelmezett beállításokkal indul.")
        config = {}

    # PIPELINE INITIALIZATION
    try:
        pipeline = MasterPipeline(config)
    except Exception as e:
        logger.exception("[MAIN] Pipeline inicializáció sikertelen.")
        sys.exit(1)

    # DAILY RUN
    try:
        result = pipeline.run_daily()
    except Exception as e:
        logger.exception("[MAIN] Pipeline futtatási hiba.")
        sys.exit(1)

    # REPORT (frontend JSON kompatibilis)
    report = {
        "date": result.get("date"),
        "tips": result.get("tips", []),
        "kombi": result.get("kombi"),
        "engine_status": "OK"
    }

    logger.info("[MAIN] Napi riport elkészült.")
    print(json.dumps(report, indent=4, ensure_ascii=False))


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
