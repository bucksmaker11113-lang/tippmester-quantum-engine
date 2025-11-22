# backend/pipeline/output_builder.py

import json
import os
from datetime import datetime
from backend.utils.logger import get_logger

class OutputBuilder:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger()
        self.output_path = "backend/data/tippmix_cache.json"

    # ----------------------------------------------------------
    # FŐ METÓDUS: FINAL OUTPUT GENERÁLÁS
    # ----------------------------------------------------------
    def build(self, tips):
        self.logger.info("OutputBuilder: végső output építése...")

        final_output = {
            "timestamp": datetime.utcnow().isoformat(),
            "engine": "Tippmaster Quantum Engine 4.0",
            "tips_count": len(tips),
            "tips": tips
        }

        # Mentés cache-be
        self._save(final_output)

        return final_output

    # ----------------------------------------------------------
    # CACHE MENTÉS
    # ----------------------------------------------------------
    def _save(self, output):
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=4)
        except Exception as e:
            self.logger.error(f"Output cache mentési hiba: {e}")
