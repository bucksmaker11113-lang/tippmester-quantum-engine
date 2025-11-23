# backend/core/fusion_engine.py

import os
import importlib
import numpy as np
from backend.utils.logger import get_logger


class FusionEngine:
    """
    FUSION ENGINE – PRO VERSION
    ---------------------------
    Feladata:
        • automatikus engine-felderítés (backend/engine)
        • engine-ek futtatása hibatűrő módban
        • reliability metrika számítása
        • consensus score (engine-k egyezése)
        • weighted probability fusion
        • sharp-safe fallback
    """

    def __init__(self, config=None):
        self.config = config or {}

        self.logger = get_logger()
        self.engine_dir = os.path.join("backend", "engine")

        self.max_engines = self.config.get("fusion", {}).get("max_engines", 50)

        self.engines = self._load_engines()

    # ======================================================================
    # ENGINE AUTO-DETECTION
    # ======================================================================
    def _class_guess(self, filename):
        """score_pred_engine → ScorePredEngine"""
        base = filename.replace("_engine", "")
        parts = base.split("_")
        return "".join([p.capitalize() for p in parts]) + "Engine"

    def _load_engines(self
