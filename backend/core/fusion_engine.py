# backend/core/fusion_engine.py

import os
import importlib
import numpy as np
from backend.utils.logger import get_logger

class FusionEngine:
    """
    QUANTUM FUSION ENGINE – PRO EDITION
    -----------------------------------
    • Automatikus engine-felderítés
    • Dinamikus súlyozás (reliability + consensus)
    • Multi-engine probability összevonás
    • Sharp-safe fallback rendszer
    • Bayesian / Bias / Value kompatibilitás
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # alap súlyozás, ha egy engine nem ad sajátot
        self.default_weight = config.get("default_engine_weight", 1.0)

        # engine mappa
        self.engine_dir = "backend/engine"

        # engine registry
        self.engines = self._discover_engines()

    # ------------------------------------------------------------
    # ENGINE FELFEDEZÉSEK (plug-in rendszer)
    # ------------------------------------------------------------
    def _discover_engines(self):
        engines = {}

        for file in os.listdir(self.engine_dir):
            if file.endswith("_engine.py"):

                name = file.replace(".py", "")

                try:
                    module = importlib.import_module(f"backend.engine.{name}")
                    cls = getattr(module, self._class_guess(name), None)

                    if cls:
                        engines[name] = cls(self.config)
                        self.logger.info(f"[FusionEngine] Engine betöltve: {name}")
                except
