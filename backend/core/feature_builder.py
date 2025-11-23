# backend/core/feature_builder.py

import numpy as np


class FeatureBuilder:
    """
    FEATURE BUILDER – PRO VERSION
    -----------------------------
    Feladata:
        • Minden engine output egységes feature formátumba alakítása
        • Garantált sorrend (fix engine layout)
        • Normalizálás 0–1 skálára
        • Hiányzó engine → 0-vector
        • Dinamikus bővíthetőség
        • DeepValue modellhez 128 dimenziós input
    """

    # Fix engine sorrend (ha bővül, csak hozzá kell adni)
    ENGINE_LAYOUT = [
        "poisson",
        "score_pred",
        "trend",
        "public_money",
        "weather",
        "sharp_money",
        "fusion",
        "bayesian",
        "bias",
        "live_engine",
        "meta_optimizer",
        "deep_value",
    ]

    # Minden engine-ből 6 feature-t veszünk:
    # p, c, r, volatility, reliability, bias_strength
    FEATURES_PER_ENGINE = 6

    def __init__(self, config):
        self.config = config
        self.feature_size = config.get("deep_value", {}).get("input_dim", 128)

    # ======================================================================
    # NORMALIZÁLÁS HELPEREK
    # ======================================================================
    def _norm(self, x, lo=-1, hi=1):
        """Normalizál bármilyen tartományból 0–1 közé"""
        try:
            return float((x - lo) / (hi - lo))
        except:
            return 0.5

    # ======================================================================
    # FŐ FUNKCIÓ
    # ======================================================================
    def build_feature_vector(self, engine_outputs):
        """
        engine_outputs:
            {
                "poisson": {
                    "probability": 0.61,
                    "confidence": 0.72,
                    "risk": 0.31,
                    "volatility": 0.12,
                    "reliability": 0.88,
                    "bias_strength": -0.08
                },
                ...
            }
        """

        vec = []

        for eng in self.ENGINE_LAYOUT:

            data = engine_outputs.get(eng, {})

            # Raw értékek defaulttal
            p = float(data.get("probability", 0.5))
            c = float(data.get("confidence", 0.5))
            r = float(data.get("risk", 0.5))
            vol = float(data.get("volatility", 0.0))
            rel =
