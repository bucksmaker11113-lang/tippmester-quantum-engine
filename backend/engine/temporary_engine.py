# backend/engine/temporary_engine.py

import numpy as np
from backend.utils.logger import get_logger

class TemporaryEngine:
    """
    TEMPORARY ENGINE – FAILSAFE EDITION
    ------------------------------------
    Feladata:
        • Ha bármelyik modell hibázik vagy üres outputot ad → stabil fallback predikció
        • Biztonsági háló a FusionEngine és Pipeline számára
        • Normalizált probability + confidence + risk
        • Minimális, de stabil baseline, hogy a rendszer soha ne álljon le
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # fallback probability (safe baseline)
        self.base_prob = config.get("temporary", {}).get("base_prob", 0.52)

        # minimum confidence
        self.min_conf = config.get("temporary", {}).get("min_confidence", 0.55)

    # ----------------------------------------------------------------------
    # PUBLIC: fallback predikció
    # ----------------------------------------------------------------------
    def predict(self, match_data):
        outputs = {}

        for match_id in match_data:

            # Nincs szükség konkrét adatra → mindig működik
            prob = self._normalize(self.base_prob)
            conf = self._confidence(prob)
            risk = self._risk(prob, conf)

            outputs[match_id] = {
                "probability": round(prob, 4),
                "confidence": round(conf, 3),
                "risk": round(risk, 3),
                "meta": {"fallback": True},
                "source": "Temporary"
            }

        return outputs

    # ----------------------------------------------------------------------
    # NORMALIZATION
    # ----------------------------------------------------------------------
    def _normalize(self, p):
        return float(max(0.01, min(0.99, p)))

    # ----------------------------------------------------------------------
    # CONFIDENCE
    # ----------------------------------------------------------------------
    def _confidence(self, prob):
        # Baseline confidence – semmi extra adat, csak safe mode
        conf = 0.5 + abs(prob - 0.5)
        return float(max(self.min_conf, min(1.0, conf)))

    # ----------------------------------------------------------------------
    # RISK
    # ----------------------------------------------------------------------
    def _risk(self, prob, conf):
        return float(min(1.0, max(0.0, (1 - prob) * 0.4 + (1 - conf) * 0.6)))
