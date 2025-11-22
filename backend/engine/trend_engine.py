# backend/engine/trend_engine.py

import numpy as np
from backend.utils.logger import get_logger

class TrendEngine:
    """
    TREND ENGINE – PRO EDITION
    ---------------------------
    Feladata:
        • ligaszintű és csapatszintű trendek elemzése
        • goal trend, form trend, over/under trend
        • regressziós stabilizáció
        • streak-ek hatása
        • pace trend (liga tempó)
        • output → win probability (trend-based)
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # scaling
        self.trend_scaling = config.get("trend", {}).get("trend_scaling", 0.35)
        self.regression_factor = config.get("trend", {}).get("regression_factor", 0.25)

        self.fallback_prob = 0.54
        self.min_conf = config.get("trend", {}).get("min_confidence", 0.57)

    # ----------------------------------------------------------------------
    # PUBLIC: trend predikció
    # ----------------------------------------------------------------------
    def predict(self, match_data):
        outputs = {}

        for match_id, data in match_data.items():

            try:
                prob = self._trend_core(data)
            except Exception as e:
                self.logger.error(f"[Trend] Hiba → fallback: {e}")
                prob = self.fallback_prob

            prob = self._normalize(prob)
            conf = self._confidence(prob, data)
            risk = self._risk(prob, conf)

            outputs[match_id] = {
                "probability": round(prob, 4),
                "confidence": round(conf, 3),
                "risk": round(risk, 3),
                "meta": {
                    "trend_scaling": self.trend_scaling,
                    "regression_factor": self.regression_factor
                },
                "source": "Trend"
            }

        return outputs

    # ----------------------------------------------------------------------
    # TREND MAG
    # ----------------------------------------------------------------------
    def _trend_core(self, data):
        """
        Trend komponensek:
            • form_trend        (utolsó 10 meccs súlyozása)
            • goal_trend        (gólformák)
            • ou_trend          (over/under trend)
            • league_pace_trend (liga tempó)
        """

        form_trend = data.get("form_trend", 0.5)          # 0.0 – 1.0
        goal_trend = data.get("goal_trend", 0.5)          # 0.0 – 1.0
        ou_trend = data.get("ou_trend", 0.5)              # expected over/under
        league_pace = data.get("league_pace", 1.0)        # 0.8 – 1.2

        # alap trend erősség
        base = (
            form_trend * 0.40 +
            goal_trend * 0.30 +
            ou_trend * 0.20 +
            (league_pace - 1.0) * 0.10
        )

        # regresszió (minél extrémebb → annál jobban visszahúz)
        regression = (0.5 - base) * self.regression_factor

        prob = 0.5 + (base * self.trend_scaling) + regression

        return float(prob)

    # ----------------------------------------------------------------------
    # NORMALIZÁLÁS
    # ----------------------------------------------------------------------
    def _normalize(self, p):
        return float(max(0.01, min(0.99, p)))

    # ----------------------------------------------------------------------
    # CONFIDENCE
    # ----------------------------------------------------------------------
    def _confidence(self, prob, data):
        stability = 1 - abs(0.5 - prob)
        data_quality = data.get("data_quality", 0.80)

        conf = stability * 0.4 + data_quality * 0.6
        return float(max(self.min_conf, min(1.0, conf)))

    # ----------------------------------------------------------------------
    # RISK
    # ----------------------------------------------------------------------
    def _risk(self, prob, conf):
        return float(min(1.0, max(0.0, (1 - prob) * 0.5 + (1 - conf) * 0.5)))
