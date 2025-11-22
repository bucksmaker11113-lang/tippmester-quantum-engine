# backend/core/value_analyzer.py

import numpy as np
from backend.utils.logger import get_logger

class ValueAnalyzer:
    """
    Value Analyzer – Ensemble Layer 5
    Kiszámolja:
        - expected value (EV)
        - value score
        - kockázati szint
        - final recommendation metaadatok
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # Minimum EV threshold a configból
        self.min_ev = config.get("value_thresholds", {}).get("min_ev", 0.05)
        self.min_value_score = config.get("value_thresholds", {}).get("min_value_score", 0.2)

    # --------------------------------------------------------------
    # FŐ FÜGGVÉNY – VALUE SZÁMÍTÁS
    # --------------------------------------------------------------
    def evaluate(self, bias_corrected):
        self.logger.info("ValueAnalyzer: value számítás...")

        final = {}

        for match_id, pred in bias_corrected.items():

            prob = pred.get("probability", 0.33)
            odds = pred.get("odds", {"1": 2.5, "X": 3.2, "2": 2.8})

            # Expected Value (EV)
            ev = self._expected_value(prob, odds)

            # Value Score (normalizált)
            value_score = self._value_score(ev)

            # Risk score (variancia + public bias alapján)
            risk_score = self._risk(prob, value_score)

            final[match_id] = {
                "probability": round(prob, 4),
                "ev": round(ev, 4),
                "value_score": round(value_score, 3),
                "risk": round(risk_score, 3),
                "confidence": self._confidence(prob, value_score, risk_score),
                "source": "ValueAnalyzer"
            }

        return final

    # --------------------------------------------------------------
    # EXPECTED VALUE
    # --------------------------------------------------------------
    def _expected_value(self, prob, odds):
        # 1-es kimenet oddsát vesszük alapértelmezésként (később bővíthető 1X2-re)
        o1 = float(odds.get("1", 2.5))

        return prob * o1 - (1 - prob)

    # --------------------------------------------------------------
    # VALUE SCORE – normalizálás
    # --------------------------------------------------------------
    def _value_score(self, ev):
        # skálázás -1 és +1 között
        return float(np.tanh(ev))

    # --------------------------------------------------------------
    # RISK SCORE
    # --------------------------------------------------------------
    def _risk(self, prob, value_score):
        """
        Magas valószínűség + magas value -> alacsony risk
        Alacsony valószínűség + magas variance -> magas risk
        """
        base_risk = 1 - prob
        v_adj = (1 - value_score) * 0.3

        return float(max(0.0, min(1.0, base_risk + v_adj)))

    # --------------------------------------------------------------
    # CONFIDENCE számítás
    # --------------------------------------------------------------
    def _confidence(self, prob, value_score, risk):
        return float(max(0.0, min(1.0, prob * 0.5 + value_score * 0.4 + (1 - risk) * 0.1)))
