# backend/core/bias_engine.py

import numpy as np
from backend.utils.logger import get_logger

class BiasEngine:
    """
    Bias Engine – Ensemble Layer 4
    Kiszűri az odds-, model- és market-alapú torzításokat.
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # Bias súlyozások a configból
        self.bias_weights = config.get("bias_weights", {
            "odds_bias": 0.4,
            "market_bias": 0.3,
            "model_bias": 0.2,
            "team_bias": 0.1
        })

    # ----------------------------------------------------------
    # FŐ FÜGGVÉNY – BIAS KORREKCIÓ
    # ----------------------------------------------------------
    def apply(self, posterior_data):
        self.logger.info("BiasEngine: torzítások korrigálása...")

        corrected = {}

        for match_id, pred in posterior_data.items():

            prob = pred.get("probability", 0.33)

            # 4 féle bias számítása
            odds_b = self._odds_bias(match_id)
            market_b = self._market_bias(match_id)
            model_b = self._model_bias(match_id)
            team_b = self._team_bias(match_id)

            # Súlyozott bias-kompenzáció
            bias_total = (
                odds_b * self.bias_weights["odds_bias"] +
                market_b * self.bias_weights["market_bias"] +
                model_b * self.bias_weights["model_bias"] +
                team_b * self.bias_weights["team_bias"]
            )

            # Prob korrigálása
            final_prob = prob + bias_total
            final_prob = float(max(0.01, min(0.99, final_prob)))  # clamp

            corrected[match_id] = {
                "probability": round(final_prob, 4),
                "bias_components": {
                    "odds": odds_b,
                    "market": market_b,
                    "model": model_b,
                    "team": team_b
                },
                "source": "BiasEngine"
            }

        return corrected

    # ----------------------------------------------------------
    # 1) ODDS BIAS
    # ----------------------------------------------------------
    def _odds_bias(self, match_id):
        """
        Ha az odds túl gyorsan vagy túl lassan reagált,
        akkor enyhe korrigálást végzünk.
        """
        # Stub: később odds driftből jön
        drift = np.random.uniform(-0.03, 0.03)
        return float(drift)

    # ----------------------------------------------------------
    # 2) MARKET BIAS
    # ----------------------------------------------------------
    def _market_bias(self, match_id):
        """
        A tömeges fogadási torzítás (public money) szimulálása.
        Később a Tippmix API-ból lehet mérni.
        """
        return float(np.random.uniform(-0.02, 0.02))

    # ----------------------------------------------------------
    # 3) MODEL BIAS
    # ----------------------------------------------------------
    def _model_bias(self, match_id):
        """
        Ha egy modell túlságosan dominálja a többieket,
        akkor csökkentjük a hatását.
        """
        return float(np.random.uniform(-0.015, 0.015))

    # ----------------------------------------------------------
    # 4) TEAM BIAS
    # ----------------------------------------------------------
    def _team_bias(self, match_id):
        """
        Csapatforma okozta torzítás.
        Később SofaScore-ból hozhatjuk a valós formát.
        """
        return float(np.random.uniform(-0.01, 0.01))
