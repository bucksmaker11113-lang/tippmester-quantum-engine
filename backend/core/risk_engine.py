# backend/core/risk_engine.py

import numpy as np
from backend.utils.logger import get_logger

class RiskEngine:
    """
    RISK ENGINE
    -----------
    Kiszámolja:
        - variancia alapú kockázat
        - public money risk
        - volatility risk
        - injury/weather/form impact
        - market movement risk
        - final risk index (0–1)
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.logger = get_logger()

    # ------------------------------------------------------
    # FŐ KOCKÁZATI INDEX
    # ------------------------------------------------------
    def compute_risk(self, meta):
        """
        meta:
            {
                "prob": ...,
                "value_score": ...,
                "public_money": ...,
                "volatility": ...,
                "injury_risk": ...,
                "weather_risk": ...,
                "market_shift": ...,
                ...
            }
        """

        prob = meta.get("prob", 0.5)
        value_score = meta.get("value_score", 0.0)

        public_risk = meta.get("public_money", 0.0) * 0.15
        vol_risk = meta.get("volatility", 0.0) * 0.25
        injury_risk = meta.get("injury_risk", 0.0) * 0.2
        weather_risk = meta.get("weather_risk", 0.0) * 0.1
        market_risk = meta.get("market_shift", 0.0) * 0.2

        # alap: alacsony prob = nagy risk
        base_risk = (1 - prob) * 0.3

        total = base_risk + public_risk + vol_risk + injury_risk + weather_risk + market_risk

        # value csökkenti a kockázatot
        total -= (value_score * 0.2)

        return float(np.clip(total, 0, 1))
