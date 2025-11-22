# backend/engine/gameflow_engine.py

import numpy as np
from backend.utils.logger import get_logger

class GameflowEngine:
    """
    Gameflow Engine – meccs-momentum modellezése
    Kimenet:
        - flow_score
        - goal_expectancy
        - momentum_trend
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

    # --------------------------------------------------------------
    # FŐ FÜGGVÉNY – GAMEFLOW ANALYSIS
    # --------------------------------------------------------------
    def predict(self, preprocessed):
        self.logger.info("GameflowEngine: gameflow elemzés fut...")

        results = {}

        for event in preprocessed.get("events", []):
            match_id = event.get("id", "unknown")

            # 1) Gólexpektancia (xG típusú szimuláció)
            goal_exp = self._goal_expectancy(event)

            # 2) Momentum trend – időfüggő valószínűség
            momentum = self._momentum(event, goal_exp)

            # 3) Flow Score – összesített mérőszám
            flow_score = self._flow_score(goal_exp, momentum)

            results[match_id] = {
                "goal_expectancy": round(goal_exp, 3),
                "momentum_trend": round(momentum, 3),
                "flow_score": round(flow_score, 3),
                "probability": float(flow_score),  # pipeline kompatibilitás
                "source": "GameflowEngine"
            }

        return results

    # --------------------------------------------------------------
    # 1) GÓLEXPECTANCY SZIMULÁCIÓ
    # --------------------------------------------------------------
    def _goal_expectancy(self, event):
        """
        Gyors, Poisson-szerű várható gólmodell.
        """
        odds = event.get("odds", {})
        try:
            o1 = float(odds.get("1", 2.5))
            o2 = float(odds.get("2", 2.8))

            # Alap xG: minél kisebb az odds, annál nagyobb a gólvárakozás
            xg = (1/o1 + 1/o2) / 2
            return xg
        except:
            return 0.5

    # --------------------------------------------------------------
    # 2) MOMENTUM TREND
    # --------------------------------------------------------------
    def _momentum(self, event, goal_exp):
        """
        Momentum score:
            - odds mozgás
            - előf
