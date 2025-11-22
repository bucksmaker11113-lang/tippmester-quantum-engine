# backend/engine/poisson_engine.py

import numpy as np
from math import exp, factorial
from backend.utils.logger import get_logger

class PoissonEngine:
    """
    Poisson Engine – Gólmodell
    Kimenet:
        - home goal expectancy (lambda_home)
        - away goal expectancy (lambda_away)
        - 1X2 valószínűségek Poisson alapján
        - over/under predikció
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

    # --------------------------------------------------------------
    # FŐ FÜGGVÉNY – Poisson predikció
    # --------------------------------------------------------------
    def predict(self, preprocessed):
        self.logger.info("PoissonEngine: Poisson-alapú gólmodell fut...")

        results = {}

        for event in preprocessed.get("events", []):
            match_id = event.get("id", "unknown")
            odds = event.get("odds", {})

            # Gólvárakozások (lambda értékek)
            lam_home, lam_away = self._estimate_lambdas(odds)

            # 1X2 Poisson
            prob1, probx, prob2 = self._poisson_1x2(lam_home, lam_away)

            # Over/Under predikció (O2.5 példaként)
            over25 = self._poisson_over(lam_home + lam_away, line=2.5)

            results[match_id] = {
                "lambda_home": round(lam_home, 3),
                "lambda_away": round(lam_away, 3),
                "prob1": round(prob1, 4),
                "probx": round(probx, 4),
                "prob2": round(prob2, 4),
                "probability": round(prob1, 4),   # pipeline-kompatibilitás (1-es kimenet)
                "over25": round(over25, 4),
                "source": "PoissonEngine"
            }

        return results

    # --------------------------------------------------------------
    # LAMBDA BECSLÉS
    # --------------------------------------------------------------
    def _estimate_lambdas(self, odds):
        """
        A lambda értékeket a szorzók alapján becsüljük.
        Ez NEM statikus, hanem odds-függő adaptív becslés.
        """
        try:
            o1 = float(odds.get("1", 2.5))
            o2 = float(odds.get("2", 2.8))
        except:
            return 1.2, 1.0

        # gyengébbre: nagy odds -> kevés gól
        # erősebbre: kis odds -> több gól
        lam_home = max(0.2, min(3.0, 2 / o1))
        lam_away = max(0.2, min(3.0, 2 / o2))

        return lam_home, lam_away

    # --------------------------------------------------------------
    # POISSON 1X2 VALÓSZÍNŰSÉG
    # --------------------------------------------------------------
    def _poisson_1x2(self, lam_home, lam_away):
        max_goals = 10

        prob1 = 0
        probx = 0
        prob2 = 0

        for h in range(max_goals):
            for a in range(max_goals):

                p = self._poisson_pmf(h, lam_home) * self._poisson_pmf(a, lam_away)

                if h > a:
                    prob1 += p
                elif h == a:
                    probx += p
                else:
                    prob2 += p

        return prob1, probx, prob2

    # ---
