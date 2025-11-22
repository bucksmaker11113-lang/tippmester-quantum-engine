# backend/engine/montecarlo_v3_engine.py
import random
import numpy as np
from backend.utils.logger import get_logger

class MonteCarloV3:
    """
    MonteCarlo V3 – Tippmaster Quantum Engine
    Teljes MC-szimulációs motor:
        - Single
        - Live
        - Kombi eseményekre
    Rétegek:
        - alap MC futtatás
        - variance-szimuláció
        - drift-szimuláció
        - value stress-test
        - outcome-disztribúció (1X2)
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        self.single_sims = config["mc3"]["single_simulations"]
        self.var_sims = config["mc3"]["single_variance"]
        self.drift_sims = config["mc3"]["single_drift"]
        self.value_stress_sims = config["mc3"]["single_value_stress"]

        self.live_sims = config["mc3"]["live_simulations"]
        self.live_drift_sims = config["mc3"]["live_drift"]
        self.live_value_sims = config["mc3"]["live_value_stress"]

        self.kombi_event_sims = config["mc3"]["kombi_simulations_per_event"]
        self.kombi_parlay_sims = config["mc3"]["kombi_parlay_simulations"]

    # --------------------------------------------------------------
    # PUBLIC FÜGGVÉNY – QUANTUM PIPELINE IDE HÍVJA
    # --------------------------------------------------------------
    def predict(self, preprocessed):
        """
        preprocessed: dictionary
            - events
            - temporal
            - bias_corrected
            - odds_drift
        """
        results = {}

        try:
            for e in preprocessed["events"]:
                match_id = e.get("id", "unknown")

                odds = e.get("odds", {})
                probs = self._normalize_probs(e)

                # 1X2 MC-szimuláció
                mc_base = self._run_mc_simulation(probs)

                # Variancia-szimuláció
                mc_var = self._variance_test(probs)

                # Drift-szimuláció
                drift_factor = self._drift_test(probs, preprocessed.get("odds_drift", {}), match_id)

                # Value stress-test
                value_factor = self._value_stress(probs, odds)

                # Végső predikció összeállítása
                final_prob = (
                    mc_base["prob"] * 0.6 +
                    mc_var["prob"] * 0.2 +
                    drift_factor * 0.1 +
                    value_factor * 0.1
                )

                results[match_id] = {
                    "probability": round(final_prob, 4),
                    "variance": mc_var["var"],
                    "drift_factor": drift_factor,
                    "value_factor": value_factor,
                    "confidence": self._confidence(final_prob, mc_var["var"]),
                    "tip": mc_base["best_outcome"],
                    "source": "MonteCarloV3"
                }

        except Exception as e:
            self.logger.error(f"MC3 predict hiba: {e}")

        return results

    # --------------------------------------------------------------
    # 1) MONTECARLO ALAPSZIMULÁCIÓ
    # --------------------------------------------------------------
    def _run_mc_simulation(self, probs):
        p1, px, p2 = probs["1"], probs["X"], probs["2"]

        count = {"1": 0, "X": 0, "2": 0}

        for _ in range(self.single_sims):
            r = random.random()
            if r < p1:
                count["1"] += 1
            elif r < p1 + px:
                count["X"] += 1
            else:
                count["2"] += 1

        best = max(count, key=count.get)
        prob = count[best] / self.single_sims

        return {
            "best_outcome": best,
            "prob": prob,
            "distribution": count
        }

    # --------------------------------------------------------------
    # 2) VARIANCE-SZIMULÁCIÓ
    # --------------------------------------------------------------
    def _variance_test(self, probs):
        dist = []
        p1, px, p2 = probs["1"], probs["X"], probs["2"]

        for _ in range(self.var_sims):
            r = random.random()
            if r < p1:
                dist.append(1)
            elif r < p1 + px:
                dist.append(0.5)
            else:
                dist.append(0)

        variance = np.var(dist)
        average = np.mean(dist)

        return {"var": float(variance), "prob": float(average)}

    # --------------------------------------------------------------
    # 3) DRIFT-SZIMULÁCIÓ
    # --------------------------------------------------------------
    def _drift_test(self, probs, drift_data, match_id):
        drift = drift_data.get(match_id, 0)
        drift_strength = max(-0.1, min(0.1, drift))

        weighted_prob = (
            probs["1"] * (1 + drift_strength) +
            probs["X"] * (1 + drift_strength / 2) +
            probs["2"] * (1 + drift_strength * -1)
        )

        return float(weighted_prob / 3)

    # --------------------------------------------------------------
    # 4) VALUE STRESS TEST
    # --------------------------------------------------------------
    def _value_stress(self, probs, odds):
        try:
            p = probs["1"]
            o = odds.get("1", 1.0)

            expected_value = p * o

            if expected_value > 1.05:
                return 1.0
            if expected_value > 1.0:
                return 0.8
            if expected_value > 0.9:
                return 0.5

            return 0.2
        except:
            return 0.3

    # --------------------------------------------------------------
    # SEGÉD – PROBABILITÁS NORMALIZÁLÁS
    # --------------------------------------------------------------
    def _normalize_probs(self, e):
        odds = e.get("odds", {})
        try:
            o1 = float(odds.get("1", 2.5))
            ox = float(odds.get("X", 3.2))
            o2 = float(odds.get("2", 2.8))

            p1 = 1 / o1
            px = 1 / ox
            p2 = 1 / o2

            s = p1 + px + p2
            return {"1": p1 / s, "X": px / s, "2": p2 / s}
        except:
            return {"1": 0.33, "X": 0.33, "2": 0.34}

    # --------------------------------------------------------------
    # KONFIDENCIA SZÁMÍTÁS
    # --------------------------------------------------------------
    def _confidence(self, prob, var):
        stability = 1 - min(1.0, var * 2)
        return round((prob * 0.7 + stability * 0.3), 4)
