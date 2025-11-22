# backend/engine/rl_engine.py

import numpy as np
from backend.utils.logger import get_logger

class ReinforcementEngine:
    """
    Policy-Based Reinforcement Learning modell
    (Offline RL előkészítés, online policy futtatás)

    Cél:
        - hosszútávú EV alapján döntést hozni
        - megtanult jutalmazási policy alapján módosítani a predikciókat
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # RL súlyok (policy paraméterei)
        self.weights = config.get("rl_weights", {
            "prob": 0.5,
            "value": 0.3,
            "risk": -0.2,     # kockázat negatív jutalom
            "momentum": 0.1
        })

        # Reward beállítások
        self.reward_scale = config.get("rl_reward_scale", 1.0)

    # --------------------------------------------------------------
    # FŐ RL-PREDIKCIÓ
    # --------------------------------------------------------------
    def predict(self, preprocessed):
        self.logger.info("ReinforcementEngine: RL policy futtatása...")

        results = {}

        # RL a preprocesselt adatokkal dolgozik
        events = preprocessed.get("events", [])
        temporal = preprocessed.get("temporal", {})
        drift = preprocessed.get("odds_drift", {})

        for event in events:
            match_id = event.get("id", "unknown")
            odds = event.get("odds", {})

            # Feature-ek a policy-hez
            prob = self._initial_prob(odds)
            value = self._value_prob(odds, prob)
            risk = self._risk(value, prob)
            mom = self._momentum(match_id, drift)

            # Policy output = súlyozott döntési értékelés
            policy_score = (
                prob * self.weights["prob"] +
                value * self.weights["value"] +
                risk * self.weights["risk"] +
                mom * self.weights["momentum"]
            )

            reward = self._reward(value, risk)

            results[match_id] = {
                "probability": max(0.01, min(0.99, policy_score)),
                "value_score": round(value, 3),
                "risk": round(risk, 3),
                "momentum": round(mom, 3),
                "reward": round(reward, 3),
                "source": "RL_Engine"
            }

        return results

    # --------------------------------------------------------------
    # ALAP VALÓSZÍNŰSÉG BECSSLÉSE
    # --------------------------------------------------------------
    def _initial_prob(self, odds):
        try:
            o1 = float(odds.get("1", 2.5))
            p = 1 / o1
            return p
        except:
            return 0.33

    # --------------------------------------------------------------
    # VALUE ALAPÚ VALÓSZÍNŰSÉG
    # --------------------------------------------------------------
    def _value_prob(self, odds, prob):
        try:
            o1 = float(odds.get("1", 2.5))
            ev = prob * o1
            return float(np.tanh(ev - 1))   # normalizált value
        except:
            return 0.0

    # --------------------------------------------------------------
    # RISK SZÁMÍTÁS
    # --------------------------------------------------------------
    def _risk(self, value, prob):
        return float(max(0.0, min(1.0, (1 - prob) * (1 - value))))

    # --------------------------------------------------------------
    # MOMENTUM / DRIFT
    # --------------------------------------------------------------
    def _momentum(self, match_id, drift_data):
        d = drift_data.get(match_id, 0)
        return float(max(-1.0, min(1.0, d * 5)))

    # --------------------------------------------------------------
    # REWARD SZÁMÍTÁS
    # --------------------------------------------------------------
    def _reward(self, value, risk):
        reward = value - risk
        return reward * self.reward_scale
