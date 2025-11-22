# backend/core/meta_layer_ensemble_optimizer.py

import numpy as np
from backend.utils.logger import get_logger

class MetaLayerEnsembleOptimizer:
    """
    META-LAYER ENSEMBLE OPTIMIZER – AI ENSEMBLE BRAIN
    -------------------------------------------------
    Feladata:
        • Az összes engine (20+) súlyozásának folyamatos optimalizálása
        • Adott sportág, liga, csapat, időjárás, napszak alapján új súlyok tanulása
        • Engine performance tracking
        • Dynamic retraining
        • Performance-based weighting
        • Stability scoring
        • FusionEngine-hez valós idejű súlyok kiszolgálása

    Ez lesz az egész rendszer irányító neurális hálózata.
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # alap engine nevek (dinamikusan kiegészülhet)
        self.engines = config.get("optimizer", {}).get("engine_list", [
            "montecarlo", "lstm", "gnn", "poisson", "rl",
            "gameflow", "trend", "injury", "weather", "public",
            "anomaly", "scorepred", "temporary",
            "marketmicro", "oddsmaker", "arbitrage", "closingline"
        ])

        # kezdeti súlyok
        self.weights = {engine: 1.0 / len(self.engines) for engine in self.engines}

        # learning rate for dynamic adjustment
        self.lr = config.get("optimizer", {}).get("learning_rate", 0.05)

        # stability factor
        self.stability = config.get("optimizer", {}).get("stability", 0.85)

    # ----------------------------------------------------------------------
    # PUBLIC API – fusion engine ezt fogja kérni real-time
    # ----------------------------------------------------------------------
    def get_weights(self):
        """Visszaadja az aktuális optimalizált súlyokat."""
        return self.weights

    # ----------------------------------------------------------------------
    # UPDATE FROM PERFORMANCE DATA
    # ----------------------------------------------------------------------
    def update_weights(self, performance_data):
        """
        Várt input:
            {
                "montecarlo": 0.62,
                "lstm": 0.58,
                "gnn": 0.61,
                ...
            }

        Ezek 0–1 score-ok → minél magasabb, annál jobb.
        """

        try:
            self._apply_performance_update(performance_data)
        except Exception as e:
            self.logger.error(f"[MetaOptimizer] Weight update error: {e}")

        return self.weights

    # ----------------------------------------------------------------------
    # PERFORMANCE-BASED GRADIENT UPDATE
    # ----------------------------------------------------------------------
    def _apply_performance_update(self, perf):
        total_perf = sum(max(0.001, p) for p in perf.values())

        # normalizált target súlyok performance alapján
        target_weights = {
            eng: perf.get(eng, 0.001) / total_perf
            for eng in self.engines
        }

        # GRADIENT DESCENT-HEZ HASONLÓ ADJUSTMENT
        for eng in self.engines:
            old = self.weights.get(eng, 1.0 / len(self.engines))
            target = target_weights.get(eng, old)

            new_weight = (
                old * self.stability +  # stabilitás
                target * (1 - self.stability) +  # új adat
                self.lr * (target - old)  # tanulás
            )

            self.weights[eng] = float(max(0.001, new_weight))

        # újra normalizáció
        self._normalize_weights()

    # ----------------------------------------------------------------------
    # NORMALIZÁLÁS
    # ----------------------------------------------------------------------
    def _normalize_weights(self):
        total = sum(self.weights.values())
        if total == 0:
            return
        for eng in self.weights:
            self.weights[eng] = self.weights[eng] / total
