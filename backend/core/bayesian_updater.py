# backend/core/bayesian_updater.py

import numpy as np
from backend.utils.logger import get_logger

class BayesianUpdater:
    """
    Bayesian Updater – Ensemble Layer 3
    Frissíti a Fusion Engine outputját valamennyi modell likelihood értékével.
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # smoothing faktor – megakadályozza a 0-val osztást és kilengéseket
        self.smoothing = config.get("bayes_smoothing", 0.05)

        # likelihood súlyozás
        self.likelihood_weights = config.get("bayes_likelihood_weights", {
            "mc3": 0.40,
            "lstm": 0.25,
            "gnn": 0.20,
            "poisson": 0.10,
            "rl": 0.05
        })

    # ----------------------------------------------------------
    # FŐ FÜGGVÉNY
    # ----------------------------------------------------------
    def update(self, fused_predictions):
        self.logger.info("BayesianUpdater: posterior valószínűségek számítása...")

        posterior_output = {}

        for match_id, pred in fused_predictions.items():

            prior = pred.get("probability", 0.33)

            # Likelihood értékek begyűjtése
            likelihood = self._calculate_likelihood(match_id)

            posterior = self._bayesian_formula(prior, likelihood)

            posterior_output[match_id] = {
                "probability": round(posterior, 4),
                "source": "BayesianUpdater"
            }

        return posterior_output

    # ----------------------------------------------------------
    # BAYES FORMULA
    # posterior = prior * likelihood  (normalizálva)
    # ----------------------------------------------------------
    def _bayesian_formula(self, prior, likelihood):
        # smoothing gátolja a 0 vagy túl nagy értékeket
        raw = (prior + self.smoothing) * (likelihood + self.smoothing)
        return float(min(1.0, max(0.0, raw)))

    # ----------------------------------------------------------
    # LIKELIHOOD SZÁMÍTÁS – több modellből
    # ----------------------------------------------------------
    def _calculate_likelihood(self, match_id):
        score = 0
        weight_sum = 0

        # Itt minden modell likelihood értékét vesszük
        from backend.pipeline.model_runner import ModelRunner

        # Meglévő modellek eredményei
        mode
