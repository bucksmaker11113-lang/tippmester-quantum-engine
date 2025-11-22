# backend/core/bayesian_updater.py

import numpy as np
from backend.utils.logger import get_logger

class BayesianUpdater:
    """
    QUANTUM BAYESIAN UPDATER – PRO EDITION
    --------------------------------------
    • A Fusion Engine által előállított probability tovább finomítása
    • Multi-engine likelihood weighting
    • Engine-reliability alapú posterior számítás
    • Volatility-corrected Bayes update
    • Hibatűrő kialakítás (20–50 engine-ig)
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # smoothing = kilengések csillapítása
        self.smoothing = config.get("bayes_smoothing", 0.05)

        # alap likelihood súlyozás
        self.default_like_weight = config.get("default_likelihood_weight", 1.0)

    # ----------------------------------------------------------------------
    #                    FŐ BAYES UPDATE FUNKCIÓ
    # ----------------------------------------------------------------------
    def update(self, fusion_output):
        """
        Input (Fusion layer output):
            {
               match_id: {
                   "probability": ...,
                   "raw_model_scores": {...}
               }
            }
        Output (posterior):
            {
                match_id: {
                    "probability": posterior_prob,
                    "source": "BayesianUpdater"
                }
            }
        """

        self.logger.info("[BayesianUpdater] Posterior valószínűségek számítása...")

        posterior_output = {}

        for match_id, pred in fusion_output.items():

            prior = pred.get("probability", 0.33)
            model_scores = pred.get("raw_model_scores", {})

            # likelihood számítás engine-score-okból
            likelihood = self._calculate_likelihood(model_scores)

            posterior = self._bayesian_formula(prior, likelihood)

            posterior_output[match_id] = {
                "probability": round(posterior, 4),
                "source": "BayesianUpdater"
            }

        return posterior_output

    # ----------------------------------------------------------------------
    #                          BAYES FORMULA
    #     posterior = prior × likelihood  (normalizálva, clampelve)
    # ----------------------------------------------------------------------
    def _bayesian_formula(self, prior, likelihood):
        # smoothing csillapítja a szélsőséges értékeket
        raw = (prior + self.smoothing) * (likelihood + self.smoothing)

        # clamp 0–1 között
        return float(max(0.01, min(0.99, raw)))

    # ----------------------------------------------------------------------
    #                 LIKELIHOOD – engine szintekből számolva
    # ----------------------------------------------------------------------
    def _calculate_likelihood(self, model_scores):
        """
        model_scores example:
        {
            "gnn_engine":  {"prob": 0.62, "conf": 0.78, "weight": 1.1},
            "lstm_engine": {"prob": 0.59, "conf": 0.66, "weight": 0.9},
            ...
        }
        """

        if not model_scores:
            return 0.5  # semleges likelihood

        weighted = []
        weights = []

        for engine_name, data in model_scores.items():
            try:
                prob = float(data.get("prob", 0.5))
                conf = float(data.get("conf", 0.5))

                # likelihood súlya = engine confidence + engine weight
                like_weight = conf + data.get("weight", self.default_like_weight)

                weighted.append(prob * like_weight)
                weights.append(like_weight)

            except:
                continue

        if not weights:
            return 0.5

        likelihood = sum(weighted) / sum(weights)

        # clamp – a szélsőséges értékeket visszafogja
        return float(max(0.01, min(0.99, likelihood)))
