# backend/core/bias_engine.py

import numpy as np
from backend.utils.logger import get_logger

class BiasEngine:
    """
    QUANTUM BIAS ENGINE – PRO EDITION
    ---------------------------------
    Feladata a torzítások detektálása és korrigálása:
        • odds drift (gyors piaci reagálások)
        • market pressure (public money szint)
        • model deviation (modellszórás alapú torzítás)
        • form / anomaly detection
        • volatility alapú bias weighting
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        self.bias_weights = config.get("bias_weights", {
            "odds_bias": 0.35,
            "market_bias": 0.30,
            "model_bias": 0.20,
            "team_bias": 0.15
        })

    # -------------------------------------------------------------------
    #                    FŐ BIAS KORREKCIÓS FUNKCIÓ
    # -------------------------------------------------------------------
    def apply(self, bayes_output):
        """
        Input:
            bayes_output = BayesianUpdater posterior probability set
        Output:
            {
                match_id: {
                    probability: ...,
                    bias_components: {...},
                    source: "BiasEngine"
                }
            }
        """

        self.logger.info("[BiasEngine] Bias korrekció indul...")

        corrected = {}

        for match_id, pred in bayes_output.items():

            prob = pred.get("probability", 0.33)
