# backend/core/feature_builder.py

import numpy as np

class FeatureBuilder:
    """
    Összefűzi az összes engine outputot egy 128 dimenziós feature vectorrá.
    Ez kerül a DeepValueEngine bemenetére.
    """

    def __init__(self, config):
        self.config = config
        self.feature_size = config.get("deep_value", {}).get("input_dim", 128)

    def build_feature_vector(self, engine_outputs):
        """
        engine_outputs = {
            "montecarlo": {...},
            "lstm": {...},
            "gnn": {...},
            "poisson": {...},
            ...
        }

        Minden engine-ből 3 értéket veszünk:
            • probability
            • confidence
            • risk

        Majd kitöltjük paddinggel 128 dimenzióra.
        """

        vec = []

        for eng, data in engine_outputs.items():
            p = float(data.get("probability", data.get("value_score", 0.5)))
            c = float(data.get("confidence", 0.5))
            r = float(data.get("risk", 0.5))

            vec.extend([p, c, r])

        # ha kevesebb mint 128 → kitöltjük
        if len(vec) < self.feature_size:
            vec.extend([0.0] * (self.feature_size - len(vec)))

        # ha több → levágjuk
        vec = vec[:self.feature_size]

        return np.array(vec, dtype=np.float32)
