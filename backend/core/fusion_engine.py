# backend/core/fusion_engine.py

from backend.utils.logger import get_logger

class FusionEngine:
    """
    Fusion Engine – Ensemble Layer 2
    A Quantum Synth Engine után végzi el a második súlyozást és stabilizálást.
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # Súlyozás a configból
        self.weights = config.get("fusion_layer2_weights", {
            "quantum_synth": 0.50,
            "mc3": 0.20,
            "lstm": 0.10,
            "gnn": 0.10,
            "poisson": 0.05,
            "rl": 0.05
        })

    # ----------------------------------------------------------
    # Második rétegű modell összehangolás
    # ----------------------------------------------------------
    def combine(self, model_outputs):
        self.logger.info("Fusion Engine: második ensemble réteg fut...")

        final = {}
        match_ids = self._collect_all_ids(model_outputs)

        for match_id in match_ids:

            combined = 0
            ws = 0

            for model_name, data in model_outputs.items():
                if match_id in data:
                    prob = data[match_id].get("probability", 0)
                    w = self.weights.get(model_name, 0)

                    combined += prob * w
                    ws += w

            final_prob = combined / ws if ws else 0

            final[match_id] = {
                "probability": round(final_prob, 4),
                "source": "FusionEngine"
            }

        return final

    # ----------------------------------------------------------
    # match id-k begyűjtése
    # ----------------------------------------------------------
    def _collect_all_ids(self, outputs):
        ids = set()
        for model_name, d in outputs.items():
            if isinstance(d, dict):
                for key in d:
                    ids.add(key)
        return ids
