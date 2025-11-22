# backend/engine/quantum_synth_engine.py

from backend.utils.logger import get_logger

class QuantumSynthEngine:
    """
    Quantum Synth Engine
    Több modell eredményének elsődleges összehangolása.

    Input:
        - model_outputs dictionary:
            {
                "mc3": {...},
                "lstm": {...},
                "gnn": {...},
                "poisson": {...},
                "rl": {...},
                "gameflow": {...},
                "value_model": {...}
            }

    Output:
        - per-event összehangolt valószínűségek
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # Súlyok a configból
        self.weights = config.get("fusion_weights", {
            "mc3": 0.35,
            "lstm": 0.20,
            "gnn": 0.15,
            "poisson": 0.10,
            "rl": 0.05,
            "gameflow": 0.10,
            "value_model": 0.05
        })

    # ----------------------------------------------------------
    # FŐ FÜGGVÉNY: MODELLÖSSZEVONÁS
    # ----------------------------------------------------------
    def fuse(self, model_outputs):
        self.logger.info("QuantumSynth: modellek első rétegű összehangolása...")

        final = {}

        # Megnézzük minden modell eredményét és meccseket egyesítjük
        match_ids = self._collect_all_match_ids(model_outputs)

        for match_id in match_ids:
            combined_score = 0
            weight_sum = 0

            # Modellek végigmennek
            for model_name, result_dict in model_outputs.items():
                if match_id in result_dict:
                    prob = result_dict[match_id].get("probability", 0)
                    w = self.weights.get(model_name, 0)

                    combined_score += prob * w
                    weight_sum += w

            if weight_sum == 0:
                final_prob = 0
            else:
                final_prob = combined_score / weight_sum

            final[match_id] = {
                "probability": round(final_prob, 4),
                "source": "QuantumSynth"
            }

        return final

    # ----------------------------------------------------------
    # HELPER – MATCH ID-K ÖSSZEGYŰJTÉSE
    # ----------------------------------------------------------
    def _collect_all_match_ids(self, outputs):
        ids = set()
        for model_name, model_dict in outputs.items():
            if isinstance(model_dict, dict):
                for k in model_dict.keys():
                    ids.add(k)
        return list(ids)
