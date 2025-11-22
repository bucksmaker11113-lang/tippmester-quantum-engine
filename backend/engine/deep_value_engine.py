# backend/engine/deep_value_engine.py

import os
import torch
import numpy as np
from backend.utils.logger import get_logger

class DeepValueNet(torch.nn.Module):
    """
    Deep Neural Network for Value Prediction
    ---------------------------------------
    Bemenet:
        ~100-300 dimenzió, az összes engine összefűzött outputja

    Kimenet:
        value_score (0–1)
    """
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),

            torch.nn.Linear(hidden_dim//2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class DeepValueEngine:
    """
    Deep Value Engine – A rendszer mélytanulásos agya
    -------------------------------------------------
    Feladata:
        - összegyűjti az összes engine numerikus adatát
        - betölti a tanult neurális hálót
        - value score-t ad egy meccsre
        - value_engine előtt fut, és brutálisan javítja az EV pontosságot
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        self.model_path = "backend/data/models/model_weights.pth"
        self.input_dim = config.get("deep_value", {}).get("input_dim", 128)
        self.model = DeepValueNet(input_dim=self.input_dim)

        self.device = "cpu"
        self.model.to(self.device)

        # model betöltése, ha létezik
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
                self.logger.info("[DeepValue] Model weights loaded.")
            except Exception as e:
                self.logger.error(f"[DeepValue] Could not load weights → {e}")
        else:
            self.logger.warning("[DeepValue] No model weights found → cold start.")

        self.model.eval()

    # --------------------------------------------------
    # fő predikciós függvény
    # --------------------------------------------------
    def predict(self, feature_vectors):
        """
        feature_vectors:
            {
                match_id: {
                    "features": [ ... ~100-300 szám ... ]
                }
            }
        """
        results = {}

        for match_id, data in feature_vectors.items():
            fv = np.array(data.get("features", []), dtype=np.float32)

            # ha nincs elég input → skip
            if len(fv) == 0 or len(fv) != self.input_dim:
                results[match_id] = {
                    "value_score": 0.5,
                    "confidence": 0.55,
                    "risk": 0.45,
                    "source": "DeepValueEngine (fallback)"
                }
                continue

            x = torch.tensor(fv).float().to(self.device)
            with torch.no_grad():
                score = float(self.model(x).cpu().numpy())

            score = max(0.01, min(0.99, score))

            # confidence = minél távolabb 0.5-től → annál erősebb jel
            confidence = float(min(1.0, 0.55 + abs(score - 0.5)))

            # risk = inverz
            risk = float(1 - confidence)

            results[match_id] = {
                "value_score": round(score, 4),
                "confidence": round(confidence, 3),
                "risk": round(risk, 3),
                "source": "DeepValueEngine"
            }

        return results
