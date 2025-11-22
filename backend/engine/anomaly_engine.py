# backend/engine/anomaly_engine.py

import numpy as np
from backend.utils.logger import get_logger

class AnomalyEngine:
    """
    ANOMALY ENGINE – PRO EDITION
    ------------------------------
    Feladata:
        • Gyanús oddsmozgások detektálása
        • Market manipulation minták felismerése
        • Szokatlan fogadási volumen (bet spikes)
        • Sharp/public torzulás extrém értékei
        • AI alapú anomaly score → win probability shift
        • Piaci hibák és extrém drift kiszűrése
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # scaling faktorok
        self.drift_scaling = config.get("anomaly", {}).get("drift_scaling", 0.25)
        self.spike_scaling = config.get("anomaly", {}).get("spike_scaling", 0.20)
        self.suspicion_scaling = config.get("anomaly", {}).get("suspicion_scaling", 0.30)

        # fallback
        self.fallback_prob = 0.50
        self.min_conf = config.get("anomaly", {}).get("min_confidence", 0.60)

    # ----------------------------------------------------------------------
    # PUBLIC: fő predikció
    # ----------------------------------------------------------------------
    def predict(self, match_data):
        outputs = {}

        for match_id, data in match_data.items():

            try:
                prob = self._anomaly_core(data)
            except Exception as e:
                self.logger.error(f"[Anomaly] Hiba → fallback: {e}")
                prob = self.fallback_prob

            prob = self._normalize(prob)
            conf = self._confidence(prob)
            risk = self._risk(prob, conf)

            outputs[match_id] = {
                "probability": round(prob, 4),
                "confidence": round(conf, 3),
                "risk": round(risk, 3),
                "meta": {
                    "drift_scaling": self.drift_scaling,
                    "spike_scaling": self.spike_scaling,
                    "suspicion_scaling": self.suspicion_scaling
                },
                "source": "Anomaly"
            }

        return outputs

    # ----------------------------------------------------------------------
    # ANOMALY MAG – MAIN LOGIC
    # ----------------------------------------------------------------------
    def _anomaly_core(self, data):
        """
        Várt input:
            • odds_open
            • odds_now
            • volume_open
            • volume_now
            • suspicious_mark (boolean / 0-1 AI flag)
            • drift_intensity (0–1)
        """

        odds_open = data.get("odds_open", 2.00)
        odds_now  = data.get("odds_now", 2.00)

        volume_open = data.get("volume_open", 1000)
        volume_now  = data.get("volume_now", 1000)

        suspicious_mark = data.get("suspicious_mark", 0.0)
        drift_intensity = data.get("drift_intensity", 0.0)

        # ODDS DRIFT
        drift = odds_open - odds_now
        drift_effect = drift * self.drift_scaling

        # BET SPIKE (fogadási volumen ugrás)
        if volume_open > 0:
            volume_spike = (volume_now - volume_open) / max(volume_open, 1)
        else:
            volume_spike = 0

        spike_effect = volume_spike * self.spike_scaling

        # SUSPICIOUS MARK (AI jelzés / külső adat)
        suspicion_effect = suspicious_mark * self.suspicion_scaling

        # Combined anomaly shift
        total_shift = drift_effect + spike_effect + suspicion_effect

        prob = 0.5 + total_shift
        return float(prob)

    # ----------------------------------------------------------------------
    # NORMALIZÁLÁS
    # ----------------------------------------------------------------------
    def _normalize(self, p):
        return float(max(0.01, min(0.99, p)))

    # ----------------------------------------------------------------------
    # CONFIDENCE
    # ----------------------------------------------------------------------
    def _confidence(self, prob):
        stability = 1 - abs(prob - 0.5)
        conf = max(self.min_conf, min(1.0, 0.6 + stability * 0.4))
        return float(conf)

    # ----------------------------------------------------------------------
    # RISK
    # ----------------------------------------------------------------------
    def _risk(self, prob, conf):
        """
        Minél gyanúsabb a market, annál nagyobb a risk.
        """
        return float(min(1.0, max(0.0, (1 - prob) * 0.4 + (1 - conf) * 0.6)))
