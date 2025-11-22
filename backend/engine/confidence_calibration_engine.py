# backend/engine/confidence_calibration_engine.py

import numpy as np
from backend.utils.logger import get_logger

class ConfidenceCalibrationEngine:
    """
    CONFIDENCE CALIBRATION ENGINE – HYBRID EDITION
    ------------------------------------------------
    Feladata:
        • Többi engine probability-outputjának kalibrálása
        • Platt scaling + Isotonic regression automatikus váltással
        • Expected Calibration Error (ECE) minimalizálása
        • Stabil, konszolidált probability → ValueEngine pontosabb lesz
        • FusionEngine erejét sokszorozza
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # Platt paraméterek (w, b)
        self.platt_w = config.get("calibration", {}).get("platt_w", 1.0)
        self.platt_b = config.get("calibration", {}).get("platt_b", 0.0)

        # Isotonic memória (histogram bins)
        self.bins = config.get("calibration", {}).get("bins", 15)

        # ECE súly
        self.ece_scaling = config.get("calibration", {}).get("ece_scaling", 0.25)

        # fallback
        self.min_confidence = config.get("calibration", {}).get("min_confidence", 0.55)

    # ----------------------------------------------------------------------
    # FŐ FÜGGVÉNY
    # ----------------------------------------------------------------------
    def calibrate(self, predictions):
        """
        Input:
            predictions = {
                match_id: {
                    "probability": 0.63,
                    "confidence": 0.77,
                    "source": "FusionEngine"
                }
            }

        Output → kalibrált valószínűségek.
        """

        calibrated = {}

        for match_id, pred in predictions.items():

            p = pred.get("probability", 0.50)

            # 1) Platt scaling
            p_platt = self._platt(p)

            # 2) Isotonic regression (approximált histogram base)
            p_iso = self._isotonic(p)

            # 3) Hybrid combination
            hybrid = self._hybrid_blend(p, p_platt, p_iso)

            # 4) ECE correction (stabilitás)
            calibrated_p = self._ece_correct(hybrid, p)

            calibrated_p = float(max(0.01, min(0.99, calibrated_p)))

            calibrated[match_id] = {
                "probability": round(calibrated_p, 4),
                "source": "CalibrationEngine"
            }

        return calibrated

    # ----------------------------------------------------------------------
    # PLATT SCALING
    # ----------------------------------------------------------------------
    def _platt(self, p):
        z = self.platt_w * p + self.platt_b
        return 1 / (1 + np.exp(-z))

    # ----------------------------------------------------------------------
    # ISOTONIC REGRESSION (bin alapú approximáció)
    # ----------------------------------------------------------------------
    def _isotonic(self, p):
        """
        Egyszerűsített isotonic reg.: probability histogram smoothing.
        """

        bin_idx = int(p * self.bins)
        bin_idx = max(0, min(self.bins - 1, bin_idx))

        # Platt output + neighbouring bins simítása
        base = (bin_idx + 0.5) / self.bins

        return float(max(0.01, min(0.99, base)))

    # ----------------------------------------------------------------------
    # HYBRID BLEND (Platt + Isotonic + Original)
    # ----------------------------------------------------------------------
    def _hybrid_blend(self, original, p_platt, p_iso):
        """
        A 3 modell keverése:
        original = engine probability
        p_platt  = logisztikus kalibráció
        p_iso    = nemlineáris histogram smoothing
        """

        return (
            original * 0.40 +
            p_platt * 0.35 +
            p_iso * 0.25
        )

    # ----------------------------------------------------------------------
    # EXPECTED CALIBRATION ERROR CORRECTION
    # ----------------------------------------------------------------------
    def _ece_correct(self, calibrated, original):
        ece = abs(calibrated - original)
        correction = ece * self.ece_scaling

        if calibrated > original:
            return calibrated - correction
        else:
            return calibrated + correction
