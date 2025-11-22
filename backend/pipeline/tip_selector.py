# backend/pipeline/tip_selector.py

from backend.utils.logger import get_logger

class TipSelector:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # Thresholdok
        self.min_probability = config["thresholds"]["min_probability"]
        self.min_value_score = config["thresholds"]["min_value"]
        self.min_confidence = config["thresholds"]["min_confidence"]

    # ----------------------------------------------------------
    # FŐ TIPPVÁLASZTÓ
    # ----------------------------------------------------------
    def select(self, fused_predictions):
        self.logger.info("TipSelector: tippek kiválasztása...")

        tips = []

        for match_id, pred in fused_predictions.items():

            try:
                prob = pred.get("probability", 0)
                value = pred.get("value_score", 0)
                conf
