# backend/pipeline/preprocess.py

import numpy as np
from backend.utils.logger import get_logger
from backend.core.temporal_model import TemporalModel
from backend.core.bias_engine import BiasEngine

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # Temporal + Bias modulok
        self.temporal = TemporalModel(config)
        self.bias = BiasEngine(config)

    # ----------------------------------------------------------
    # FŐ PREPROCESS METÓDUS
    # ----------------------------------------------------------
    def prepare(self, data):
        self.logger.info("Preprocessor: input adatok előfeldolgozása...")

        cleaned = {}

        # 1) Tippmix adatok tisztítása
        cleaned["events"] = self._clean_events(data.get("tippmix", {}))

        # 2) Odds drift számítása (szorzók változása)
        cleaned["odds_drift"] = self._compute_odds_drift(data)

        # 3) Temporal trendek előállítása
        cleaned["temporal"] = self.temporal.transform(cleaned["events"])

        # 4) Bias korrekció
        cleaned["bias_corrected"] = self.bias.apply(cleaned["events"])

        return cleaned

    # ----------------------------------------------------------
    # TISZTÍTÁS
    # ----------------------------------------------------------
    def _clean_events(self, events):
        output = []
        for e in events.values() if isinstance(events, dict) else []:
            try:
                if not all(k in e for k in ("home", "away", "odds")):
                    continue
                output.append(e)
            except:
                continue
        return output

    # ----------------------------------------------------------
    # ODDS DRIFT
    # ----------------------------------------------------------
    def _compute_odds_drift(self, data):
        drift = {}
        try:
            t1 = data.get("tippmix", {})
            t2 = data.get("oddsportal", {})

            for match_id in t1:
                if match_id in t2:
                    try:
                        o1 = t1[match_id]["odds"]
                        o2 = t2[match_id]["odds"]
                        drift[match_id] = float(o2) - float(o1)
                    except:
                        pass
        except Exception as e:
            self.logger.error(f"Odds drift hiba: {e}")

        return drift

