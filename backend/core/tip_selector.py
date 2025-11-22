# backend/core/tip_selector.py

import numpy as np
from backend.utils.logger import get_logger

class TipSelector:
    """
    QUANTUM TIPSELECTOR – PRO EDITION
    ----------------------------------
    A tippek végső kiválasztásáért felelős modul.
    3 fő funkció:
        • SINGLE TIPPEK kiválasztása
        • KOMBI TIPPEK előkészítése
        • LIVE TIPPEK jelölése
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # SINGLE TIP LIMIT
        self.max_singles = config.get("selector", {}).get("max_single_tips", 4)

        # minimum quality thresholds
        self.min_conf = config.get("selector", {}).get("min_confidence", 0.55)
        self.min_value = config.get("selector", {}).get("min_value_score", 0.15)
        self.max_risk = config.get("selector", {}).get("max_risk", 0.75)

        # kombi thresholds (előfeltétel)
        self.kombi_enabled = True
        self.min_kombi_conf = 0.58

    # -----------------------------------------------------------------------
    # FŐ SELECTOR FUNKCIÓ
    # -----------------------------------------------------------------------
    def select(self, value_output):
        """
        value_output = Value Engine eredménye
        Return:
            {
                single_tips: [...],
                kombi_tips: [...],
                live_candidates: [...],
                count: ...
            }
        """

        self.logger.info("[TipSelector] Tipp kiválasztás indul...")

        # teljes lista
        tips = self._filter_valid(value_output)

        # SINGLE TIPPEK
        single_tips = self._select_single(tips)

        # KOMBI TIPPEK (előre jelölve)
        kombi_candidates = self._select_kombi_candidates(tips)

        # LIVE TIPP jelölés (valós drift + volatility később)
        live_candidates = self._select_live_candidates(tips)

        return {
            "single_tips": single_tips,
            "kombi_candidates": kombi_candidates,
            "live_candidates": live_candidates,
            "count": len(single_tips)
        }

    # -----------------------------------------------------------------------
    # FILTER: csak a valóban játszható tippek
    # -----------------------------------------------------------------------
    def _filter_valid(self, value_output):
        valid = []

        for match_id, data in value_output.items():

            if not data.get("playable_on_tippmix", False):
                continue

            prob = data.get("probability", 0.33)
            conf = data.get("confidence", 0.0)
            value_score = data.get("value_score", 0.0)
            risk = data.get("risk", 1.0)

            # alap feltételek
            if conf < self.min_conf:
                continue

            if value_score < self.min_value:
                continue

            if risk > self.max_risk:
                continue

            valid.append({
                "match_id": match_id,
                "probability": prob,
                "confidence": conf,
                "value_score": value_score,
                "risk": risk,
                "global_odds": data.get("global_odds", {}),
                "tippmix_odds": data.get("tippmix_odds", {}),
                "source": "TipSelector"
            })

        # confidence szerint sorbarendezve
        valid.sort(key=lambda x: x["confidence"], reverse=True)

        return valid

    # -----------------------------------------------------------------------
    # SINGLE TIPS kiválasztása
    # -----------------------------------------------------------------------
    def _select_single(self, tips):
        if not tips:
            return []

        # LIMITÁLT SINGLE LISTA
        return tips[:self.max_singles]

    # -----------------------------------------------------------------------
    # KOMBI TIPS – előkészítés
    # -----------------------------------------------------------------------
    def _select_kombi_candidates(self, tips):
        """
        Kombi tipp = legalább közepes confidence + jó value
        A Kombi Engine külön modul lesz, itt csak jelöljük.
        """
        kombi = []

        for t in tips:
            if (
                t["confidence"] >= self.min_kombi_conf and
                t["risk"] <= 0.70 and
                t["value_score"] >= 0.18
            ):
                kombi.append(t)

        return kombi

    # -----------------------------------------------------------------------
    # LIVE TIPP jelölés – előkészítő logika
    # -----------------------------------------------------------------------
    def _select_live_candidates(self, tips):
        """
        Live jelölés:
        • magas volatility gyanú
        • odds drift határ közelében (később scraperből jön)
        • magas confidence + gyors változás jelek
        """

        live = []

        for t in tips:
            # amíg nincs drift adat → egy egyszerű szabály:
            if t["confidence"] > 0.65 and t["value_score"] > 0.20:
                live.append(t)

        return live
