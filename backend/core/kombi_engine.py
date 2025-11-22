# backend/core/kombi_engine.py

import itertools
import numpy as np
from backend.utils.logger import get_logger

class KombiEngine:
    """
    QUANTUM KOMBI ENGINE – PRO EDITION
    ----------------------------------
    Feladata:
        • Kombinációk létrehozása a TipSelector által adott jelöltekből
        • Odds-limit, risk-limit és correlation alapján optimalizál
        • Value-optimalizált kombi ajánlások
        • TippmixPro-barát struktúra (nem túl sok meccs, nem túl magas kockázat)
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # minimum/maximum kombi méret
        self.min_size = config.get("kombi", {}).get("min_size", 2)
        self.max_size = config.get("kombi", {}).get("max_size", 3)

        # odds limit: ne legyen túl nagy a végösszeg
        self.max_total_odds = config.get("kombi", {}).get("max_total_odds", 20)

        # correlation limit
        self.max_correlation = config.get("kombi", {}).get("max_correlation", 0.70)

        # minimum confidence egy kombi taghoz
        self.min_conf = config.get("kombi", {}).get("min_confidence", 0.55)

        # minimum value score
        self.min_value = config.get("kombi", {}).get("min_value_score", 0.15)

        # maximum risk ("túl instabil" kombi kizárva)
        self.max_risk = config.get("kombi", {}).get("max_risk", 0.75)

    # ----------------------------------------------------------------------
    # PUBLIC MAIN FUNCTION
    # ----------------------------------------------------------------------
    def build_kombis(self, kombi_candidates):
        """
        kombi_candidates = TipSelector által adott lista
        Return:
            [
                {
                    matches: [...],
                    total_odds,
                    avg_probability,
                    avg_confidence,
                    avg_value,
                    max_risk,
                    correlation_score,
                    source: "KombiEngine"
                }
            ]
        """

        self.logger.info("[KombiEngine] Kombi generálás indul...")

        if not kombi_candidates or len(kombi_candidates) < self.min_size:
            return []

        # generáljuk a 2-es, 3-as (max_size) kombikat
        all_kombis = []
        for size in range(self.min_size, self.max_size + 1):
            combos = itertools.combinations(kombi_candidates, size)
            for c in combos:
                kombi = self._evaluate_combo(list(c))
                if kombi:
                    all_kombis.append(kombi)

        # rendezés: legjobb kombik elől
        all_kombis.sort(key=lambda x: x["avg_confidence"] * x["avg_value"], reverse=True)

        return all_kombis[:10]  # maximum 10 ajánlás

    # ----------------------------------------------------------------------
    # KOMBI ÉRTÉKELÉSE
    # ----------------------------------------------------------------------
    def _evaluate_combo(self, tips):
        # odds szorzat
        total_odds = 1.0
        for t in tips:
            try:
                o = t["tippmix_odds"].get("1", 2.0)
            except:
                o = 2.0
            total_odds *= o

        if total_odds > self.max_total_odds:
            return None

        # számítsunk átlagokat
        probs = [t["probability"] for t in tips]
        confs = [t["confidence"] for t in tips]
        values = [t["value_score"] for t in tips]
        risks = [t["risk"] for t in tips]

        avg_prob = float(np.mean(probs))
        avg_conf = float(np.mean(confs))
        avg_value = float(np.mean(values))
        max_risk = float(np.max(risks))

        # minimum feltételek
        if avg_conf < self.min_conf:
            return None
        if avg_value < self.min_value:
            return None
        if max_risk > self.max_risk:
            return None

        # correlation = tippek mennyire hasonlítanak egymásra
        correlation = self._correlation_score(tips)
        if correlation > self.max_correlation:
            return None

        return {
            "matches": [t["match_id"] for t in tips],
            "total_odds": round(total_odds, 3),
            "avg_probability": round(avg_prob, 3),
            "avg_confidence": round(avg_conf, 3),
            "avg_value": round(avg_value, 3),
            "max_risk": round(max_risk, 3),
            "correlation_score": round(correlation, 3),
            "source": "KombiEngine"
        }

    # ----------------------------------------------------------------------
    # CORRELATION (meccsek összefüggése)
    # ----------------------------------------------------------------------
    def _correlation_score(self, tips):
        """
        Egyszerűsített megoldás:
            |prob1 - prob2| + |value1 - value2| / tip_szám
        Később:
            • Ha két meccs azonos ligában
            • Azonos csapat/nemzet
            • Azonos idősáv
        Akkor correlation nő.
        """

        if len(tips) < 2:
            return 0.0

        diffs = []
        for a, b in itertools.combinations(tips, 2):
            dp = abs(a["probability"] - b["probability"])
            dv = abs(a["value_score"] - b["value_score"])
            diffs.append(dp + dv)

        corr = np.mean(diffs) if diffs else 0.0

        # minél kisebb a különbség → annál jobban korrelálnak → veszélyesebb
        # magas érték = nagy eltérés = alacsony correlation veszély → jó
        normalized = max(0.0, 1.0 - corr)

        return float(normalized)
