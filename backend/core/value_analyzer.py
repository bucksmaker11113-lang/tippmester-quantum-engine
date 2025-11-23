# backend/core/value_analyzer.py

import numpy as np
from backend.utils.logger import get_logger


class ValueAnalyzer:
    """
    QUANTUM VALUE ENGINE – PRO EDITION
    -----------------------------------
    Feladata:
        • Nemzetközi odds aggregálás (multi-bookmaker)
        • Piaci középár számítása (fair odds)
        • Odds spread → market disagreement detektálás
        • Volatility korrekció
        • EV = p * TMX_odds - (1 - p)
        • Value Score számítás
        • TippmixPro odds validáció
    """

    def __init__(self, config, intl_scraper, tmx_scraper):
        self.config = config
        self.logger = get_logger()

        self.intl = intl_scraper
        self.tmx = tmx_scraper

        self.min_bookmakers = self.config.get("value", {}).get("min_bookmakers", 3)
        self.volatility_weight = self.config.get("value", {}).get("volatility_weight", 0.15)
        self.spread_weight = self.config.get("value", {}).get("spread_weight", 0.25)

    # =====================================================================
    # 1) Nemzetközi oddsok aggregálása
    # =====================================================================
    def _aggregate_intl_odds(self, intl_data):
        """
        intl_data:
            {
                "1": [1.88, 1.90, 1.93, 1.85],
                "X": [3.45, 3.50, 3.60],
                "2": [4.10, 4.20, 4.30, 4.05]
            }

        return:
            avg odds per outcome
        """

        out = {}

        for k, v in intl_data.items():
            if not v or len(v) < self.min_bookmakers:
                out[k] = None
            else:
                out[k] = float(np.mean(v))

        return out

    # =====================================================================
    # 2) Spread – market disagreement
    # =====================================================================
    def _market_spread(self, values):
        v = [x for x in values if x is not None]
        if len(v) < 2:
            return 0.0
        return float(np.std(v))

    # =====================================================================
    # 3) Fair probability → fair odds
    # =====================================================================
    def _fair_probability(self, avg_o_
