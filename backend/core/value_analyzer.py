# backend/core/value_analyzer.py

import os
import importlib
import numpy as np
from backend.utils.logger import get_logger

class ValueAnalyzer:
    """
    QUANTUM VALUE ENGINE – PRO EDITION
    ----------------------------------
    • 10+ Nemzetközi bukméker scrape (plug-in rendszer)
    • Global odds → EV számítás
    • Value Score → volatility + spread alapján
    • TippmixPro validáció külön modulból
    • Ha nincs a TippmixPro-n → új value tipp szükséges
    • Ha rossz az odds → új value tipp szükséges
    • Risk + Confidence + Meta output
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # bukméker scrapers mappa
        self.bookmaker_dir = "backend/scrapers/international"
        self.tippmix_dir = "backend/scrapers/tippmixpro"

        # minimum value thresholds
        self.min_ev = config.get("value_thresholds", {}).get("min_ev", 0.03)
        self.min_value_score = config.get("value_thresholds", {}).get("min_value_score", 0.15)

        # scraper-ek dinamikus betöltése
        self.bookmaker_scrapers = self._discover_bookmakers()
        self.tippmixpro_scraper = self._load_tippmixpro()

    # -------------------------------------------------------------------
    #                   SCRAPER BETÖLTÉS
    # -------------------------------------------------------------------
    def _discover_bookmakers(self):
        scrapers = {}

        for file in os.listdir(self.bookmaker_dir):
            if file.endswith("_scraper.py"):
                name = file.replace(".py", "")

                try:
                    module = importlib.import_module(
                        f"backend.scrapers.international.{name}"
                    )
                    cls = self._guess_class(name)
                    scraper = getattr(module, cls, None)

                    if scraper:
                        scrapers[name] = scraper(self.config)
                        self.logger.info(f"[ValueEngine] Betöltve: {name}")

                except Exception as e:
                    self.logger.warning(f"[ValueEngine] SKIPPED: {name} ({e})")

        return scrapers

    # tippmixpro scraper külön mappában
    def _load_tippmixpro(self):
        try:
            for file in os.listdir(self.tippmix_dir):
                if file.endswith("_scraper.py"):
                    name = file.replace(".py", "")
                    module = importlib.import_module(
                        f"backend.scrapers.tippmixpro.{name}"
                    )
                    cls = self._guess_class(name)
                    return getattr(module, cls)(self.config)

        except Exception as e:
            self.logger.error(f"[ValueEngine] Tippmix scraper error: {e}")

        return None

    def _guess_class(self, filename):
        parts = filename.split("_")
        return "".join([p.capitalize() for p in parts])

    # -------------------------------------------------------------------
    #                      FŐ FÜGGVÉNY – VALUE SZÁMÍTÁS
    # -------------------------------------------------------------------
    def evaluate(self, bias_corrected):
        """
        Input: BiasEngine output
        Output:
        {
            match_id: {
                ev: ...,
                value_score: ...,
                risk: ...,
                confidence: ...,
                global_odds: {...},
                tippmix_odds: {...},
                playable_on_tippmix: True/False
            }
        }
        """
        self.logger.info("[ValueEngine] Value számítás indul...")

        results = {}

        for match_id, pred in bias_corrected.items():

            # előre készítjük a meta fieldet
            result = {
                "probability": pred.get("probability", 0.33),
                "global_odds": {},
                "tippmix_odds": {},
                "ev": 0.0,
                "value_score": 0.0,
                "risk": 1.0,
                "confidence": 0.0,
                "playable_on_tippmix": False,
                "source": "ValueEngine"
            }

            prob = result["probability"]

            # ---------------------------------------------------------------
            # 1) GLOBAL ODDS – 10+ bukméker scrapelése
            # ---------------------------------------------------------------
            global_odds = self._collect_global_odds(match_id)

            if not global_odds:
                self.logger.warning(f"[ValueEngine] Nincs global odds: {match_id}")
                continue

            result["global_odds"] = global_odds

            # ---------------------------------------------------------------
            # 2) EXPECTED VALUE (EV)
            # EV = prob * odds – (1 - prob)
            # ---------------------------------------------------------------
            ev = prob * global_odds["1"] - (1 - prob)
            result["ev"] = round(ev, 4)

            # értékelés – minimum EV alatt skip
            if ev < self.min_ev:
                results[match_id] = result
                continue

            # ---------------------------------------------------------------
            # 3) VALUE SCORE – normalizált EV
            # ---------------------------------------------------------------
            value_score = float(np.tanh(ev * 2.5))
            result["value_score"] = round(value_score, 3)

            if value_score < self.min_value_score:
                results[match_id] = result
                continue

            # ---------------------------------------------------------------
            # 4) TIPPMIXPRO VALIDÁCIÓ
            # ---------------------------------------------------------------
            playable, tm_odds = self._tippmixpro_check(match_id)

            result["tippmix_odds"] = tm_odds
            result["playable_on_tippmix"] = playable

            if not playable:
                # nincs a tippmixpro-n → új tipp szükséges
                self.logger.info(f"[ValueEngine] NINCS TIPPMIXPRO: {match_id}")
                results[match_id] = result
                continue

            # Tippmix odds rossz?
            if tm_odds and tm_odds.get("1", 0) < global_odds["1"] * 0.85:
                # 15%-nál jobban eltér → value elvész → új tipp
                results[match_id] = result
                continue

            # ---------------------------------------------------------------
            # 5) RISK score
            # ---------------------------------------------------------------
            risk = (1 - prob) * 0.7 + (1 - value_score) * 0.3
            result["risk"] = round(max(0, min(1, risk)), 3)

            # ---------------------------------------------------------------
            # 6) CONFIDENCE
            # ---------------------------------------------------------------
            confidence = prob * 0.4 + value_score * 0.4 + (1 - risk) * 0.2
            result["confidence"] = round(max(0, min(1, confidence)), 3)

            results[match_id] = result

        return results

    # -------------------------------------------------------------------
    #                GLOBAL ODDS AGGREGÁLÓ (MULTI-BUKI)
    # -------------------------------------------------------------------
    def _collect_global_odds(self, match_id):
        odds_list = []

        for name, scraper in self.bookmaker_scrapers.items():
            try:
                data = scraper.get_odds(match_id)
                if data and "1" in data:
                    odds_list.append(data["1"])
            except:
                pass

        if not odds_list:
            return None

        # global odds = MEDIAN a bukik között
        global_o1 = float(np.median(odds_list))

        return {"1": global_o1}

    # -------------------------------------------------------------------
    #                   TIPPMIXPRO VALIDÁTOR
    # -------------------------------------------------------------------
    def _tippmixpro_check(self, match_id):
        if not self.tippmixpro_scraper:
            self.logger.error("[ValueEngine] Nincs TippmixPro scraper.")
            return False, {}

        try:
            data = self.tippmixpro_scraper.get_odds(match_id)

            if not data:
                return False, {}

            return True, data  # data kimenete: {"1": odds, ...}

        except Exception as e:
            self.logger.warning(f"[ValueEngine] TippmixPro error: {e}")
            return False, {}
