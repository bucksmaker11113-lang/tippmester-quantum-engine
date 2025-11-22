# backend/pipeline/data_loader.py
import json
import os
from backend.scrapers.tippmixpro_scraper import TippmixProScraper
from backend.scrapers.sofascore_scraper import SofaScoreScraper
from backend.scrapers.oddsportal_scraper import OddsPortalScraper
from backend.scrapers.odds_aggregator import OddsAggregator
from backend.utils.logger import get_logger

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger()
        self.cache_path = "backend/data/"

        # Scrapers inicializálása
        self.tippmix = TippmixProScraper(config)
        self.sofascore = SofaScoreScraper(config)
        self.oddsportal = OddsPortalScraper(config)
        self.aggregator = OddsAggregator()

    # ----------------------------------------------------------
    # FŐ FÜGGVÉNY: ADATOK BETÖLTÉSE
    # ----------------------------------------------------------
    def load_data(self):
        self.logger.info("DataLoader: adatok betöltése...")

        data = {}

        # Tippmix odds és események
        try:
            data["tippmix"] = self.tippmix.fetch()
        except Exception as e:
            self.logger.error(f"Tippmix betöltés hiba: {e}")
            data["tippmix"] = {}

        # SofaScore statisztikák
        try:
            data["sofa"] = self.sofascore.fetch_stats()
        except Exception as e:
            self.logger.error(f"Sofascore hiba: {e}")
            data["sofa"] = {}

        # OddsPortal historikus + élő odds
        try:
            data["oddsportal"] = self.oddsportal.fetch()
        except Exception as e:
            self.logger.error(f"Oddsportal hiba: {e}")
            data["oddsportal"] = {}

        # Aggregált odds
        data["aggregated"] = self.aggregator.combine(
            data.get("tippmix", {}),
            data.get("oddsportal", {})
        )

        # Cache mentése
        self._save_cache("events_cache.json", data)

        return data

    # ----------------------------------------------------------
    # CACHE KEZELÉS
    # ----------------------------------------------------------
    def _save_cache(self, filename, content):
        try:
            with open(os.path.join(self.cache_path, filename), "w", encoding="utf-8") as f:
                json.dump(content, f, indent=4)
        except Exception as e:
            self.logger.error(f"Cache mentési hiba ({filename}): {e}")

