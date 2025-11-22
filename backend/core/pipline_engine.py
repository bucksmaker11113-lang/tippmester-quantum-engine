# backend/core/pipeline_engine.py

import time
import threading
from backend.utils.logger import get_logger

class PipelineEngine:
    """
    QUANTUM PIPELINE ENGINE – PRO EDITION
    -------------------------------------
    A teljes tippmotor agya.
    INTELLIGENT EVENT-TRIGGERED mód:
        • alacsony erőforrás használat (költségkímélő)
        • folyamatos figyelés (C opció)
        • csak akkor fut, amikor szükséges
    """

    def __init__(self, config, fusion, bayes, bias, value, selector, bankroll, kombi, live=None):
        self.config = config
        self.fusion = fusion
        self.bayes = bayes
        self.bias = bias
        self.value = value
        self.selector = selector
        self.bankroll = bankroll
        self.kombi = kombi
        self.live = live

        self.logger = get_logger()

        # időzítés / cooldown
        self.min_interval = config.get("pipeline", {}).get("min_interval_seconds", 30)
        self.last_run_time = 0

        # event flags
        self.event_new_odds = False
        self.event_new_match = False
        self.event_live_trigger = False

        # thread control
        self.running = False

    # ----------------------------------------------------------------------
    # PUBLIC: PIPELINE START
    # ----------------------------------------------------------------------
    def start(self):
        self.running = True
        self.logger.info("[Pipeline] Intelligent Pipeline Engine elindult.")

        thread = threading.Thread(target=self._loop)
        thread.daemon = True
        thread.start()

    # ----------------------------------------------------------------------
    # PUBLIC: STOP
    # ----------------------------------------------------------------------
    def stop(self):
        self.running = False
        self.logger.info("[Pipeline] Leállítva.")

    # ----------------------------------------------------------------------
    # EVENT TRIGGER FUNCTIONS
    # ----------------------------------------------------------------------
    def trigger_new_odds(self):
        self.event_new_odds = True

    def trigger_new_match(self):
        self.event_new_match = True

    def trigger_live_signal(self):
        self.event_live_trigger = True

    # ----------------------------------------------------------------------
    # MAIN LOOP (LOW RESOURCE)
    # ----------------------------------------------------------------------
    def _loop(self):
        while self.running:

            # csak akkor futunk, ha:
            # 1) van esemény
            # 2) letelt a minimális időköz
            if self._should_run():

                try:
                    self._run_pipeline_cycle()
                except Exception as e:
                    self.logger.error(f"[Pipeline] Hiba: {e}")

                self._reset_events()

            time.sleep(1)

    # ----------------------------------------------------------------------
    # DÖNTÉS: FUTTASSUK-E?
    # ----------------------------------------------------------------------
    def _should_run(self):
        now = time.time()

        if now - self.last_run_time < self.min_interval:
            return False

        if self.event_new_odds:
            return True
        if self.event_new_match:
            return True
        if self.event_live_trigger:
            return True

        # fallback: futtassuk 10 percenként takarékos módon
        if now - self.last_run_time > 600:
            return True

        return False

    # ----------------------------------------------------------------------
    # PIPELINE CYCLE
    # ----------------------------------------------------------------------
    def _run_pipeline_cycle(self):
        self.logger.info("[Pipeline] Új pipeline ciklus indul...")

        self.last_run_time = time.time()

        # 1️⃣ Fusion Engine
        events = self._get_events()
        fusion_results = self.fusion.combine(events)

        # 2️⃣ Bayesian Updater
        bayes_results = self.bayes.update(fusion_results)

        # 3️⃣ Bias Engine
        bias_results = self.bias.apply(bayes_results)

        # 4️⃣ Value Engine
        value_results = self.value.evaluate(bias_results)

        # 5️⃣ TipSelector Engine
        selected = self.selector.select(value_results)

        # 6️⃣ Bankroll Engine
        allocated = self.bankroll.allocate(selected["single_tips"])

        # 7️⃣ Kombi Engine
        kombis = self.kombi.build_kombis(selected["kombi_candidates"])

        # 8️⃣ Live Engine (ha élő esemény van)
        live_results = None
        if self.event_live_trigger and self.live:
            for match in self._get_live_matches():
                live_results = self.live.analyze_live_match(match)

        # 9️⃣ Final output összeállítása
        final_output = {
            "singles": allocated,
            "kombis": kombis,
            "live": live_results,
            "count": len(allocated)
        }

        # log
        self.logger.info("[Pipeline] Pipeline ciklus kész.")

        # TODO: save_final_output(final_output)
        return final_output

    # ----------------------------------------------------------------------
    # SEGÉD: MECCS ADATOK
    # ----------------------------------------------------------------------
    def _get_events(self):
        """
        Valóságban scraper vagy API töltené.
        Most: placeholder eseménylista.
        """
        return []

    def _get_live_matches(self):
        """
        Placeholder: élő meccsek listája.
        """
        return []
