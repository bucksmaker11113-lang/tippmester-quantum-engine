# backend/engine/closing_line_predictor.py

import numpy as np
import math
import random


class ClosingLinePredictor:
    """
    CLOSING LINE PREDICTOR ENGINE (CLP)
    -----------------------------------
    Feladata:
        - oddsmozgás modellezése (nyitó → záró)
        - sharp money hatás detektálása
        - value stabilitás becslése
        - expected closing odds számítása
        - tipp időzítésének optimalizálása (most vagy később?)
        - closing line value (CLV) számítása

    A CLV a profi fogadásban az egyik legfontosabb mutató.
    """

    def __init__(self, config=None):
        self.config = config or {}

    # ---------------------------------------------------------
    # SHARP MONEY DETECTION
    # ---------------------------------------------------------
    def _sharp_money_signal(self, odds_history):
        """
        odds_history például:
        [1.80, 1.78, 1.76, 1.74]

        Ha gyors esés → sharp pénz.
        Ha lassú esés → public pénz.
        Ha ingadozó → instabil piac.
        """

        if len(odds_history) < 3:
            return 0.0  # nincs elég adat

        diffs = np.diff(odds_history)

        # Ha többször csökken → sharp money
        sharp_moves = sum(1 for x in diffs if x < 0)

        score = sharp_moves / len(diffs)

        return round(score, 3)

    # ---------------------------------------------------------
    # VOLATILITY (piaci instabilitás)
    # ---------------------------------------------------------
    def _volatility(self, odds_history):
        if len(odds_history) < 3:
            return 0.1
        return float(np.std(odds_history))

    # ---------------------------------------------------------
    # EXPECTED CLOSING LINE
    # ---------------------------------------------------------
    def predict_closing_line(self, current_odds, odds_history):
        """
        Return:
            - expected_closing
            - sharp_money_score
            - volatility
        """

        sharp = self._sharp_money_signal(odds_history)
        vol = self._volatility(odds_history)

        # A legegyszerűbb jó modell:
        # closing odds = current odds - sharp*0.05 + random small noise
        movement = -sharp * 0.05 + np.random.uniform(-vol/10, vol/10)

        expected = current_odds + movement

        # Clamping 1.01-20.00
        expected = float(max(1.01, min(20.0, expected)))

        return {
            "expected_closing": round(expected, 3),
            "sharp_money": sharp,
            "volatility": round(vol, 4)
        }

    # ---------------------------------------------------------
    # CLOSING LINE VALUE (CLV) SZÁMÍTÁS
    # ---------------------------------------------------------
    def clv(self, current_odds, expected_closing):
        """
        CLV = (expected closing odds - current odds) / current odds
        Pozitív → jó value, profi edge
        Negatív → kerülendő tipp
        """
        return round((expected_closing - current_odds) / current_odds, 4)
