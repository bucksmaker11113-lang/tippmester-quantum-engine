# backend/core/bankroll_engine.py

import numpy as np
from backend.utils.logger import get_logger

class BankrollEngine:
    """
    BANKROLL ENGINE
    ----------------
    Feladata:
        - tétméretezés (stake sizing)
        - Kelly formula használata
        - AI alapú tétkorrekció (DeepValue visszacsatolás)
        - bankroll védelem (stop-loss)
        - napi max tét limit
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.logger = get_logger()

        self.base_stake = self.config.get("base_stake_pct", 0.01)   # 1%
        self.daily_limit = self.config.get("daily_limit_pct", 0.05) # 5%
        self.stop_loss = self.config.get("stop_loss_pct", 0.15)     # 15%

        self.daily_used = 0.0
        self.session_profit = 0.0

    # ----------------------------------------------------
    # KELLY FORMULA
    # ----------------------------------------------------
    def kelly(self, prob, odds):
        """
        Kelly formula (ajánlott tét arányban).
        f* = (bp - q) / b
        """
        b = odds - 1
        q = 1 - prob

        f = ((b * prob) - q) / b
        if f < 0:
            return 0
        return min(0.05, f)   # max 5%

    # ----------------------------------------------------
    # FŐ FÜGGVÉNY – TÉT SZÁMÍTÁS
    # ----------------------------------------------------
    def stake(self, bankroll, prob, odds, deep_value):
        """
        bankroll: teljes tőke
        prob: model probability
        odds: selected odds
        deep_value: 0–1 között (AI erősség)
        """

        # 1) Kelly tét
        k = self.kelly(prob, odds)

        # 2) DeepValue alapján korrekció
        dv_adj = deep_value * 0.02   # 0%–2% extra kockázat

        stake_pct = self.base_stake + k + dv_adj

        # 3) napi korlát
        if self.daily_used + stake_pct > self.daily_limit:
            stake_pct = max(0, self.daily_limit - self.daily_used)

        # 4) stop-loss védelem
        if self.session_profit < -self.stop_loss:
            stake_pct = 0  # leáll

        self.daily_used += stake_pct

        return round(bankroll * stake_pct, 2)

    # ----------------------------------------------------
    # PROFIT UPDATE
    # ----------------------------------------------------
    def update_profit(self, profit):
        self.session_profit += profit
