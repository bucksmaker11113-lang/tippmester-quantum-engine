# backend/core/bankroll_engine.py

import numpy as np
from backend.utils.logger import get_logger


class BankrollEngine:
    """
    BANKROLL ENGINE – PRO VERSION
    -----------------------------
    Feladata:
        - Stake sizing Kelly-formulával
        - Value-alapú tétkorrekció
        - RiskEngine visszacsatolás befogadása
        - Bankroll védelem (stop-loss)
        - Napi max tét limit
        - Session védelem
    """

    def __init__(self, config=None):
        self.config = config or {}

        self.logger = get_logger()

        # Bankroll
        self.bankroll = self.config.get("bankroll", 1000)

        # Kelly faktor (0–1 között, konzervatív 0.1–0.25)
        self.kelly_factor = self.config.get("kelly_factor", 0.25)

        # Napi max tét (bankroll %-ban)
        self.daily_limit = self.config.get("daily_limit", 0.15)

        # Stop-loss €/nap
        self.stop_loss = self.config.get("stop_loss", 0.10) * self.bankroll

        # Session változók
        self.daily_used_pct = 0.0
        self.session_profit = 0.0

        self.logger.info(
            f"[BankrollEngine] Inicializálva — bankroll={self.bankroll}, "
            f"kelly_factor={self.kelly_factor}, stop_loss={self.stop_loss}"
        )

    # ----------------------------------------------------------------------
    # KELLY FORMULA (value-based stake sizing)
    # ----------------------------------------------------------------------
    def _kelly(self, prob, odds):
        """
        Kelly formula: f = (bp - q) / b
        ahol:
            b = odds - 1
            p = prob
            q = 1 - p
        """

        b = odds - 1
        p = prob
        q = 1 - p

        try:
            f = (b * p - q) / b
        except ZeroDivisionError:
            return 0.0

        return max(0.0, min(f, 1.0))

    # ----------------------------------------------------------------------
    # STAKE SIZE CALCULATION
    # ----------------------------------------------------------------------
    def compute_stake(self, prob, odds, value_score=0.0, risk_adjust=1.0):
        """
        Bemenet:
            prob        → becsült nyerési esély (0–1)
            odds        → élő/előzetes odds
            value_score → értékalapú korrekció (-1 – +1)
            risk_adjust → RiskEngine skálázó faktor (0–1)

        Visszatér:
            ajánlott tét EUR-ban
        """

        bankroll = self.bankroll

        # 0) Bankroll check
        if bankroll <= 0:
            self.logger.warning("[Bankroll] Nincs bankroll → tét=0")
            return 0.0

        # 1) Kelly alap stake%
        k = self._kelly(prob, odds)

        # Kelly scaling (konzervatív mód)
        k *= self.kelly_factor

        # 2) Value correction (min: -30%, max: +30%)
        k *= (1 + np.clip(value_score, -0.3, 0.3))

        # 3) RiskEngine adjustment
        k *= np.clip(risk_adjust, 0.2, 1.0)

        # 4) Stop-loss protection
        if self.session_profit <= -self.stop_loss:
            self.logger.warning("[Bankroll] Stop-loss aktiválva → tét=0")
            return 0.0

        # 5) Napi limit betartása
        if self.daily_used_pct + k > self.daily_limit:
            self.logger.warning("[Bankroll] Napi limit elérve → tét=0")
            return 0.0

        # 6) Stake kiszámítása
        stake_eur = round(bankroll * k, 2)

        # frissítjük napi használatot
        self.daily_used_pct += k

        self.logger.info(
            f"[Bankroll] stake={stake_eur}€  (k={round(k,4)}, prob={prob}, odds={odds})"
        )

        return stake_eur

    # ----------------------------------------------------------------------
    # PROFIT UPDATE
    # ----------------------------------------------------------------------
    def update_profit(self, profit):
        """
        profit: +pozitív / -negatív összeg
        """
        self.session_profit += profit
        self.bankroll += profit

        self.logger.info(
            f"[Bankroll] Profit frissítve: session_profit={self.session_profit}, "
            f"új bankroll={self.bankroll}"
        )
