# backend/core/label_generator.py

import numpy as np

class LabelGenerator:
    """
    Value Label Generator
    ----------------------
    A DeepValue modell tanulásához 0–1 közötti címkét generál:

        1.  A meccs bejött-e? (win/loss)
        2.  Mekkora volt az EV? (expected value)
        3.  Mekkora volt a profit? (real)
        4.  Mennyire volt erős a value jel?

    A címke így áll össze:
        base_label  = 1 ha nyert, 0 ha vesztett
        ev_boost    = EV alapján + skálázás
        profit_adj  = profit alapján + korrekció

    Eredmény: 0–1 közé normalizálva
    """

    def __init__(self, config=None):
        self.config = config or {}

        # súlyok
        self.w_base = self.config.get("label_weights", {}).get("win_loss", 0.6)
        self.w_ev   = self.config.get("label_weights", {}).get("ev_weight", 0.3)
        self.w_profit = self.config.get("label_weights", {}).get("profit_weight", 0.1)

    # ------------------------------------------------------------
    # FŐ FÜGGVÉNY
    # ------------------------------------------------------------
    def generate_label(self, result: int, ev: float, profit: float, predicted_prob: float):
        """
        Paraméterek:
            result:         1 ha bejött, 0 ha vesztes (eredmény)
            ev:             expected value (predikció után)
            profit:         nettó profit (tét ± nyereség)
            predicted_prob: model által jelzett valószínűség

        Visszatér:
            float (0–1 közötti címke)
        """

        # --- 1) alap nyereség/veszteség címke ---
        base_label = 1.0 if result == 1 else 0.0

        # --- 2) EV alapú boost ---
        # poz EV -> jutalmazás
        # negatív EV -> büntetés
        ev_norm = np.tanh(ev)  # -1..1 → normalizálás
        ev_label = (ev_norm + 1) / 2   # 0..1 skála

        # --- 3) profit alapú korrekció ---
        # profit: tipmix netto eredmény
        profit_norm = np.tanh(profit)
        profit_label = (profit_norm + 1) / 2

        # --- 4) súlyozott összeg ---
        label = (
            base_label * self.w_base +
            ev_label * self.w_ev +
            profit_label * self.w_profit
        )

        # --- 5) clamp ---
        label = float(max(0.0, min(1.0, label)))

        return label
