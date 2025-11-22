# backend/engine/ai_coach_explainer.py

class AICoachExplainer:
    """
    AI COACH EXPLAINER ENGINE
    -------------------------
    Minden tipphez emberi-nyelvű magyarázatot ad:
        - value indoklás
        - statisztikai háttér
        - sharp money és CLV magyarázat
        - sérülések, időjárás, gameflow hatások
        - prop piacok indoklása (handicap, totals, BTTS stb.)
        - final verdict: miért value a tipp?
    """

    def __init__(self, config=None):
        self.config = config or {}

    # -----------------------------------------------------
    # 1) ALAP VALUE MAGYARÁZAT
    # -----------------------------------------------------
    def _explain_value(self, tip):
        prob = tip.get("probability", 0)
        odds = tip.get("odds", 0)
        val = tip.get("value_score", 0)

        if val > 0.35:
            return f"A tippet jelentős value-nak találtuk (value score: {val}). " \
                   f"A modell szerint {prob*100:.1f}% az esély, miközben az odds {odds}."

        if val > 0.15:
            return f"A tipp mérsékelt value-t mutat (value score: {val}). " \
                   f"A várható esély {prob*100:.1f}% az {odds}-höz képest."

        return f"A value alacsonyabb, de még pozitív (value score: {val})."

    # -----------------------------------------------------
    # 2) SHARP MONEY MAGYARÁZAT
    # -----------------------------------------------------
    def _explain_sharp(self, tip):
        sharp = tip.get("sharp_money", 0)
        momentum = tip.get("momentum", 0)

        if sharp > 0.65:
            return "Erős sharp money aktivitás érzékelhető a piacon, ami támogatja a tipp irányát. "

        if sharp > 0.35:
            return "Mérsékelt sharp pénzmozgás látható, ami pozitív jel a tipp számára. "

        if momentum < -0.01:
            return "A piac enyhe lefelé mozgást mutat az oddsban, ami value megerősítés lehet. "

        return "Nem látható jelentős sharp money hatás. "

    # -----------------------------------------------------
    # 3) CLOSING LINE MAGYARÁZAT
    # -----------------------------------------------------
    def _explain_clv(self, tip):
        clv = tip.get("clv", 0)
        expected_closing = tip.get("expected_closing_line", None)
        odds = tip.get("odds", None)

        if clv > 0.05:
            return f"Várhatóan javulni fog a closing line ({expected_closing}), " \
                   f"ami erős value indikátor."

        if clv > 0:
            return f"A várható closing line kissé kedvezőbb ({expected_closing}), " \
                   f"ami pozitív jel."

        if expected_closing and expected_closing < odds:
            return "A closing line előrejelzés kedvezőtlenebb, óvatos megközelítés ajánlott. "

        return "A closing line előrejelzés semleges."

    # -----------------------------------------------------
    # 4) PROP PIACOK MAGYARÁZATA
    # -----------------------------------------------------
    def _explain_prop(self, tip):
        mtype = tip.get("market_category", "")
        if mtype == "totals":
            return "A gólvárakozások (xG) alapján a totals piac releváns és stabil value-t mutat. "
        if mtype == "handicap":
            return "A handicap piac a két csapat közti valódi erőviszonyokat jól tükrözi. "
        if mtype == "btts":
            return "A két csapat támadóstílusa támogatja a BTTS piacot. "
        if mtype == "cards":
            return "A felek agresszivitása és statisztikai adatai miatt a lapok piac értelmezhető. "
        if mtype == "player_props":
            return "A játékos statisztikák (xG, lövések, formák) alapján jó value van a player propban. "

        return ""

    # -----------------------------------------------------
    # 5) RISK ELEMZÉS MAGYARÁZATA
    # -----------------------------------------------------
    def _explain_risk(self, tip):
        risk = tip.get("risk", 0)

        if risk < 0.25:
            return "A tipp alacsony kockázatú kategóriába esik. "
        if risk < 0.45:
            return "A kockázati szint elfogadható, de érdemes óvatosságot tartani. "
        return "Magasabb kockázati szint, óvatos stake ajánlott. "

    # -----------------------------------------------------
    # 6) KOMPLEX FINAL EXPLANATION ÖSSZEFŰZVE
    # -----------------------------------------------------
    def generate_explanation(self, tip):
        parts = []

        parts.append(self._explain_value(tip))
        parts.append(self._explain_sharp(tip))
        parts.append(self._explain_clv(tip))

        if tip.get("market_category"):
            parts.append(self._explain_prop(tip))

        parts.append(self._explain_risk(tip))

        # Végső összefoglalás
        parts.append("Összességében a tipp értékesnek számít a modellek és piaci adatok alapján.")

        return " ".join(parts)
