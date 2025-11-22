# backend/engine/weather_engine.py

import numpy as np
from backend.utils.logger import get_logger

class WeatherEngine:
    """
    WEATHER ENGINE – PRO EDITION
    -----------------------------
    Feladata:
        • időjárási tényezők hatásának modellezése
        • hőmérséklet (heat drop / cold resistance)
        • szél (wind impact) – lövések, beadások nehézsége
        • eső/hó (slippery pitch) – tempó csökkenés, variancia nő
        • pitch_quality – pálya állapota
        • output → win probability módosítás weather alapján
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # scaling paraméterek
        self.temp_scaling = config.get("weather", {}).get("temp_scaling", 0.08)
        self.wind_scaling = config.get("weather", {}).get("wind_scaling", 0.12)
        self.rain_scaling = config.get("weather", {}).get("rain_scaling", 0.15)
        self.pitch_scaling = config.get("weather", {}).get("pitch_scaling", 0.10)

        # fallback
        self.fallback_prob = 0.52
        self.min_conf = config.get("weather", {}).get("min_confidence", 0.56)

    # ----------------------------------------------------------------------
    # PUBLIC: fő predikció
    # ----------------------------------------------------------------------
    def predict(self, match_data):
        outputs = {}

        for match_id, data in match_data.items():
            try:
                prob = self._weather_core(data)
            except Exception as e:
                self.logger.error(f"[Weather] Hiba → fallback: {e}")
                prob = self.fallback_prob

            prob = self._normalize(prob)
            conf = self._confidence(prob, data)
            risk = self._risk(prob, conf)

            outputs[match_id] = {
                "probability": round(prob, 4),
                "confidence": round(conf, 3),
                "risk": round(risk, 3),
                "meta": {
                    "temp_scaling": self.temp_scaling,
                    "wind_scaling": self.wind_scaling,
                    "rain_scaling": self.rain_scaling,
                    "pitch_scaling": self.pitch_scaling
                },
                "source": "Weather"
            }

        return outputs

    # ----------------------------------------------------------------------
    # WEATHER MAG
    # ----------------------------------------------------------------------
    def _weather_core(self, data):
        """
        Várt input:
            • temp_celsius       (°C)
            • wind_speed         (km/h)
            • precipitation      (0 = no, 1 = rain, 2 = heavy rain, 3 = snow)
            • pitch_quality      (0–1)
            • home_weather_bias  (0–1)
            • away_weather_bias  (0–1)
        """

        temp = data.get("temp_celsius", 15.0)      # °C
        wind = data.get("wind_speed", 5.0)         # km/h
        prec = data.get("precipitation", 0)        # 0–3
        pitch = data.get("pitch_quality", 0.8)     # 0–1

        home_bias = data.get("home_weather_bias", 0.5)
        away_bias = data.get("away_weather_bias", 0.5)

        # HOME előny/hátrány
        bias_diff = (home_bias - away_bias)

        # TEMPERATURE EFFECT
        # 30°C felett vagy 5°C alatt csökken a performance
        if temp >= 30:
            temp_effect = -abs(temp - 25) * self.temp_scaling
        elif temp <= 5:
            temp_effect = -abs(temp - 10) * self.temp_scaling
        else:
            temp_effect = 0

        # WIND EFFECT
        wind_effect = -(wind / 40.0) * self.wind_scaling

        # PRECIPITATION EFFECT
        rain_effect = -prec * self.rain_scaling   # 0–3 skálán

        # PITCH QUALITY EFFECT
        pitch_effect = (pitch - 0.8) * self.pitch_scaling

        # Combined weather shift
        weather_shift = (
            temp_effect +
            wind_effect +
            rain_effect +
            pitch_effect +
            (bias_diff * 0.10)
        )

        prob = 0.5 + weather_shift
        return float(prob)

    # ----------------------------------------------------------------------
    # NORMALIZÁLÁS
    # ----------------------------------------------------------------------
    def _normalize(self, p):
        return float(max(0.01, min(0.99, p)))

    # ----------------------------------------------------------------------
    # CONFIDENCE
    # ----------------------------------------------------------------------
    def _confidence(self, prob, data):
        weather_quality = data.get("weather_data_quality", 0.75)
        stability = 1 - abs(prob - 0.5)

        conf = weather_quality * 0.6 + stability * 0.4
        return float(max(self.min_conf, min(1.0, conf)))

    # ----------------------------------------------------------------------
    # RISK
    # ----------------------------------------------------------------------
    def _risk(self, prob, conf):
        return float(min(1.0, max(0.0, (1 - prob) * 0.5 + (1 - conf) * 0.5)))
