# backend/engine/lstm_rnn_engine.py

import numpy as np
from backend.utils.logger import get_logger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam

class LSTMEngine:
    """
    BiLSTM alapú idősoros predikciós motor
    - odds trendekre
    - csapatforma idősorra
    - preprocessed temporal inputokból dolgozik
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # Model paraméterek
        self.timesteps = config.get("lstm_timesteps", 10)
        self.features = config.get("lstm_features", 4)
        self.lr = config.get("lstm_learning_rate", 0.001)

        # Modell inicializálás
        self.model = self._build_model()

    # --------------------------------------------------------------
    # LSTM modell építése
    # --------------------------------------------------------------
    def _build_model(self):
        model = Sequential()
        model.add(
            Bidirectional(
                LSTM(
                    64,
                    return_sequences=False
                ),
                input_shape=(self.timesteps, self.features)
            )
        )
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss="binary_crossentropy"
        )

        return model

    # --------------------------------------------------------------
    # Prediction (inference)
    # --------------------------------------------------------------
    def predict(self, preprocessed):
        """
        Input: temporal data csúszóablakokban
        Output: probability (0–1)
        """

        self.logger.info("LSTMEngine: BiLSTM előrejelzés fut...")

        results = {}
        temporal = preprocessed.get("temporal", {})

        for match_id, series in temporal.items():

            # Előkészítés
            x = np.array(series, dtype=np.float32)
            if x.ndim == 2:
                x = np.expand_dims(x, axis=0)

            # LSTM előrejelzés
            pred = self.model.predict(x, verbose=0)
            p = float(pred[0][0])

            results[match_id] = {
                "probability": round(p, 4),
                "trend_score": round((p - 0.5) * 2, 4),
                "source": "BiLSTM"
            }

        return results

    # --------------------------------------------------------------
    # Optional training (offline RL-barát)
    # --------------------------------------------------------------
    def train(self, X, y, epochs=5, batch=16):
        self.logger.info("LSTMEngine: tanítás indul...")
        self.model.fit(X, y, epochs=epochs, batch_size=batch, verbose=1)
