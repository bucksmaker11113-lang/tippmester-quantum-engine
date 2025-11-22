# backend/pipeline/model_runner.py

from backend.engine.montecarlo_v3_engine import MonteCarloV3
from backend.engine.lstm_rnn_engine import LSTMEngine
from backend.engine.gnn_engine import GNNEngine
from backend.engine.poisson_engine import PoissonEngine
from backend.engine.rl_engine import ReinforcementEngine
from backend.engine.gameflow_engine import GameflowEngine
from backend.engine.value_engine import ValueEngine
from backend.utils.logger import get_logger

class ModelRunner:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        # Modell instance-ek
        self.mc3 = MonteCarloV3(config)
        self.lstm = LSTMEngine(config)
        self.gnn = GNNEngine(config)
        self.poisson = PoissonEngine(config)
        self.rl = ReinforcementEngine(config)
        self.gameflow = GameflowEngine(config)
        self.value = ValueEngine(config)

    # ----------------------------------------------------------
    # MODEL FUTTATÁS
    # ----------------------------------------------------------
    def run_all(self, preprocessed):
        self.logger.info("ModelRunner: modellek futtatása...")

        results = {}

        try:
            results["mc3"] = self.mc3.predict(preprocessed)
        except Exception as e:
            self.logger.error(f"MonteCarlo3 hiba: {e}")

        try:
            results["lstm"] = self.lstm.predict(preprocessed)
        except Exception as e:
            self.logger.error(f"LSTM hiba: {e}")

        try:
            results["gnn"] = self.gnn.predict(preprocessed)
        except Exception as e:
            self.logger.error(f"GNN hiba: {e}")

        try:
            results["poisson"] = self.poisson.predict(preprocessed)
        except Exception as e:
            self.logger.error(f"Poisson hiba: {e}")

        try:
            results["rl"] = self.rl.predict(preprocessed)
        except Exception as e:
            self.logger.error(f"RL hiba: {e}")

        try:
            results["gameflow"] = self.gameflow.predict(preprocessed)
        except Exception as e:
            self.logger.error(f"GameFlow hiba: {e}")

        try:
            results["value_model"] = self.value.predict(preprocessed)
        except Exception as e:
            self.logger.error(f"Value hiba: {e}")

        return results
