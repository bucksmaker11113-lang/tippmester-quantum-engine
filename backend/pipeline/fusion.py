# backend/pipeline/fusion.py

from backend.core.fusion_engine import FusionEngine
from backend.core.bayesian_updater import BayesianUpdater
from backend.core.bias_engine import BiasEngine
from backend.core.value_analyzer import ValueAnalyzer
from backend.engine.quantum_synth_engine import QuantumSynthEngine
from backend.utils.logger import get_logger

class FusionCore:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger()

        self.fusion = FusionEngine(config)
        self.bayes = BayesianUpdater(config)
        self.bias = BiasEngine(config)
        self.value = ValueAnalyzer(config)
        self.qsynth = QuantumSynthEngine(config)

    # ----------------------------------------------------------
    # MODEL OUTPUT FÚZIÓ
    # ----------------------------------------------------------
    def combine(self, model_outputs):
        self.logger.info("FusionCore: modellek összehangolása...")

        results = {}

        # 1) Quantum Synth Engine (első aggregáció)
        try:
            quantum_combined = self.qsynth.fuse(model_outputs)
        except Exception as e:
            self.logger.error(f"QuantumSynth hiba: {e}")
            quantum_combined = {}

        # 2) Fusion Engine (második aggregáció)
        try:
            fused = self.fusion.combine(model_outputs)
        except Exception as e:
            self.logger.error(f"FusionEngine hiba: {e}")
            fused = {}

        # 3) Bayesian frissítés
        try:
            bayes = self.bayes.update(fused)
        except Exception as e:
            self.logger.error(f"Bayesian hiba: {e}")
            bayes = fused

        # 4) Bias korrekció
        try:
            bias_corrected = self.bias.apply(bayes)
        except Exception as e:
            self.logger.error(f"Bias korrekció hiba: {e}")
            bias_corrected = bayes

        # 5) Value ellenőrzés
        try:
            final = self.value.evaluate(bias_corrected)
        except Exception as e:
            self.logger.error(f"ValueAnalyzer hiba: {e}")
            final = bias_corrected

        return final
