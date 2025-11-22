# backend/core/daily_training_workflow.py

import datetime
import sqlite3
import traceback
from backend.core.label_generator import LabelGenerator
from backend.core.training_pipeline import TrainingPipeline
from backend.engine.deep_value.train_value_model import DeepValueTrainer
from backend.utils.logger import get_logger

class DailyTrainingWorkflow:
    """
    Napi AI tanulási workflow:
        1. Eredmények betöltése (scraper vagy API)
        2. Label generálás
        3. Training sample mentése
        4. DeepValue modell tanítása (train_value_model)
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.logger = get_logger()

        self.db_path = "backend/data/db/training.db"
        self.label_gen = LabelGenerator(config)
        self.pipeline = TrainingPipeline()
        self.trainer = DeepValueTrainer(config)

    # -----------------------------------------------------------
    # 0) meccseredmények lekérése — placeholder
    # -----------------------------------------------------------
    def fetch_results(self):
        """
        Itt kell lehívni az API/SCAPER eredményeket.
        Most csak stub.
        Formátum:
            [
                {
                    "match_id": "1234",
                    "result": 1/0,
                    "profit": 0.85,
                    "final_ev": 0.42,
                    "predicted_prob": 0.61
                },
                ...
            ]
        """
        self.logger.warning("DailyWorkflow: fetch_results() STUB módban fut! Implementáld később.")
        return []

    # -----------------------------------------------------------
    # FŐ FÜGGVÉNY
    # -----------------------------------------------------------
    def run_daily_training(self):
        self.logger.info("=== DailyTrainingWorkflow: START ===")

        try:
            # 1) lekérjük az adott napi eredményeket
            results = self.fetch_results()
            if not results:
                self.logger.warning("DailyTrainingWorkflow: nincs eredmény, skip.")
                return False

            self.logger.info(f"DailyTrainingWorkflow: {len(results)} eredmény feldolgozása...")

            # 2) DB kapcsolódás
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()

            for r in results:
                match_id = r["match_id"]
                base_result = r["result"]
                profit = r["profit"]
                final_ev = r["final_ev"]
                predicted_prob = r.get("predicted_prob", 0.5)

                # 3) feature vector betöltése engine_features táblából
                cur.execute("SELECT features FROM engine_features WHERE match_id = ?", (match_id,))
                row = cur.fetchone()

                if not row:
                    self.logger.error(f"Missing features for match {match_id}, skip.")
                    continue

                fv = row[0]
                # BLOB → numpy
                import numpy as np
                fv = np.frombuffer(fv, dtype=np.float32)

                # 4) label generálása
                label = self.label_gen.generate_label(
                    result=base_result,
                    ev=final_ev,
                    profit=profit,
                    predicted_prob=predicted_prob
                )

                # 5) training sample mentése
                self.pipeline.save_training_sample(
                    match_id=match_id,
                    feature_vec=fv,
                    label_value=label,
                    profit=profit,
                    final_ev=final_ev
                )

            conn.close()

            # 6) DeepValue modell tanítása
            self.logger.info("DailyTrainingWorkflow: retraining DeepValue model...")
            self.trainer.train_model()

            self.logger.info("=== DailyTrainingWorkflow: DONE ===")
            return True

        except Exception as e:
            self.logger.error("ERROR in DailyTrainingWorkflow:")
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
            return False
