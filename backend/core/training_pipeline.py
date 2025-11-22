# backend/core/training_pipeline.py

import sqlite3
import os
import numpy as np
from datetime import datetime

class TrainingPipeline:
    """
    Ment minden train adagot a SQLite-ba:
        - engine feature vector
        - label (value came true?)
        - profit
        - final_ev
    """

    def __init__(self):
        self.db_path = "backend/data/db/training.db"
        os.makedirs("backend/data/db", exist_ok=True)

    def save_training_sample(self, match_id, feature_vec, label_value, profit, final_ev):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # 1) engine_features táblába
        cur.execute("""
            INSERT OR REPLACE INTO engine_features (match_id, features, created_at)
            VALUES (?, ?, ?)
        """, (
            match_id,
            feature_vec.tobytes(),
            datetime.utcnow().isoformat()
        ))

        # 2) training_labels táblába
        cur.execute("""
            INSERT OR REPLACE INTO training_labels
            (match_id, label_value, profit, final_ev, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            match_id,
            float(label_value),
            float(profit),
            float(final_ev),
            datetime.utcnow().isoformat()
        ))

        conn.commit()
        conn.close()
