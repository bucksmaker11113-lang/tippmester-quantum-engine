# backend/engine/deep_value/train_value_model.py

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from backend.engine.deep_value_engine import DeepValueNet
from backend.utils.logger import get_logger


class ValueDataset(Dataset):
    """
    Dataset:
        X = feature_vectors (128 dim)
        y = real value (0–1)
    """
    def __init__(self, data):
        self.X = []
        self.y = []

        for item in data:
            fv = np.array(item["features"], dtype=np.float32)
            if len(fv) == 128:
                self.X.append(fv)
                self.y.append(float(item["label"]))

        self.X = np.array(self.X)
        self.y = np.array(self.y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]).float(),
            torch.tensor(self.y[idx]).float()
        )


class DeepValueTrainer:
    """
    DEEP VALUE TRAINING PIPELINE
    ----------------------------
    Feladata:
        - betölti a training_data.json-t
        - deep value modellt tanítja
        - elmenti a model_weights.pth súlyokat
    """

    def __init__(self, config):
        self.logger = get_logger()

        self.data_path = "backend/data/training/training_data.json"
        self.model_path = "backend/data/models/model_weights.pth"

        self.input_dim = config.get("deep_value", {}).get("input_dim", 128)
        self.hidden_dim = config.get("deep_value", {}).get("hidden_dim", 256)
        self.lr = config.get("deep_value", {}).get("learning_rate", 0.001)
        self.epochs = config.get("deep_value", {}).get("epochs", 8)
        self.batch_size = config.get("deep_value", {}).get("batch_size", 32)

        self.device = "cpu"

        self.model = DeepValueNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        # ha van korábbi modell → betöltjük (continual learning)
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
                self.logger.info("[DeepValueTrain] Loaded previous model weights.")
            except:
                self.logger.error("[DeepValueTrain] Failed to load previous weights.")

        # loss + optimizer
        self.loss_fn = torch.nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    # TRAIN EXECUTOR
    # ------------------------------------------------------------------
    def train_model(self):
        # 1) adat betöltése
        if not os.path.exists(self.data_path):
            self.logger.error("[DeepValueTrain] NO TRAINING DATA FOUND!")
            return False

        with open(self.data_path, "r") as f:
            training_data = json.load(f)

        dataset = ValueDataset(training_data)
        if len(dataset) < 20:
            self.logger.warning("[DeepValueTrain] Not enough data for training.")
            return False

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 2) tanulás
        self.logger.info(f"[DeepValueTrain] Training samples: {len(dataset)}")

        for epoch in range(self.epochs):
            total_loss = 0
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device)

                pred = self.model(X)
                loss = self.loss_fn(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            self.logger.info(f"[DeepValueTrain] Epoch {epoch+1}/{self.epochs} — Loss: {avg_loss:.4f}")

        # 3) új súlyok mentése
        os.makedirs("backend/data/models", exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)

        self.logger.info("[DeepValueTrain] Training complete — model saved.")
        return True
