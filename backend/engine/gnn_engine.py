# backend/engine/gnn_engine.py

import numpy as np
from backend.utils.logger import get_logger
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class GNNEdgeLayer(tf.keras.layers.Layer):
    """
    Egyszerű GCN réteg implementáció
    """
    def __init__(self, units):
        super(GNNEdgeLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform",
                                 trainable=True)

    def call(self, inputs, adj):
        h = tf.matmul(inputs, self.w)
        return tf.matmul(adj, h)  # A * H * W

class GNNEngine:
    """
    Graph Neural Network Engine
    - formagráf
    - odds-gráf
    - team-strength gráf
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()
        self.learning_rate = config.get("gnn_lr", 0.001)

        self.model = None
        self._build_model()

    # ---------------------------------------------------------
    # GNN MODEL ÉPÍTÉSE
    # ---------------------------------------------------------
    def _build_model(self):
        node_in = Input(shape=(8,), name="nodes")     # node feature: forma, odds, power, trend
        adj_in = Input(shape=(None,), name="adjacency")  # adjacency row

        gnn1 = GNNEdgeLayer(16)(node_in, adj_in)
        gnn1 = tf.nn.relu(gnn1)

        gnn2 = GNNEdgeLayer(8)(gnn1, adj_in)
        gnn2 = tf.nn.relu(gnn2)

        out = Dense(1, activation="sigmoid")(gnn2)

        self.model = Model(inputs=[node_in, adj_in], outputs=out)
        self.model.compile(optimizer=Adam(self.learning_rate), loss="binary_crossentropy")

    # ---------------------------------------------------------
    # PREDIKCIÓ
    # ---------------------------------------------------------
    def predict(self, preprocessed):

        self.logger.info("GNNEngine: gráf-alapú predikció...")

        results = {}
        graph = preprocessed.get("graph", {})

        for match_id, gdata in graph.items():

            nodes = np.array(gdata["nodes"], dtype=np.float32)
            adj = np.array(gdata["adj"], dtype=np.float32)

            # GNN előrejelzés
            pred = self.model.predict([nodes, adj], verbose=0)
            p = float(np.mean(pred))

            results[match_id] = {
                "probability": round(p, 4),
                "graph_strength": round((p - 0.5) * 2, 4),
                "source": "GNNEngine"
            }

        return results
