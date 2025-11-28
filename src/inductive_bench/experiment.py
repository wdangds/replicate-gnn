# inductive_bench/experiment.py

from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from .data_utils import stratified_split_nodes, make_subgraph
from .models import GCN, GraphSAGE
from .graph_features import compute_graph_features

@dataclass
class InductiveExperiment:
    node_df: pd.DataFrame
    edge_df: pd.DataFrame
    label_col: str = "subject"
    bow_prefix: str = "w_"
    random_state: int = 42
    device: Optional[torch.device] = None
    add_graph_features: bool = True    

    def __post_init__(self):
        # copy to avoid mutating original df
        self.node_df = self.node_df.copy()

        # encode labels
        self.le = LabelEncoder()
        self.node_df["label_id"] = self.le.fit_transform(
            self.node_df[self.label_col].values
        )

        # BoW feature columns
        self.bow_feature_cols = [
            c for c in self.node_df.columns if c.startswith(self.bow_prefix)
        ]
        # Optionally compute graph features
        self.graph_feature_cols: List[str] = []
        if self.add_graph_features:
            self.node_df, self.graph_feature_cols = compute_graph_features(
                self.node_df, self.edge_df
            )

        # Combined feature set for "BoW + graph" baselines
        self.bow_plus_graph_cols = self.bow_feature_cols + self.graph_feature_cols

        # device
        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

    # -------------------------
    # Baseline models
    # -------------------------

    def run_baselines(
        self,
        train_nodes,
        val_nodes,
        test_nodes,
    ) -> List[Dict]:
        """
        Logistic Regression, Random Forest, MLP.

        For each model we train:
          - *_BoW      : using only bag-of-words features
          - *_BoWGraph: using bag-of-words + graph features
        """
        results = []

        # --- BoW-only features ---
        X_train_bow = self.node_df.loc[train_nodes, self.bow_feature_cols].values
        y_train = self.node_df.loc[train_nodes, "label_id"].values

        X_val_bow = self.node_df.loc[val_nodes, self.bow_feature_cols].values
        y_val = self.node_df.loc[val_nodes, "label_id"].values

        X_test_bow = self.node_df.loc[test_nodes, self.bow_feature_cols].values
        y_test = self.node_df.loc[test_nodes, "label_id"].values

        scaler_bow = StandardScaler()
        X_train_bow_sc = scaler_bow.fit_transform(X_train_bow)
        X_val_bow_sc = scaler_bow.transform(X_val_bow)
        X_test_bow_sc = scaler_bow.transform(X_test_bow)

        # --- BoW + graph features (if available) ---
        if len(self.graph_feature_cols) > 0:
            X_train_full = self.node_df.loc[
                train_nodes, self.bow_plus_graph_cols
            ].values
            X_val_full = self.node_df.loc[
                val_nodes, self.bow_plus_graph_cols
            ].values
            X_test_full = self.node_df.loc[
                test_nodes, self.bow_plus_graph_cols
            ].values

            scaler_full = StandardScaler()
            X_train_full_sc = scaler_full.fit_transform(X_train_full)
            X_val_full_sc = scaler_full.transform(X_val_full)
            X_test_full_sc = scaler_full.transform(X_test_full)
        else:
            X_train_full = X_val_full = X_test_full = None
            X_train_full_sc = X_val_full_sc = X_test_full_sc = None

        # Helper to add result dicts
        def add_result(name: str, val_acc: float, test_acc: float):
            results.append(
                {"model": name, "val_acc": val_acc, "test_acc": test_acc}
            )

        # ----- Logistic Regression -----
        logreg = LogisticRegression(
            C=0.1,
            max_iter=1000,
            n_jobs=-1,
            random_state=self.random_state,
        )
        logreg.fit(X_train_bow_sc, y_train)
        val_acc = accuracy_score(y_val, logreg.predict(X_val_bow_sc))
        test_acc = accuracy_score(y_test, logreg.predict(X_test_bow_sc))
        add_result("LogReg_BoW", val_acc, test_acc)

        if X_train_full_sc is not None:
            logreg_full = LogisticRegression(
                C=0.1,
                max_iter=1000,
                n_jobs=-1,
                random_state=self.random_state,
            )
            logreg_full.fit(X_train_full_sc, y_train)
            val_acc = accuracy_score(y_val, logreg_full.predict(X_val_full_sc))
            test_acc = accuracy_score(y_test, logreg_full.predict(X_test_full_sc))
            add_result("LogReg_BoWGraph", val_acc, test_acc)

        # ----- Random Forest -----
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=self.random_state,
        )
        rf.fit(X_train_bow, y_train)
        val_acc = accuracy_score(y_val, rf.predict(X_val_bow))
        test_acc = accuracy_score(y_test, rf.predict(X_test_bow))
        add_result("RandomForest_BoW", val_acc, test_acc)

        if X_train_full is not None:
            rf_full = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                random_state=self.random_state,
            )
            rf_full.fit(X_train_full, y_train)
            val_acc = accuracy_score(y_val, rf_full.predict(X_val_full))
            test_acc = accuracy_score(y_test, rf_full.predict(X_test_full))
            add_result("RandomForest_BoWGraph", val_acc, test_acc)

        # ----- MLP -----
        mlp = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=200,
            random_state=self.random_state,
        )
        mlp.fit(X_train_bow_sc, y_train)
        val_acc = accuracy_score(y_val, mlp.predict(X_val_bow_sc))
        test_acc = accuracy_score(y_test, mlp.predict(X_test_bow_sc))
        add_result("MLP_BoW", val_acc, test_acc)

        if X_train_full_sc is not None:
            mlp_full = MLPClassifier(
                hidden_layer_sizes=(128,),
                activation="relu",
                alpha=1e-4,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=200,
                random_state=self.random_state,
            )
            mlp_full.fit(X_train_full_sc, y_train)
            val_acc = accuracy_score(y_val, mlp_full.predict(X_val_full_sc))
            test_acc = accuracy_score(y_test, mlp_full.predict(X_test_full_sc))
            add_result("MLP_BoWGraph", val_acc, test_acc)

        return results

    # -------------------------
    # GNN helpers
    # -------------------------

    @staticmethod
    def _gnn_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
        preds = logits.argmax(dim=-1)
        return (preds == y).float().mean().item()

    def run_gnn(
        self,
        model_name: str,
        train_data,
        val_data,
        test_data,
        hidden: int = 64,
        dropout: float = 0.6,
        aggr: str = "mean",
        lr: float = 0.01,
        weight_decay: float = 5e-3,
        max_epochs: int = 400,
        patience: int = 40,
        return_model: bool=False
    ) -> Dict:
        train_data = train_data.to(self.device)
        val_data = val_data.to(self.device)
        test_data = test_data.to(self.device)

        if model_name == "GraphSAGE":
            model = GraphSAGE(
                in_channels=train_data.num_features,
                hidden_channels=hidden,
                out_channels=len(self.le.classes_),
                dropout=dropout,
                aggr=aggr,
            ).to(self.device)
        else:
            model = GCN(
                in_channels=train_data.num_features,
                hidden_channels=hidden,
                out_channels=len(self.le.classes_),
                dropout=dropout,
            ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        best_val, best_test = 0.0, 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(train_data.x, train_data.edge_index)
            loss = criterion(out, train_data.y)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_out = model(val_data.x, val_data.edge_index)
                test_out = model(test_data.x, test_data.edge_index)

                train_acc = self._gnn_accuracy(out, train_data.y)
                val_acc = self._gnn_accuracy(val_out, val_data.y)
                test_acc = self._gnn_accuracy(test_out, test_data.y)

            if val_acc > best_val + 1e-4:
                best_val, best_test = val_acc, test_acc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 50 == 0 or epoch == 1:
                print(
                    f"{model_name} Epoch {epoch:3d} | "
                    f"Loss {loss.item():.4f} | Train {train_acc:.3f} | "
                    f"Val {val_acc:.3f} | Test {test_acc:.3f}"
                )

            if patience_counter >= patience:
                print(f"{model_name}: early stopping at epoch {epoch}")
                break

        if best_state is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        result = {"model": model_name, "val_acc": best_val, "test_acc": best_test}
        if return_model:
            result["trained_model"] = model
        return result
    # -------------------------
    # High-level APIs (same as before)
    # -------------------------

    def run_single_split(
        self,
        train_frac: float,
        include_gnns: bool = True,
    ) -> pd.DataFrame:
        print(f"\n=== Running single split with train_frac = {train_frac:.2f} ===")

        train_nodes, val_nodes, test_nodes = stratified_split_nodes(
            self.node_df,
            train_frac=train_frac,
            label_col="label_id",
            random_state=self.random_state,
        )

        results = []

        # baselines (BoW + BoWGraph)
        base_results = self.run_baselines(train_nodes, val_nodes, test_nodes)
        for r in base_results:
            r["train_frac"] = train_frac
            results.append(r)

        if include_gnns:
            # NOTE: GNNs only see BoW features
            train_data = make_subgraph(
                train_nodes, self.node_df, self.edge_df, self.bow_feature_cols
            )
            val_data = make_subgraph(
                val_nodes, self.node_df, self.edge_df, self.bow_feature_cols
            )
            test_data = make_subgraph(
                test_nodes, self.node_df, self.edge_df, self.bow_feature_cols
            )

            gcn_res = self.run_gnn(
                "GCN", train_data, val_data, test_data, hidden=64, dropout=0.6
            )
            gcn_res["train_frac"] = train_frac
            results.append(gcn_res)

            sage_res = self.run_gnn(
                "GraphSAGE",
                train_data,
                val_data,
                test_data,
                hidden=64,
                dropout=0.6,
                aggr="mean",
            )
            sage_res["train_frac"] = train_frac
            results.append(sage_res)

        return pd.DataFrame(results)

    def run_grid(
        self,
        train_fracs: List[float],
        include_gnns: bool = True,
    ) -> pd.DataFrame:
        all_results = []
        for frac in train_fracs:
            df_frac = self.run_single_split(frac, include_gnns=include_gnns)
            all_results.append(df_frac)
        return pd.concat(all_results, ignore_index=True)

    # -------------------------
    # Plotting helpers
    # -------------------------

    @staticmethod
    def plot_overall(
        results_df: pd.DataFrame,
        metric: str = "test_acc",
        title: str = "Inductive performance vs train fraction",
    ):
        plt.figure(figsize=(8, 5))
        for model_name in results_df["model"].unique():
            sub = results_df[results_df["model"] == model_name].copy()
            sub = sub.sort_values("train_frac")
            plt.plot(
                sub["train_frac"],
                sub[metric],
                marker="o",
                label=model_name,
            )

        plt.xlabel("Train fraction")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_split(
        results_df: pd.DataFrame,
        train_frac: float,
        metric: str = "test_acc",
        title: Optional[str] = None,
    ):
        sub = results_df[results_df["train_frac"] == train_frac].copy()
        sub = sub.sort_values("model")

        plt.figure(figsize=(6, 4))
        plt.bar(sub["model"], sub[metric])
        plt.xticks(rotation=45)
        plt.ylabel(metric.replace("_", " ").title())
        if title is None:
            title = f"Performance at train_frac = {train_frac:.2f}"
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices_rf_sage(
        self,
        train_frac: float,
        use_graph_features: bool = True,
        normalize: Optional[str] = None,
    ):
        """
        Train a RandomForest baseline and GraphSAGE on the given train_frac,
        then plot test-set confusion matrices for both models.

        Parameters
        ----------
        train_frac : float
            Fraction of nodes used for training (val/test split remaining equally).
        use_graph_features : bool
            If True and graph features exist, use BoW + graph features for RF;
            otherwise use BoW only.
        normalize : {'true', 'pred', 'all', None}
            Passed to sklearn.metrics.confusion_matrix.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # --- 1) Stratified split ---
        train_nodes, val_nodes, test_nodes = stratified_split_nodes(
            self.node_df,
            train_frac=train_frac,
            label_col="label_id",
            random_state=self.random_state,
        )

        # -----------------------------
        # RandomForest confusion matrix
        # -----------------------------
        if use_graph_features and len(self.graph_feature_cols) > 0:
            cols_rf = self.bow_plus_graph_cols
            rf_label = "RandomForest_BoWGraph"
        else:
            cols_rf = self.bow_feature_cols
            rf_label = "RandomForest_BoW"

        X_train_rf = self.node_df.loc[train_nodes, cols_rf].values
        y_train_rf = self.node_df.loc[train_nodes, "label_id"].values

        X_test_rf = self.node_df.loc[test_nodes, cols_rf].values
        y_test_rf = self.node_df.loc[test_nodes, "label_id"].values

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=self.random_state,
        )
        rf.fit(X_train_rf, y_train_rf)
        y_pred_rf = rf.predict(X_test_rf)

        cm_rf = confusion_matrix(
            y_test_rf,
            y_pred_rf,
            labels=np.arange(len(self.le.classes_)),
            normalize=normalize,
        )

        disp_rf = ConfusionMatrixDisplay(
            confusion_matrix=cm_rf,
            display_labels=self.le.classes_,
        )
        plt.figure(figsize=(6, 5))
        disp_rf.plot(include_values=True, cmap="Blues", xticks_rotation=45, colorbar=False)
        plt.title(f"{rf_label} (test confusion matrix, train_frac={train_frac:.2f})")
        plt.tight_layout()
        plt.show()

        # -----------------------------
        # GraphSAGE confusion matrix
        # -----------------------------
        # GNNs always use BoW features in this package
        train_data = make_subgraph(
            train_nodes, self.node_df, self.edge_df, self.bow_feature_cols
        )
        val_data = make_subgraph(
            val_nodes, self.node_df, self.edge_df, self.bow_feature_cols
        )
        test_data = make_subgraph(
            test_nodes, self.node_df, self.edge_df, self.bow_feature_cols
        )

        sage_res = self.run_gnn(
            "GraphSAGE",
            train_data,
            val_data,
            test_data,
            hidden=64,
            dropout=0.6,
            aggr="mean",
            return_model=True,  # <--- get the trained model back
        )
        sage_model = sage_res["trained_model"]
        sage_model.eval()

        with torch.no_grad():
            logits = sage_model(
                test_data.x.to(self.device),
                test_data.edge_index.to(self.device),
            )
            y_test_gs = test_data.y.cpu().numpy()
            y_pred_gs = logits.argmax(dim=-1).cpu().numpy()

        cm_gs = confusion_matrix(
            y_test_gs,
            y_pred_gs,
            labels=np.arange(len(self.le.classes_)),
            normalize=normalize,
        )

        disp_gs = ConfusionMatrixDisplay(
            confusion_matrix=cm_gs,
            display_labels=self.le.classes_,
        )
        plt.figure(figsize=(6, 5))
        disp_gs.plot(include_values=True, cmap="Greens", xticks_rotation=45, colorbar=False)
        plt.title(f"GraphSAGE (test confusion matrix, train_frac={train_frac:.2f})")
        plt.tight_layout()
        plt.show()
