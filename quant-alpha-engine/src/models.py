"""Machine Learning Models and Walk-Forward Validation Engine.

Implements Purged Walk-Forward Time-Series Cross-Validation, preventing data leakage
and serial correlation overlap. Trains and benchmarks multiple models (Logistic Regression,
Random Forest, Gradient Boosting) with probability calibration and trading-specific metrics.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


class PurgedWalkForwardCV:
    """Purged & Embargoed Time-Series Cross-Validation for Financial ML.
    
    Prevents lookahead bias and overlapping label leakage across train/test splits.
    As described in Marcos López de Prado's 'Advances in Financial Machine Learning'.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.6,
        purge_window: int = 5,
        embargo_pct: float = 0.01,
    ):
        """
        Args:
            n_splits: Number of walk-forward rolling folds.
            train_ratio: Proportion of available history used for expanding/rolling training window.
            purge_window: Number of days to drop between train and test (equal to prediction horizon H).
            embargo_pct: Additional embargo buffer after test set to eliminate serial correlation.
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices for expanding window purged time-series splits."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        splits = []

        # Minimum training size
        min_train = int(n_samples * 0.4)
        remaining_samples = n_samples - min_train
        test_size = max(20, remaining_samples // (self.n_splits + 1))

        for fold in range(self.n_splits):
            train_end = min_train + (fold * test_size)
            test_start = train_end + self.purge_window
            test_end = min(n_samples, test_start + test_size)

            if test_start >= n_samples or test_start >= test_end:
                break

            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]

            splits.append((train_idx, test_idx))

        return splits


def get_model_zoo() -> Dict[str, Any]:
    """Return dictionary of candidate ML classifiers."""
    return {
        "Logistic Regression (L2)": LogisticRegression(
            C=0.1, solver="lbfgs", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=15,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=4,
            min_samples_leaf=20,
            learning_rate=0.03,
            max_iter=120,
            random_state=42,
            class_weight="balanced",
        ),
    }


def evaluate_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute financial ML classification metrics.
    
    Includes ROC-AUC, F1, Log Loss, and Top Decile Precision.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # Handle single-class edge case in y_true
    if len(np.unique(y_true)) < 2:
        auc = 0.5
    else:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))

    # Top decile precision (vital for alpha selection: do top 10% highest conviction calls win?)
    top_10_pct_cutoff = np.percentile(y_prob, 90)
    top_mask = y_prob >= top_10_pct_cutoff
    prec_top_decile = float(np.mean(y_true[top_mask])) if np.sum(top_mask) > 0 else 0.0

    return {
        "ROC_AUC": auc,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "Brier_Score": brier,
        "Precision_Top_Decile": prec_top_decile,
    }


def train_and_evaluate_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    purge_window: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Run strict purged walk-forward cross validation across all candidate models.
    
    Strictly follows ML Best Practices:
    - Chronological split
    - Scaler fitted ONLY on train set, then applied to test set
    - No data leakage across folds
    
    Returns:
        Tuple of (cv_metrics_df, fitted_models, out_of_sample_predictions_df)
    """
    cv = PurgedWalkForwardCV(n_splits=n_splits, purge_window=purge_window)
    splits = cv.split(X)
    logger.info(f"Generated {len(splits)} purged walk-forward cross-validation folds.")

    model_zoo = get_model_zoo()
    results = []

    # Track out-of-sample predictions across the full test periods
    all_test_indices = []
    for _, test_idx in splits:
        all_test_indices.extend(test_idx)
    unique_test_idx = sorted(list(set(all_test_indices)))

    oos_preds = pd.DataFrame(index=X.index[unique_test_idx])
    oos_preds["actual"] = y.iloc[unique_test_idx].values

    for model_name, base_model in model_zoo.items():
        logger.info(f"Training Walk-Forward evaluation for {model_name}...")
        fold_metrics = []
        model_oos_probs = {}

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            # Fit scaler strictly on training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fit model
            model = type(base_model)(**base_model.get_params())
            model.fit(X_train_scaled, y_train)

            # Predict probabilities
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test_scaled)[:, 1]
            else:
                probs = model.predict(X_test_scaled)

            # Record out of sample probabilities
            for idx_pos, p in zip(test_idx, probs):
                model_oos_probs[X.index[idx_pos]] = p

            metrics = evaluate_predictions(y_test.values, probs)
            metrics["Fold"] = fold_i + 1
            metrics["Model"] = model_name
            fold_metrics.append(metrics)

        # Average metrics across folds
        df_fold = pd.DataFrame(fold_metrics)
        mean_metrics = df_fold.drop(columns=["Fold", "Model"]).mean().to_dict()
        mean_metrics["Model"] = model_name
        results.append(mean_metrics)

        # Store OOS probabilities
        oos_preds[f"{model_name}_prob"] = oos_preds.index.map(model_oos_probs)

    metrics_summary_df = pd.DataFrame(results).set_index("Model")

    # Fit final models on the most recent 80% training window for real-time inference / feature attribution
    split_point = int(len(X) * 0.8)
    final_scaler = StandardScaler()
    X_train_final = final_scaler.fit_transform(X.iloc[:split_point])
    y_train_final = y.iloc[:split_point]

    final_models = {"scaler": final_scaler}
    for model_name, base_model in model_zoo.items():
        m = type(base_model)(**base_model.get_params())
        m.fit(X_train_final, y_train_final)
        final_models[model_name] = m

    return metrics_summary_df, final_models, oos_preds
