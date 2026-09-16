"""Model Explainability and Alpha Factor Attribution Module.

Computes Gini Feature Importance, Permutation Feature Importance on out-of-sample data,
and aggregated domain factor group attributions (Momentum vs Volatility vs Liquidity vs Macro).
"""

from __future__ import annotations
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


def compute_feature_importance(
    model: Any,
    feature_names: List[str],
    factor_groups: Dict[str, List[str]],
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute global feature importance and grouped category importance.
    
    Args:
        model: Trained scikit-learn classifier (RandomForest or LogisticRegression).
        feature_names: List of column names in feature matrix.
        factor_groups: Dictionary mapping group names to list of feature names.
        X_val: Optional validation matrix for permutation importance.
        y_val: Optional validation targets for permutation importance.
        
    Returns:
        Tuple of (feature_importance_df, group_importance_df).
    """
    importances = {}

    # 1. Intrinsic Model Feature Weights
    if hasattr(model, "feature_importances_"):
        raw_imp = model.feature_importances_
        importances["Gini_Importance"] = raw_imp
    elif hasattr(model, "coef_"):
        raw_coef = np.abs(model.coef_[0])
        importances["Absolute_Coefficient"] = raw_coef

    # 2. Permutation Importance (Out-of-sample validation impact)
    if X_val is not None and y_val is not None:
        try:
            perm = permutation_importance(
                model, X_val, y_val, n_repeats=5, random_state=42, scoring="roc_auc"
            )
            importances["Permutation_Importance"] = perm.importances_mean
        except Exception as e:
            logger.warning(f"Permutation importance skipped: {e}")

    feat_df = pd.DataFrame(importances, index=feature_names)
    # Prefer Gini_Importance for stable positive tree weights, or absolute permutation score
    primary_metric = "Gini_Importance" if "Gini_Importance" in feat_df.columns else (
        "Absolute_Coefficient" if "Absolute_Coefficient" in feat_df.columns else "Permutation_Importance"
    )

    # Ensure non-negative contribution for percentage breakdown
    weights = np.maximum(0.0, feat_df[primary_metric].values)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    feat_df["Relative_Importance_Pct"] = (weights / weights.sum()) * 100.0
    feat_df = feat_df.sort_values(by="Relative_Importance_Pct", ascending=False)

    # 3. Grouped Category Importance
    group_scores = {}
    for group, cols in factor_groups.items():
        present_cols = [c for c in cols if c in feat_df.index]
        if present_cols:
            group_scores[group] = float(feat_df.loc[present_cols, "Relative_Importance_Pct"].sum())

    group_df = pd.DataFrame(list(group_scores.items()), columns=["Factor_Category", "Share_Pct"])
    group_df["Share_Pct"] = (group_df["Share_Pct"] / group_df["Share_Pct"].sum()) * 100.0
    group_df = group_df.sort_values(by="Share_Pct", ascending=False).reset_index(drop=True)

    return feat_df, group_df
