"""
FairAI Pro — Core Bias Detection Engine
Implements Statistical Parity analysis with composite fairness scoring.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def get_dataset_info(filepath):
    """
    Read a CSV and return metadata about its columns for UI configuration.
    """
    df = pd.read_csv(filepath)
    columns = []
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "unique_values": int(df[col].nunique()),
            "null_count": int(df[col].isnull().sum()),
            "sample_values": df[col].dropna().unique()[:10].tolist()
        }
        # Convert numpy types to native Python for JSON
        col_info["sample_values"] = [
            int(v) if isinstance(v, (np.integer,)) else
            float(v) if isinstance(v, (np.floating,)) else
            str(v) for v in col_info["sample_values"]
        ]
        columns.append(col_info)

    return {
        "filename": filepath.split('\\')[-1].split('/')[-1],
        "rows": len(df),
        "columns": columns,
        "preview": df.head(10).to_dict(orient='records')
    }


def analyze_bias(df, target_col, sensitive_col, privileged_value):
    """
    Core bias detection pipeline.

    Steps:
        1. Ingest & validate data
        2. Train Logistic Regression classifier
        3. Generate predictions on test set
        4. Disaggregate predictions by sensitive attribute
        5. Compute fairness metrics (Selection Rate, SPD, DIR)
        6. Calculate composite fairness score (0-100)

    Args:
        df: pandas DataFrame with the dataset
        target_col: name of the binary target column (0/1)
        sensitive_col: name of the sensitive attribute column
        privileged_value: value representing the privileged group

    Returns:
        dict with comprehensive bias analysis results
    """

    # ── Step 1: Validate & Clean ─────────────────────────────────
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    if sensitive_col not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_col}' not found in dataset")

    df = df.dropna(subset=[target_col, sensitive_col]).copy()

    # Ensure target is binary
    df[target_col] = df[target_col].astype(int)
    unique_targets = df[target_col].unique()
    if not set(unique_targets).issubset({0, 1}):
        raise ValueError(f"Target column must be binary (0/1). Found: {unique_targets}")

    # Convert privileged_value to match column dtype
    try:
        if df[sensitive_col].dtype in ['int64', 'float64']:
            privileged_value = type(df[sensitive_col].iloc[0])(privileged_value)
    except (ValueError, TypeError):
        pass

    # ── Step 2: Prepare Features & Train Model ───────────────────
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Store original sensitive column before encoding
    sensitive_values = X[sensitive_col].copy()

    # Encode categorical features
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Logistic Regression
    model = LogisticRegression(max_iter=100, solver="liblinear", random_state=42)
    model.fit(X_train, y_train)

    # ── Step 3: Generate Predictions ─────────────────────────────
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Model performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # ── Step 4: Disaggregate by Group ────────────────────────────
    test_sensitive = sensitive_values.loc[X_test.index]

    priv_mask = test_sensitive == privileged_value
    unpriv_mask = ~priv_mask

    priv_preds = y_pred[priv_mask]
    unpriv_preds = y_pred[unpriv_mask]
    priv_actual = y_test.values[priv_mask]
    unpriv_actual = y_test.values[unpriv_mask]

    # ── Step 5: Compute Fairness Metrics ─────────────────────────

    # 5a. Selection Rates
    n_priv = len(priv_preds)
    n_unpriv = len(unpriv_preds)

    sr_priv = priv_preds.sum() / n_priv if n_priv > 0 else 0.0
    sr_unpriv = unpriv_preds.sum() / n_unpriv if n_unpriv > 0 else 0.0

    # 5b. Statistical Parity Difference
    spd = sr_unpriv - sr_priv

    # 5c. Disparate Impact Ratio
    dir_ratio = (sr_unpriv / sr_priv) if sr_priv > 0 else 0.0

    # ── Step 6: Composite Fairness Score (0-100) ─────────────────
    spd_score = max(0, (1 - abs(spd) / 0.5)) * 50
    dir_score = min(dir_ratio / 1.0, 1.0) * 50
    fairness_score = max(0, min(100, round(spd_score + dir_score, 1)))

    # Determine bias level
    if fairness_score >= 80:
        bias_level = "Fair"
        bias_color = "green"
    elif fairness_score >= 60:
        bias_level = "Moderate Bias"
        bias_color = "yellow"
    else:
        bias_level = "Significant Bias"
        bias_color = "red"

    # Determine bias direction
    if abs(spd) < 0.05:
        bias_direction = "No significant directional bias detected"
    elif spd < 0:
        unpriv_groups = [str(v) for v in test_sensitive[unpriv_mask].unique()]
        bias_direction = f"Bias against: {', '.join(unpriv_groups)}"
    else:
        bias_direction = f"Bias against: {privileged_value}"

    # Group-level accuracy
    acc_priv = (priv_preds == priv_actual).mean() if n_priv > 0 else 0
    acc_unpriv = (unpriv_preds == unpriv_actual).mean() if n_unpriv > 0 else 0

    # Get unique group labels
    group_labels = sorted(test_sensitive.unique().tolist(), key=str)

    # Per-group detailed breakdown
    group_details = []
    for group_val in group_labels:
        g_mask = test_sensitive == group_val
        g_preds = y_pred[g_mask]
        g_actual = y_test.values[g_mask]
        g_total = len(g_preds)
        g_positive = int(g_preds.sum())
        g_sr = g_positive / g_total if g_total > 0 else 0
        g_acc = (g_preds == g_actual).mean() if g_total > 0 else 0
        group_details.append({
            "group": str(group_val),
            "is_privileged": bool(group_val == privileged_value),
            "total": g_total,
            "positive_predictions": g_positive,
            "negative_predictions": g_total - g_positive,
            "selection_rate": round(g_sr, 4),
            "accuracy": round(float(g_acc), 4)
        })

    # Feature importance (from logistic regression coefficients)
    feature_names = X.columns.tolist()
    coefficients = model.coef_[0]
    feature_importance = sorted(
        [{"feature": name, "importance": round(float(abs(coef)), 4), "direction": "positive" if coef > 0 else "negative"}
         for name, coef in zip(feature_names, coefficients)],
        key=lambda x: x["importance"],
        reverse=True
    )

    # ── Build Response ───────────────────────────────────────────
    return {
        # Model Performance
        "model": {
            "type": "Logistic Regression",
            "accuracy": round(float(accuracy) * 100, 2),
            "precision": round(float(precision) * 100, 2),
            "recall": round(float(recall) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "confusion_matrix": cm.tolist(),
            "feature_importance": feature_importance
        },

        # Fairness Metrics
        "fairness": {
            "fairness_score": fairness_score,
            "bias_level": bias_level,
            "bias_color": bias_color,
            "bias_direction": bias_direction,
            "selection_rate_privileged": round(float(sr_priv), 4),
            "selection_rate_unprivileged": round(float(sr_unpriv), 4),
            "statistical_parity_difference": round(float(spd), 4),
            "disparate_impact_ratio": round(float(dir_ratio), 4),
            "spd_fair_range": [-0.10, 0.10],
            "dir_fair_range": [0.80, 1.25]
        },

        # Group Details
        "groups": {
            "privileged_group": str(privileged_value),
            "unprivileged_group": [str(v) for v in test_sensitive[unpriv_mask].unique()],
            "privileged_count": int(n_priv),
            "unprivileged_count": int(n_unpriv),
            "accuracy_privileged": round(float(acc_priv) * 100, 2),
            "accuracy_unprivileged": round(float(acc_unpriv) * 100, 2),
            "details": group_details
        },

        # Dataset Info
        "dataset": {
            "total_samples": len(df),
            "test_samples": len(y_test),
            "train_samples": len(y_train),
            "positive_rate": round(float(y[y == 1].count() / len(y)), 4),
            "sensitive_column": sensitive_col,
            "target_column": target_col
        }
    }


def _selection_rate_difference(y_pred, sensitive_series):
    """
    Fairness score used for mitigation analysis:
    difference between max and min group selection rates.
    """
    rates = []
    for group in sensitive_series.dropna().unique():
        group_mask = sensitive_series == group
        if group_mask.sum() == 0:
            continue
        rates.append(float(y_pred[group_mask].mean()))

    if len(rates) < 2:
        return 0.0
    return max(rates) - min(rates)


def _fairness_score_from_selection_rate_diff(selection_rate_difference):
    """
    Convert selection-rate difference to a user-friendly fairness score:
    higher is better, scaled to 0-100.
    """
    fairness_score = (1 - float(selection_rate_difference)) * 100
    fairness_score = max(0.0, min(100.0, fairness_score))
    return round(fairness_score, 2)


def analyze_mitigation(df, target_col, sensitive_col):
    """
    Mitigation analysis pipeline:
    1) Train Logistic Regression with all features
    2) Compute fairness score (selection rate difference)
    3) Retrain without sensitive column
    4) Compute fairness score again
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    if sensitive_col not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_col}' not found in dataset")

    work_df = df.dropna(subset=[target_col, sensitive_col]).copy()

    if work_df.empty:
        raise ValueError("Dataset is empty after dropping missing target/sensitive values")

    y_raw = work_df[target_col]
    if y_raw.nunique() != 2:
        raise ValueError("Target column must be binary for logistic regression")
    y = LabelEncoder().fit_transform(y_raw.astype(str))

    X_full = work_df.drop(columns=[target_col])
    sensitive_values = work_df[sensitive_col].copy()

    X_train_full, X_test_full, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X_full,
        y,
        sensitive_values,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train_full_enc = pd.get_dummies(X_train_full, drop_first=False)
    X_test_full_enc = pd.get_dummies(X_test_full, drop_first=False).reindex(
        columns=X_train_full_enc.columns, fill_value=0
    )

    model_full = LogisticRegression(max_iter=100, solver="liblinear", random_state=42)
    model_full.fit(X_train_full_enc, y_train)
    y_pred_full = model_full.predict(X_test_full_enc)
    selection_rate_difference_before = _selection_rate_difference(
        y_pred_full,
        sensitive_test.reset_index(drop=True)
    )
    fairness_score_before = _fairness_score_from_selection_rate_diff(selection_rate_difference_before)

    X_train_mitigated = X_train_full.drop(columns=[sensitive_col], errors='ignore')
    X_test_mitigated = X_test_full.drop(columns=[sensitive_col], errors='ignore')

    X_train_mitigated_enc = pd.get_dummies(X_train_mitigated, drop_first=False)
    X_test_mitigated_enc = pd.get_dummies(X_test_mitigated, drop_first=False).reindex(
        columns=X_train_mitigated_enc.columns, fill_value=0
    )

    model_mitigated = LogisticRegression(max_iter=100, solver="liblinear", random_state=42)
    model_mitigated.fit(X_train_mitigated_enc, y_train)
    y_pred_mitigated = model_mitigated.predict(X_test_mitigated_enc)
    selection_rate_difference_after = _selection_rate_difference(
        y_pred_mitigated,
        sensitive_test.reset_index(drop=True)
    )
    fairness_score_after = _fairness_score_from_selection_rate_diff(selection_rate_difference_after)
    improvement = round(fairness_score_after - fairness_score_before, 2)

    return {
        "before_score": float(fairness_score_before),
        "after_score": float(fairness_score_after),
        "improvement": float(improvement)
    }
