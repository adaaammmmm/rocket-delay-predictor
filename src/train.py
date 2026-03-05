"""
src/train.py
Trains Stage-1 (XGBoost classifier) and Stage-2 (LightGBM regressor).
Saves models to /models/.

Run:  python src/train.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib
import json

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.data_pipeline import load_and_split, build_preprocessor, get_feature_names

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Utilities ─────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def clf_metrics(name, y_true, y_pred, y_prob) -> dict:
    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
    }
    print(f"\n  [{name}]")
    for k, v in metrics.items():
        if k != "model":
            print(f"    {k:<12} {v}")
    return metrics


# ── Stage 1: Classifier ───────────────────────────────────────────────────────

def train_classifier(X_train, X_test, y_train, y_test):
    print_section("STAGE 1 — DELAY CLASSIFIER")

    preprocessor = build_preprocessor()
    results = []

    # --- Baseline 1: Logistic Regression ---
    lr_pipe = Pipeline([
        ("prep",  preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred = lr_pipe.predict(X_test)
    y_prob = lr_pipe.predict_proba(X_test)[:, 1]
    results.append(clf_metrics("Logistic Regression", y_test, y_pred, y_prob))

    # --- Baseline 2: Decision Tree ---
    dt_pipe = Pipeline([
        ("prep",  build_preprocessor()),
        ("model", DecisionTreeClassifier(max_depth=6, class_weight="balanced")),
    ])
    dt_pipe.fit(X_train, y_train)
    y_pred = dt_pipe.predict(X_test)
    y_prob = dt_pipe.predict_proba(X_test)[:, 1]
    results.append(clf_metrics("Decision Tree", y_test, y_pred, y_prob))

    # --- XGBoost with Optuna tuning ---
    print("\n  [XGBoost] Tuning hyperparameters with Optuna (50 trials)...")

    prep_xgb    = build_preprocessor()
    X_train_t   = prep_xgb.fit_transform(X_train, y_train)
    X_test_t    = prep_xgb.transform(X_test)
    scale_pos_w = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 9),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":  scale_pos_w,
            "random_state":      42,
            "eval_metric":       "auc",
            "use_label_encoder": False,
        }
        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train_t, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos_w
    best_params["random_state"]     = 42
    best_params["eval_metric"]      = "auc"

    print(f"  Best CV ROC-AUC: {study.best_value:.4f}")
    print(f"  Best params: {best_params}")

    xgb_model = xgb.XGBClassifier(**best_params)
    xgb_model.fit(X_train_t, y_train)
    y_pred = xgb_model.predict(X_test_t)
    y_prob = xgb_model.predict_proba(X_test_t)[:, 1]
    results.append(clf_metrics("XGBoost (Tuned)", y_test, y_pred, y_prob))

    print(f"\n  Full classification report (XGBoost):")
    print(classification_report(y_test, y_pred))

    # Save
    joblib.dump(prep_xgb,   os.path.join(MODELS_DIR, "clf_preprocessor.pkl"))
    joblib.dump(xgb_model,  os.path.join(MODELS_DIR, "xgb_classifier.pkl"))

    feature_names = get_feature_names(prep_xgb)
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    return xgb_model, prep_xgb, results, X_test_t, y_test


# ── Stage 2: Regressor ────────────────────────────────────────────────────────

def train_regressor(X_del_train, X_del_test, y_del_train, y_del_test):
    print_section("STAGE 2 — DELAY DURATION REGRESSOR")

    prep_reg = build_preprocessor()
    X_tr = prep_reg.fit_transform(X_del_train, y_del_train)
    X_te = prep_reg.transform(X_del_test)

    lgb_model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X_tr, y_del_train)
    y_pred = lgb_model.predict(X_te)

    mae  = mean_absolute_error(y_del_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_del_test, y_pred))
    print(f"  MAE:  {mae:.2f} hours")
    print(f"  RMSE: {rmse:.2f} hours")

    joblib.dump(prep_reg,  os.path.join(MODELS_DIR, "reg_preprocessor.pkl"))
    joblib.dump(lgb_model, os.path.join(MODELS_DIR, "lgb_regressor.pkl"))

    return lgb_model, prep_reg, {"mae": round(mae, 3), "rmse": round(rmse, 3)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n🚀 Rocket Launch Delay Predictor — Model Training")

    splits = load_and_split()
    X_train, X_test, y_train, y_test = splits["clf"]
    X_del_train, X_del_test, y_del_train, y_del_test = splits["reg"]

    xgb_model, prep_clf, clf_results, X_test_t, y_test_clf = train_classifier(
        X_train, X_test, y_train, y_test
    )

    lgb_model, prep_reg, reg_results = train_regressor(
        X_del_train, X_del_test, y_del_train, y_del_test
    )

    # Save results summary
    summary = {
        "classifier_results": clf_results,
        "regressor_results":  reg_results,
    }
    with open(os.path.join(MODELS_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print_section("TRAINING COMPLETE")
    print(f"  Models saved → {MODELS_DIR}/")
    print(f"  Best classifier ROC-AUC: {max(r['roc_auc'] for r in clf_results):.4f}")
    print(f"  Regressor MAE: {reg_results['mae']} hrs")


if __name__ == "__main__":
    main()
