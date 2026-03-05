"""
src/data_pipeline.py
Loads raw launch data, engineers features, and returns train/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

RAW_PATH       = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "launches.csv")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

NUMERIC_FEATURES = [
    "rocket_age_years",
    "provider_success_rate",
    "site_success_rate",
    "launches_this_month",
    "temp_celsius",
    "wind_speed_kmh",
    "precipitation_mm",
    "cloud_cover_pct",
    "is_crewed",
]

CATEGORICAL_FEATURES = [
    "company",
    "rocket",
    "site",
    "mission_type",
    "launch_month",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_CLF   = "delayed"
TARGET_REG   = "delay_hours"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, parse_dates=["launch_date"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add or refine any feature columns."""
    df = df.copy()

    # Season (1=Winter,2=Spring,3=Summer,4=Autumn) for Northern hemisphere bias
    df["season"] = df["launch_month"].map(
        lambda m: 1 if m in [12, 1, 2] else (2 if m in [3, 4, 5] else (3 if m in [6, 7, 8] else 4))
    )

    # High-wind flag
    df["high_wind"] = (df["wind_speed_kmh"] > 40).astype(int)

    # Decade bucket (captures era-level technology improvements)
    df["decade"] = (df["launch_year"] // 10 * 10).astype(str)

    # Combined weather severity score
    df["weather_severity"] = (
        df["wind_speed_kmh"] / 100 +
        df["precipitation_mm"] / 20 +
        df["cloud_cover_pct"] / 200
    )

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    extended_num = NUMERIC_FEATURES + ["season", "high_wind", "weather_severity"]
    extended_cat = CATEGORICAL_FEATURES + ["decade"]

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, extended_num),
        ("cat", cat_pipe,     extended_cat),
    ], remainder="drop")
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return flat list of feature names after transform."""
    num_names = preprocessor.named_transformers_["num"]["imputer"].feature_names_in_.tolist()
    cat_enc   = preprocessor.named_transformers_["cat"]["encoder"]
    cat_names = cat_enc.get_feature_names_out().tolist()
    return num_names + cat_names


def load_and_split(test_size: float = 0.2, random_state: int = 42):
    """Full pipeline: load → engineer → split → return."""
    df = load_raw()
    df = engineer_features(df)

    X = df[ALL_FEATURES + ["season", "high_wind", "weather_severity", "decade"]]
    y_clf = df[TARGET_CLF]
    y_reg  = df.loc[df[TARGET_CLF] == 1, TARGET_REG]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
    )

    # Regression split: only delayed rows
    delayed_idx = y_clf[y_clf == 1].index
    X_del = df.loc[delayed_idx, ALL_FEATURES + ["season", "high_wind", "weather_severity", "decade"]]
    y_del = df.loc[delayed_idx, TARGET_REG]
    X_del_train, X_del_test, y_del_train, y_del_test = train_test_split(
        X_del, y_del, test_size=test_size, random_state=random_state
    )

    return {
        "clf":  (X_train, X_test, y_train, y_test),
        "reg":  (X_del_train, X_del_test, y_del_train, y_del_test),
        "full_df": df,
    }


if __name__ == "__main__":
    splits = load_and_split()
    X_train, X_test, y_train, y_test = splits["clf"]
    print(f"Classification train: {X_train.shape}, test: {X_test.shape}")
    print(f"Delay rate train: {y_train.mean():.1%}")
