"""
Predictive model for women's rugby match attendance.

Given the small-N reality of this dataset (see features.py docstring),
this script:

  1. Uses Leave-One-Out Cross-Validation (LOOCV) instead of a train/test
     split or k-fold CV, since a random split on ~50-100 rows gives you
     an unstable, meaningless test set.
  2. Fits two models: Ridge regression (interpretable, coefficients you
     can report directly) and a small Random Forest (captures
     non-linearities / interactions, e.g. "final AND host nation" effects
     that a linear model would miss).
  3. Reports feature importance / coefficients as the primary deliverable,
     not just accuracy metrics -- with this little data, "what predicts
     attendance" is more defensible than "how accurately can I predict it".

Run with more rows in data/matches_verified.csv as you expand it from
Wikipedia's Women's Six Nations / WXV / Rugby World Cup match-by-match
pages (each has an attendance column in its results table).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from features import (
    FEATURE_COLUMNS_CATEGORICAL,
    FEATURE_COLUMNS_NUMERIC,
    TARGET_COLUMN,
    add_features,
    load_matches,
)

DATA_PATH = "/Users/svaradeshmukh/WRugby_Match_Prediction/data/matches_with_capacity.csv"
MIN_ROWS_FOR_ML_WARNING = 30


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, FEATURE_COLUMNS_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLUMNS_CATEGORICAL),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "ridge": Pipeline([("prep", preprocessor), ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline(
            [("prep", preprocessor), ("model", RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42))]
        ),
    }


def loocv_evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds[test_idx] = pipeline.predict(X.iloc[test_idx])
    mae = mean_absolute_error(y, preds)
    mape = mean_absolute_percentage_error(y, preds)
    return {"mae": mae, "mape": mape, "predictions": preds}


def report_ridge_coefficients(pipeline: Pipeline, X: pd.DataFrame) -> pd.Series:
    pipeline.fit(X, X[TARGET_COLUMN]) if TARGET_COLUMN in X.columns else None
    prep = pipeline.named_steps["prep"]
    feature_names = prep.get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_
    return pd.Series(coefs, index=feature_names).sort_values(key=abs, ascending=False)


def main():
    df = load_matches(DATA_PATH)
    df = add_features(df)

    n_rows = len(df)
    print(f"Loaded {n_rows} matches.\n")
    if n_rows < MIN_ROWS_FOR_ML_WARNING:
        print(
            f"NOTE: only {n_rows} rows loaded. This is a methodology demo, not a\n"
            f"trustworthy trained model yet. Expand matches_verified.csv toward\n"
            f"{MIN_ROWS_FOR_ML_WARNING}+ rows (Six Nations + WXV + more World Cup\n"
            f"matches from Wikipedia) before drawing real conclusions.\n"
        )

    X = df[FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL]
    y = df[TARGET_COLUMN]

    models = build_models()

    for name, pipeline in models.items():
        print(f"--- {name} (Leave-One-Out CV) ---")
        results = loocv_evaluate(pipeline, X, y)
        print(f"MAE:  {results['mae']:,.0f} attendees")
        print(f"MAPE: {results['mape']:.1%}")
        comparison = pd.DataFrame(
            {"match_id": df["match_id"], "actual": y.values, "predicted": results["predictions"].round(0)}
        )
        print(comparison.to_string(index=False))
        print()

    # Fit ridge on full data for interpretable coefficients
    print("--- Ridge coefficients (full-data fit, for interpretation only) ---")
    ridge_pipeline = models["ridge"]
    ridge_pipeline.fit(X, y)
    prep = ridge_pipeline.named_steps["prep"]
    feature_names = prep.get_feature_names_out()
    coefs = pd.Series(ridge_pipeline.named_steps["model"].coef_, index=feature_names)
    print(coefs.sort_values(key=abs, ascending=False).to_string())


if __name__ == "__main__":
    main()
