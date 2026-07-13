"""
Feature engineering for women's rugby match attendance prediction.

Design notes
------------
This dataset will be SMALL (dozens, maybe low hundreds of rows even after
you've fully populated it from Six Nations / WXV / World Cup match lists).
That shapes every choice below:

- Features are kept simple and interpretable (no high-cardinality
  one-hot explosions that would out-number your rows).
- Team "quality" is represented by World Rugby ranking if you add it,
  not by team dummy variables (15+ teams would blow up a ~50-100 row
  dataset).
- Time is captured as a simple trend feature (days since a fixed anchor
  date) so the model can pick up the sport's rapid post-2023 growth,
  rather than needing many years of data to learn seasonality.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

ANCHOR_DATE = pd.Timestamp("2010-01-01")  # arbitrary early anchor for a time-trend feature


def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Time trend ---
    df["days_since_anchor"] = (df["date"] - ANCHOR_DATE).dt.days
    df["year"] = df["date"].dt.year

    # --- Competition tier (ordinal-ish, but kept as category for tree models) ---
    df["competition"] = df["competition"].astype("category")

    # --- Match format: 15s vs 7s draw very differently sized crowds ---
    df["format"] = df["format"].astype("category")

    # --- Binary flags already in source data ---
    df["is_final"] = df["is_final"].fillna(0).astype(int)
    df["is_opener"] = df["is_opener"].fillna(0).astype(int)

    # --- Host-nation playing flag (a genuinely predictive feature in rugby -
    #     England matches in England draw far bigger crowds) ---
    df["host_nation_playing"] = (
        (df["home_team"] == "England") & (df["city"].isin(["London", "Sunderland", "Manchester", "Bristol", "Exeter"]))
    ).astype(int)

    
    # --- Venue capacity: the single biggest confound on raw attendance.
    #     A sellout at a 5,000-seat ground and a sellout at Twickenham (82,000)
    #     look wildly different in raw numbers for reasons that have nothing
    #     to do with demand. Use log scale since capacity spans two orders
    #     of magnitude (5k-82k) in this dataset. Missing values (a venue not
    #     yet in data/venue_capacities.csv) are left as NaN rather than
    #     silently imputed -- see model.py's handling of this.
    if "venue_capacity" in df.columns:
        df["log_venue_capacity"] = np.log(df["venue_capacity"])
    else:
        df["log_venue_capacity"] = float("nan")

    return df

FEATURE_COLUMNS_NUMERIC = ["days_since_anchor", "year", "is_final", "is_opener", "host_nation_playing", "log_venue_capacity"]
FEATURE_COLUMNS_CATEGORICAL = ["competition", "format"]
TARGET_COLUMN = "attendance"


if __name__ == "__main__":
    df = load_matches("/Users/svaradeshmukh/WRugby_Match_Prediction/data/matches_with_capacity.csv")
    df = add_features(df)
    print(df[["match_id", "date"] + FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL + [TARGET_COLUMN]])
