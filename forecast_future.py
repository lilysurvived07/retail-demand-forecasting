import os
import json
import joblib
import numpy as np
import pandas as pd


DATA_PATH = os.path.join("data", "retail_demand_data.csv")
OUTPUT_DIR = "outputs"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)

    group_cols = ["store_id", "product_id"]
    df["lag_1"] = df.groupby(group_cols)["units_sold"].shift(1)
    df["lag_7"] = df.groupby(group_cols)["units_sold"].shift(7)
    df["lag_14"] = df.groupby(group_cols)["units_sold"].shift(14)

    df["rolling_mean_7"] = (
        df.groupby(group_cols)["units_sold"]
        .transform(lambda s: s.shift(1).rolling(7).mean())
    )

    df["rolling_mean_14"] = (
        df.groupby(group_cols)["units_sold"]
        .transform(lambda s: s.shift(1).rolling(14).mean())
    )

    return df


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    model = joblib.load(os.path.join(OUTPUT_DIR, "forecast_model.pkl"))

    with open(os.path.join(OUTPUT_DIR, "model_features.json"), "r", encoding="utf-8") as f:
        feature_cols = json.load(f)["feature_cols"]

    df = add_features(df)
    latest = df.dropna().sort_values("date").groupby(["store_id", "product_id"]).tail(1).copy()

    latest["date"] = latest["date"] + pd.Timedelta(days=1)
    latest["day_of_week"] = latest["date"].dt.dayofweek
    latest["month"] = latest["date"].dt.month
    latest["weekend_flag"] = latest["day_of_week"].isin([5, 6]).astype(int)
    latest["holiday_flag"] = 0
    latest["payday_flag"] = latest["date"].dt.day.isin([25, 26, 27, 28, 29, 30, 31, 1, 2]).astype(int)

    future_df = pd.get_dummies(latest, columns=["store_size", "category"], drop_first=True)

    for col in feature_cols:
        if col not in future_df.columns:
            future_df[col] = 0

    future_df = future_df[feature_cols]
    preds = model.predict(future_df)

    result = latest[["date", "store_id", "product_id"]].copy()
    result["forecast_units"] = np.round(preds, 0).astype(int)

    result.to_csv(os.path.join(OUTPUT_DIR, "future_forecast.csv"), index=False)

    print("Future forecast saved to outputs/future_forecast.csv")
    print(result.head())