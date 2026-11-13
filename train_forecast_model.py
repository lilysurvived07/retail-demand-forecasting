import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


DATA_PATH = os.path.join("data", "retail_demand_data.csv")
OUTPUT_DIR = "outputs"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
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


def prepare_model_data(df: pd.DataFrame):
    df = add_time_features(df)
    df = df.dropna().copy()

    df = pd.get_dummies(df, columns=["store_size", "category"], drop_first=True)

    feature_cols = [
        "price",
        "promo_flag",
        "holiday_flag",
        "payday_flag",
        "day_of_week",
        "month",
        "weekend_flag",
        "stock_available",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14",
    ] + [c for c in df.columns if c.startswith("store_size_") or c.startswith("category_")]

    split_date = df["date"].quantile(0.8)

    train_df = df[df["date"] <= split_date].copy()
    test_df = df[df["date"] > split_date].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["units_sold"]

    X_test = test_df[feature_cols]
    y_test = test_df["units_sold"]

    return train_df, test_df, X_train, X_test, y_train, y_test, feature_cols


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    train_df, test_df, X_train, X_test, y_train, y_test, feature_cols = prepare_model_data(df)

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds)

    metrics = {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
    }

    with open(os.path.join(OUTPUT_DIR, "model_metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    pd.DataFrame({
        "actual": y_test.values,
        "predicted": preds
    }).to_csv(os.path.join(OUTPUT_DIR, "forecast_results.csv"), index=False)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.xlabel("Importance")
    plt.title("Top Forecast Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()

    sample_n = min(500, len(y_test))
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values[:sample_n], label="Actual")
    plt.plot(preds[:sample_n], label="Predicted")
    plt.xlabel("Observation")
    plt.ylabel("Units Sold")
    plt.title("Actual vs Predicted Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "forecast_plot.png"))
    plt.close()

    model_payload = {
        "feature_cols": feature_cols,
    }
    with open(os.path.join(OUTPUT_DIR, "model_features.json"), "w", encoding="utf-8") as f:
        json.dump(model_payload, f)

    import joblib
    joblib.dump(model, os.path.join(OUTPUT_DIR, "forecast_model.pkl"))

    print("Training complete.")
    print(metrics)