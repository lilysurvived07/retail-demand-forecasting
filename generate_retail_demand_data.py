import os
import numpy as np
import pandas as pd


def generate_retail_demand_data(
    start_date="2024-01-01",
    end_date="2024-12-31",
    n_stores=20,
    n_products=50,
    random_state=42,
):
    np.random.seed(random_state)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    store_ids = [f"S{str(i).zfill(3)}" for i in range(1, n_stores + 1)]
    product_ids = [f"P{str(i).zfill(4)}" for i in range(1, n_products + 1)]

    categories = ["Beverages", "Snacks", "Household", "Dairy", "Personal Care"]
    product_category = {
        product_id: np.random.choice(categories)
        for product_id in product_ids
    }

    rows = []
    for date in dates:
        day_of_week = date.dayofweek
        month = date.month
        weekend_flag = 1 if day_of_week in [5, 6] else 0
        holiday_flag = 1 if (month == 12 and date.day >= 20) or (month == 1 and date.day == 1) else 0
        payday_flag = 1 if date.day in [25, 26, 27, 28, 29, 30, 31, 1, 2] else 0

        for store_id in store_ids:
            store_size = np.random.choice(["Small", "Medium", "Large"], p=[0.2, 0.5, 0.3])
            store_multiplier = {"Small": 0.8, "Medium": 1.0, "Large": 1.25}[store_size]

            for product_id in product_ids:
                category = product_category[product_id]

                base_price = {
                    "Beverages": np.random.uniform(15, 40),
                    "Snacks": np.random.uniform(10, 30),
                    "Household": np.random.uniform(30, 100),
                    "Dairy": np.random.uniform(12, 35),
                    "Personal Care": np.random.uniform(25, 80),
                }[category]

                promo_flag = np.random.binomial(1, 0.15 + 0.05 * payday_flag)
                promo_discount = np.random.uniform(0.05, 0.20) if promo_flag else 0
                price = round(base_price * (1 - promo_discount), 2)

                seasonal_effect = 1 + 0.1 * np.sin(2 * np.pi * month / 12)
                weekend_effect = 1.12 if weekend_flag and category in ["Beverages", "Snacks", "Dairy"] else 1.0
                holiday_effect = 1.25 if holiday_flag else 1.0
                payday_effect_mult = 1.10 if payday_flag else 1.0
                promo_effect = 1.18 if promo_flag else 1.0

                category_base_demand = {
                    "Beverages": 35,
                    "Snacks": 28,
                    "Household": 16,
                    "Dairy": 24,
                    "Personal Care": 14,
                }[category]

                demand = (
                    category_base_demand
                    * store_multiplier
                    * seasonal_effect
                    * weekend_effect
                    * holiday_effect
                    * payday_effect_mult
                    * promo_effect
                    * (40 / max(price, 1)) ** 0.35
                    * np.random.lognormal(mean=0, sigma=0.18)
                )

                stock_available = int(np.random.uniform(20, 150) * store_multiplier)
                units_sold = int(max(0, min(round(demand), stock_available)))

                rows.append(
                    {
                        "date": date,
                        "store_id": store_id,
                        "store_size": store_size,
                        "product_id": product_id,
                        "category": category,
                        "price": price,
                        "promo_flag": promo_flag,
                        "holiday_flag": holiday_flag,
                        "payday_flag": payday_flag,
                        "day_of_week": day_of_week,
                        "month": month,
                        "weekend_flag": weekend_flag,
                        "stock_available": stock_available,
                        "units_sold": units_sold,
                    }
                )

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = generate_retail_demand_data()
    output_path = os.path.join("data", "retail_demand_data.csv")
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Rows: {len(df):,}")
    print(df.head())