import pandas as pd
import lightgbm as lgb
from feature_creator import create_features

REQUIRED_COLUMNS = ["hour", "banner_pos", "device_type", "device_model", "site_id", "app_id"]
DEFAULT_ROW = {
    "hour": 1300,
    "banner_pos": 0,
    "device_type": 1,
    "device_model": "model_abc",
    "site_id": "site_001",
    "app_id": "app_001"
}

def run_sql_to_ctr_predictions(sql_query, spark_df):
    try:
        # Register Spark SQL table
        spark_df.createOrReplaceTempView("ctr_df")

        # Run SQL query
        input_df = spark.sql(sql_query).toPandas()
        user_requested_cols = input_df.columns.tolist()

        # Edge case 1: Empty result from SQL
        if input_df.empty:
            input_df = pd.DataFrame([DEFAULT_ROW])
            user_requested_cols = list(DEFAULT_ROW.keys())

        # ✅ Edge case 2: Query returns aggregate (skip prediction)
        # If all returned columns are numeric (and no inputs match model features), skip
        if not any(col in input_df.columns for col in REQUIRED_COLUMNS):
            input_df["predicted_ctr"] = None  # Optional: Placeholder
            return input_df

        # Fill required model features
        for col in REQUIRED_COLUMNS:
            if col not in input_df.columns:
                input_df[col] = DEFAULT_ROW[col]
            else:
                input_df[col] = input_df[col].fillna(DEFAULT_ROW[col])

        # Drop any existing predicted_ctr variant
        input_df.columns = [col.strip() for col in input_df.columns]
        user_requested_cols = [col.strip() for col in user_requested_cols]
        cols_to_drop = [col for col in input_df.columns if col.lower() == "predicted_ctr"]
        input_df = input_df.drop(columns=cols_to_drop, errors="ignore")
        user_requested_cols = [col for col in user_requested_cols if col.lower() != "predicted_ctr"]

        # Feature engineering
        features_df = create_features(input_df)

        # Load model and predict
        booster = lgb.Booster(model_file="/databricks/driver/ctr-predictor-module/model/ctr_model.txt")
        preds = booster.predict(features_df)
        input_df["predicted_ctr"] = preds

        return input_df[user_requested_cols + ["predicted_ctr"]]

    except Exception as e:
        return pd.DataFrame({"error": [f" Error: {str(e)}"]})
