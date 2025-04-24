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
    "app_id": "app_001",
    "app_category": "07d7df22"  # Add default top category for simulation
}

# Columns that can be safely simulated in fallback
SIMULATABLE_FEATURES = {
    "device_type", "app_category", "day_of_week", "hour_of_day", "banner_pos"
}

def run_sql_to_ctr_predictions(sql_query, spark_df):
    try:
        # Register SQL table
        spark_df.createOrReplaceTempView("ctr_df")

        # Run the GPT-generated SQL query
        input_df = spark.sql(sql_query).toPandas()
        user_requested_cols = input_df.columns.tolist()

        # Fallback: SQL returned nothing
        if input_df.empty:
            input_df = pd.DataFrame([DEFAULT_ROW])
            user_requested_cols = list(DEFAULT_ROW.keys())

        # Normalize column names
        input_df.columns = [col.strip() for col in input_df.columns]
        user_requested_cols = [col.strip() for col in user_requested_cols]

        # Drop any existing predicted_ctr columns (case-insensitive)
        cols_to_drop = [col for col in input_df.columns if col.lower() == "predicted_ctr"]
        input_df = input_df.drop(columns=cols_to_drop, errors="ignore")
        user_requested_cols = [col for col in user_requested_cols if col.lower() != "predicted_ctr"]

        # CASE 1: Row-wise model inference (all or most features are present)
        if any(col in input_df.columns for col in REQUIRED_COLUMNS):
            for col in REQUIRED_COLUMNS:
                if col not in input_df.columns:
                    input_df[col] = DEFAULT_ROW[col]
                else:
                    input_df[col] = input_df[col].fillna(DEFAULT_ROW[col])
            features_df = create_features(input_df)
            booster = lgb.Booster(model_file="/databricks/driver/ctr-predictor-module/model/ctr_model.txt")
            preds = booster.predict(features_df)
            input_df["predicted_ctr"] = preds
            return input_df[user_requested_cols + ["predicted_ctr"]]

        # CASE 2: Group-level simulation using simulatable column (like app_category, device_type, etc.)
        simulatable_cols = [col for col in input_df.columns if col in SIMULATABLE_FEATURES]
        if len(simulatable_cols) == 1:
            sim_col = simulatable_cols[0]
            booster = lgb.Booster(model_file="/databricks/driver/ctr-predictor-module/model/ctr_model.txt")
            pred_rows = []

            for _, row in input_df.iterrows():
                sim_row = DEFAULT_ROW.copy()
                sim_row[sim_col] = row[sim_col]
                features_df = create_features(sim_row)
                pred = booster.predict(features_df)[0]
                row["predicted_ctr"] = pred
                pred_rows.append(row)

            result_df = pd.DataFrame(pred_rows)
            return result_df[user_requested_cols + ["predicted_ctr"]]

        # CASE 3: No model features or simulatable fallback — just return single default prediction
        booster = lgb.Booster(model_file="/databricks/driver/ctr-predictor-module/model/ctr_model.txt")
        features_df = create_features(DEFAULT_ROW)
        fallback_pred = booster.predict(features_df)[0]
        input_df["predicted_ctr"] = fallback_pred
        return input_df[user_requested_cols + ["predicted_ctr"]]

    except Exception as e:
        return pd.DataFrame({"error": [f" Error: {str(e)}"]})
