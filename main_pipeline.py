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

# Features that support dynamic fallback simulation (used in training)
SIMULATABLE_FEATURES = {"device_type", "app_category", "day_of_week"}

def run_sql_to_ctr_predictions(sql_query, spark_df):
    try:
        # Register SQL temp view
        spark_df.createOrReplaceTempView("ctr_df")
        input_df = spark.sql(sql_query).toPandas()
        user_requested_cols = input_df.columns.tolist()

        if input_df.empty:
            input_df = pd.DataFrame([DEFAULT_ROW])
            user_requested_cols = list(DEFAULT_ROW.keys())

        # Clean column names
        input_df.columns = [col.strip() for col in input_df.columns]
        user_requested_cols = [col.strip() for col in user_requested_cols]

        # Drop duplicate predicted_ctr variants
        cols_to_drop = [col for col in input_df.columns if col.lower() == "predicted_ctr"]
        input_df = input_df.drop(columns=cols_to_drop, errors="ignore")
        user_requested_cols = [col for col in user_requested_cols if col.lower() != "predicted_ctr"]

        # ✅ CASE 1: Regular prediction (row-level or fillable rows)
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

        # ✅ CASE 2: Simulatable group fallback — e.g. by device_type, app_category, etc.
        simulatable_cols = [col for col in input_df.columns if col in SIMULATABLE_FEATURES]
        if len(simulatable_cols) == 1:
            sim_col = simulatable_cols[0]
            booster = lgb.Booster(model_file="/databricks/driver/ctr-predictor-module/model/ctr_model.txt")
            pred_results = []

            for _, row in input_df.iterrows():
                sim_row = DEFAULT_ROW.copy()
                sim_row[sim_col] = row[sim_col]
                features = create_features(sim_row)
                pred = booster.predict(features)[0]
                row["predicted_ctr"] = pred
                pred_results.append(row)

            return pd.DataFrame(pred_results)[user_requested_cols + ["predicted_ctr"]]

        # ✅ CASE 3: No valid features and no simulatable column — use global fallback
        features = create_features(DEFAULT_ROW)
        booster = lgb.Booster(model_file="/databricks/driver/ctr-predictor-module/model/ctr_model.txt")
        fallback_pred = booster.predict(features)[0]
        input_df["predicted_ctr"] = fallback_pred
        return input_df[user_requested_cols + ["predicted_ctr"]]

    except Exception as e:
        return pd.DataFrame({"error": [f" Error: {str(e)}"]})
