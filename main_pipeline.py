import pandas as pd
import lightgbm as lgb
from feature_creator import create_features

REQUIRED_COLUMNS = ["hour", "banner_pos", "device_type", "device_model", "site_id", "app_id"]
SIMULATABLE_FEATURES = {
    "device_type", "app_category", "day_of_week", "hour_of_day", "banner_pos"
}

def run_sql_to_ctr_predictions(sql_query, spark_df):
    try:
        # Register Spark SQL view
        spark_df.createOrReplaceTempView("ctr_df")
        input_df = spark.sql(sql_query).toPandas()
        user_requested_cols = input_df.columns.tolist()

        # Fallback if empty SQL result
        if input_df.empty:
            input_df = pd.DataFrame([{col: None for col in REQUIRED_COLUMNS}])
            user_requested_cols = list(input_df.columns)

        # Normalize columns
        input_df.columns = [col.strip() for col in input_df.columns]
        user_requested_cols = [col.strip() for col in user_requested_cols]

        # Drop duplicate predicted_ctr if it exists
        input_df = input_df.drop(columns=[col for col in input_df.columns if col.lower() == "predicted_ctr"], errors="ignore")
        user_requested_cols = [col for col in user_requested_cols if col.lower() != "predicted_ctr"]

        # ✅ CASE 0: Descriptive-only query — skip model
        if not any(col in input_df.columns for col in REQUIRED_COLUMNS) and \
           not any(col in input_df.columns for col in SIMULATABLE_FEATURES):
            return input_df

        # ✅ CASE 1: Row-wise prediction
        if any(col in input_df.columns for col in REQUIRED_COLUMNS):
            for col in REQUIRED_COLUMNS:
                if col not in input_df.columns:
                    input_df[col] = spark_df.select(col).dropna().limit(1).toPandas().iloc[0, 0]  # smart fill
                else:
                    input_df[col] = input_df[col].fillna(spark_df.select(col).dropna().limit(1).toPandas().iloc[0, 0])

            features_df = create_features(input_df)
            booster = lgb.Booster(model_file="/databricks/driver/ctr-predictor-module/model/ctr_model.txt")
            preds = booster.predict(features_df)
            input_df["predicted_ctr"] = preds
            return input_df[user_requested_cols + ["predicted_ctr"]]

        # ✅ CASE 2: Smart group-wise simulation
        simulatable_cols = [col for col in input_df.columns if col in SIMULATABLE_FEATURES]
        if len(simulatable_cols) == 1:
            sim_col = simulatable_cols[0]
            booster = lgb.Booster(model_file="/databricks/driver/ctr-predictor-module/model/ctr_model.txt")
            smart_row = spark_df.toPandas().mode(numeric_only=False).iloc[0].to_dict()

            prediction_rows = []
            for _, row in input_df.iterrows():
                sim_row = smart_row.copy()
                sim_row[sim_col] = row[sim_col]
                sim_df = pd.DataFrame([sim_row])
                features_df = create_features(sim_df)
                pred = booster.predict(features_df)[0]
                row_dict = row.to_dict()
                row_dict["predicted_ctr"] = pred
                prediction_rows.append(row_dict)

            return pd.DataFrame(prediction_rows)[user_requested_cols + ["predicted_ctr"]]

        # ✅ CASE 3: No usable features, use full-mode fallback
        smart_row = spark_df.toPandas().mode(numeric_only=False).iloc[0].to_dict()
        sim_df = pd.DataFrame([smart_row])
        features_df = create_features(sim_df)
        booster = lgb.Booster(model_file="/databricks/driver/ctr-predictor-module/model/ctr_model.txt")
        pred = booster.predict(features_df)[0]
        input_df["predicted_ctr"] = pred
        return input_df[user_requested_cols + ["predicted_ctr"]]

    except Exception as e:
        return pd.DataFrame({"error": [f" Error: {str(e)}"]})
