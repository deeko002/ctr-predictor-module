import pandas as pd
import lightgbm as lgb
from feature_creator import create_features

REQUIRED_COLUMNS = ["hour", "banner_pos", "device_type", "device_model", "site_id", "app_id"]
SIMULATABLE_FEATURES = {
    "device_type", "app_category", "day_of_week", "hour_of_day", "banner_pos"
}

def run_sql_to_ctr_predictions(sql_query, spark_df, user_question: str = ""):
    try:
        # Step 0: Setup
        spark_df.createOrReplaceTempView("ctr_df")
        input_df = spark.sql(sql_query).toPandas()
        user_requested_cols = input_df.columns.tolist()

        # Step 1: Normalize column names
        input_df.columns = [col.strip() for col in input_df.columns]
        user_requested_cols = [col.strip() for col in user_requested_cols]

        # Step 2: Remove any predicted_ctr column from SQL
        input_df = input_df.drop(columns=[col for col in input_df.columns if col.lower() == "predicted_ctr"], errors="ignore")
        user_requested_cols = [col for col in user_requested_cols if col.lower() != "predicted_ctr"]

        # Step 3: Check if prediction was actually requested
        predict_mode = any(kw in user_question.lower() for kw in ["predict", "predicted", "likely", "expected"])
        if not predict_mode:
            return input_df  # 🔕 Descriptive query — skip prediction

        # Step 4: Handle empty SQL output
        if input_df.empty:
            input_df = pd.DataFrame([{col: None for col in REQUIRED_COLUMNS}])
            user_requested_cols = list(input_df.columns)

        # Step 5: CASE 1 — Row-level prediction
        if any(col in input_df.columns for col in REQUIRED_COLUMNS):
            for col in REQUIRED_COLUMNS:
                if col not in input_df.columns:
                    input_df[col] = spark_df.select(col).dropna().limit(1).toPandas().iloc[0, 0]
                else:
                    input_df[col] = input_df[col].fillna(spark_df.select(col).dropna().limit(1).toPandas().iloc[0, 0])

            features_df = create_features(input_df)
            booster = lgb.Booster(model_file="/databricks/driver/ctr-predictor-module/model/ctr_model.txt")
            preds = booster.predict(features_df)
            input_df["predicted_ctr"] = preds
            return input_df[user_requested_cols + ["predicted_ctr"]]

        # Step 6: CASE 2 — Group-wise simulation fallback
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

            result_df = pd.DataFrame(prediction_rows)
            return result_df[user_requested_cols + ["predicted_ctr"]]

        # Step 7: CASE 3 — Nothing usable, fallback to smart row
        smart_row = spark_df.toPandas().mode(numeric_only=False).iloc[0].to_dict()
        sim_df = pd.DataFrame([smart_row])
        features_df = create_features(sim_df)
        booster = lgb.Booster(model_file="/databricks/driver/ctr-predictor-module/model/ctr_model.txt")
        pred = booster.predict(features_df)[0]
        input_df["predicted_ctr"] = pred
        return input_df[user_requested_cols + ["predicted_ctr"]]

    except Exception as e:
        return pd.DataFrame({"error": [f" Error: {str(e)}"]})
