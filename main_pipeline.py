import pandas as pd
import lightgbm as lgb
from feature_creator import create_features

REQUIRED_COLUMNS = ["hour", "banner_pos", "device_type", "device_model"]
DEFAULT_ROW = {
    "hour": 1300,
    "banner_pos": 0,
    "device_type": 1,
    "device_model": "model_abc"
}
SIMULATABLE_FEATURES = {"device_type", "device_model", "day_of_week", "hour_of_day", "app_category"}

booster = None  # For lazy loading

def load_model():
    global booster
    if booster is None:
        booster = lgb.Booster(model_file="model/ctr_model.txt")

def run_sql_to_ctr_predictions(sql_query, spark_df, user_question: str = ""):
    try:
        spark_df.createOrReplaceTempView("ctr_df")
        input_df = spark_df.sql_ctx.sql(sql_query).toPandas()
        user_requested_cols = list(input_df.columns)

        input_df = input_df.drop(columns=[col for col in input_df.columns if col.lower() == "predicted_ctr"], errors="ignore")
        user_requested_cols = [col for col in user_requested_cols if col.lower() != "predicted_ctr"]

        predict_mode = any(kw in user_question.lower() for kw in ["predict", "predicted", "likely", "expected"])
        if not predict_mode:
            return input_df

        if input_df.empty:
            input_df = pd.DataFrame([DEFAULT_ROW])
            user_requested_cols = list(input_df.columns)

        for col in REQUIRED_COLUMNS:
            if col not in input_df.columns:
                input_df[col] = DEFAULT_ROW[col]
            else:
                input_df[col] = input_df[col].fillna(DEFAULT_ROW[col])

        if all(col in input_df.columns for col in REQUIRED_COLUMNS):
            load_model()
            features_df = create_features(input_df)
            input_df["predicted_ctr"] = booster.predict(features_df)
            return input_df[user_requested_cols + ["predicted_ctr"]]

        sim_cols = [col for col in input_df.columns if col in SIMULATABLE_FEATURES]
        if len(sim_cols) == 1:
            sim_col = sim_cols[0]
            smart_row = spark_df.toPandas().mode(numeric_only=False).iloc[0].to_dict()
            load_model()
            predicted_rows = []
            for _, row in input_df.iterrows():
                sim_input = smart_row.copy()
                sim_input[sim_col] = row[sim_col]
                features_df = create_features(pd.DataFrame([sim_input]))
                row_out = row.to_dict()
                row_out["predicted_ctr"] = booster.predict(features_df)[0]
                predicted_rows.append(row_out)
            return pd.DataFrame(predicted_rows)[user_requested_cols + ["predicted_ctr"]]

        smart_row = spark_df.toPandas().mode(numeric_only=False).iloc[0].to_dict()
        features_df = create_features(pd.DataFrame([smart_row]))
        load_model()
        pred = booster.predict(features_df)[0]
        input_df["predicted_ctr"] = pred
        return input_df[user_requested_cols + ["predicted_ctr"]]

    except Exception as e:
        return pd.DataFrame({"error": [f"Error: {str(e)}"]})
