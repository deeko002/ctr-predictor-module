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
        # Register Spark table
        spark_df.createOrReplaceTempView("ctr_df")

        # Run SQL
        input_df = spark.sql(sql_query).toPandas()

        # Fallback: use default row if SQL result is empty
        if input_df.empty:
            input_df = pd.DataFrame([DEFAULT_ROW])

        # Ensure required features exist
        for col in REQUIRED_COLUMNS:
            if col not in input_df.columns:
                input_df[col] = DEFAULT_ROW[col]
            else:
                input_df[col] = input_df[col].fillna(DEFAULT_ROW[col])

        # Feature Engineering
        features_df = create_features(input_df)

        # Load trained model
        booster = lgb.Booster(model_file="C:/Users/dtaru/ctr-predictor-module/model/ctr_model.txt")

        # Predict
        preds = booster.predict(features_df)
        input_df["predicted_ctr"] = preds

        return input_df  # return full input + predictions

    except Exception as e:
        return pd.DataFrame({"error": [f" Error: {str(e)}"]})
