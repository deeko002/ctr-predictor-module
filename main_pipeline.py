import pandas as pd
import lightgbm as lgb
from feature_creator import create_features

# Columns required for CTR prediction
REQUIRED_COLUMNS = ["hour", "banner_pos", "device_type", "device_model"]

# Default values for fallback when features are missing in SQL output
DEFAULT_ROW = {
    "hour": 1300,
    "banner_pos": 0,
    "device_type": 1,
    "device_model": "model_abc"
}

def run_sql_to_ctr_predictions(sql_query, spark_df):
    try:
        # Step 1: Register the Spark DataFrame as a SQL table
        spark_df.createOrReplaceTempView("ctr_df")

        # Step 2: Run the input SQL query
        input_df = spark.sql(sql_query).toPandas()

        # Step 3: Fallback to default row if SQL result is empty
        if input_df.empty:
            input_df = pd.DataFrame([DEFAULT_ROW])

        # Step 4: Ensure all required columns exist
        for col in REQUIRED_COLUMNS:
            if col not in input_df.columns:
                input_df[col] = DEFAULT_ROW[col]
            else:
                input_df[col] = input_df[col].fillna(DEFAULT_ROW[col])

        # Step 5: Feature engineering
        features_df = create_features(input_df)

        # Step 6: Load LightGBM model
        booster = lgb.Booster(model_file="/databricks/driver/ctr-predictor-module/model/ctr_model.txt")

        # Step 7: Predict CTR
        preds = booster.predict(features_df)
        input_df["predicted_ctr"] = preds

        # Step 8: Return full result row-wise with prediction
        result_df = input_df[REQUIRED_COLUMNS + ["predicted_ctr"]]

        return result_df

    except Exception as e:
        return pd.DataFrame({"error": [f" Error: {str(e)}"]})
