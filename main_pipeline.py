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

def run_sql_to_ctr_predictions(sql_query, spark_df):
    try:
        # Register temp view for SQL querying
        spark_df.createOrReplaceTempView("ctr_df")

        # Run query and convert to Pandas
        input_df = spark.sql(sql_query).toPandas()

        # Fallback to default if SQL returns no rows
        if input_df.empty:
            input_df = pd.DataFrame([DEFAULT_ROW])

        # Fill missing required columns
        for col in REQUIRED_COLUMNS:
            if col not in input_df.columns:
                input_df[col] = DEFAULT_ROW[col]
            else:
                input_df[col] = input_df[col].fillna(DEFAULT_ROW[col])

        # Step 2: Feature Engineering
        features_df = create_features(input_df)

        # Step 3: Load Model
        booster = lgb.Booster(model_file="/databricks/driver/ctr-predictor-module/model/ctr_model.txt")


        # Step 4: Predict
        preds = booster.predict(features_df)
        input_df["predicted_ctr"] = preds

        # Step 5: Group by device_type (or return all predictions if no group)
        if "device_type" in input_df.columns:
            result_df = input_df.groupby("device_type")["predicted_ctr"].mean().reset_index()
        else:
            result_df = input_df[["predicted_ctr"]]

        return result_df

    except Exception as e:
        return pd.DataFrame({"error": [f"Error: {str(e)}"]})
