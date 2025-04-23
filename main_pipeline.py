import pandas as pd
import duckdb
from feature_creator import create_features
from predict_ctr import model
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# Required columns and defaults for fallback
REQUIRED_COLUMNS = ["hour", "banner_pos", "device_type", "device_model"]
DEFAULT_ROW = {
    "hour": 1300,
    "banner_pos": 0,
    "device_type": 1,
    "device_model": "model_abc"
}

# Load your sample CSV (instead of full train.csv)
file_path = "dbfs:/FileStore/ctr-predictor-module/train.gz"

total_data = 40428967

# load 10% data
avazu_df = spark.read.csv(file_path, header=True, inferSchema=True).limit(int(total_data * 0.1))
    


def run_sql_to_ctr_predictions(sql_query):
    """
    Takes a SQL query string, runs it on avazu_df,
    converts result to model features, returns CTR predictions
    """
    try:
        # Step 1: Run SQL on avazu_df
        raw_df = duckdb.query_df(avazu_df, "avazu_df", sql_query).to_df()

        if raw_df.empty:
            return "No results from query."

        # Step 2: Fill required columns with fallback values
        for col in REQUIRED_COLUMNS:
            if col not in raw_df.columns:
                raw_df[col] = DEFAULT_ROW[col]
            else:
                raw_df[col] = raw_df[col].fillna(DEFAULT_ROW[col])

        # Step 3: Convert to model-ready features
        features_df = create_features(raw_df)

        # Step 4: Predict CTR
        ctr_preds = model.predict(features_df)

        # Step 5: Append predictions to result
        raw_df["predicted_ctr"] = ctr_preds

        return raw_df[["hour", "banner_pos", "device_type", "device_model", "predicted_ctr"]]

    except Exception as e:
        return f"Error: {str(e)}"
