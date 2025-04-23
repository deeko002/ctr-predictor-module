from main_pipeline import run_sql_to_ctr_predictions
from pyspark.sql import SparkSession

# Create dummy Spark session
spark = SparkSession.builder.getOrCreate()

# Sample SQL and DataFrame
sql_query = """
SELECT hour, banner_pos, device_type, device_model
FROM avazu_df
LIMIT 5
"""

