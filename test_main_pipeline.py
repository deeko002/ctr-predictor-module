# test_main_pipeline.py

from main_pipeline import run_sql_to_ctr_predictions

# Sample GPT-style SQL (to simulate actual usage)
sql_query = """
SELECT hour, banner_pos, device_type, device_model
FROM __THIS__
WHERE hour = 14102100
LIMIT 5
"""

# Replaces GPT-style token with DuckDB table name
sql_query = sql_query.replace("__THIS__", "ctr_df")

# Exported test function
def test_ctr_prediction(df):
    result = run_sql_to_ctr_predictions(sql_query, df)
    print("Predicted CTRs:")
    print(result)
    return result
