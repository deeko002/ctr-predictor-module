from main_pipeline import run_sql_to_ctr_predictions

# Sample SQL query like from NLP
sql = """
SELECT hour, banner_pos, device_type, device_model
FROM avazu_df
LIMIT 5
"""

# Run it
results = run_sql_to_ctr_predictions(sql)

# Print predictions
print("Predicted CTRs from SQL Query:")
print(results)
