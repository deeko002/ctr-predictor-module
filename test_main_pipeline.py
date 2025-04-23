from main_pipeline import run_sql_to_ctr_predictions

# Sample SQL query like from NLP
sql = """
SELECT hour, COUNT(*) AS total_clicks, COUNT(*) / COUNT(*) AS ctr
FROM avazu_df
GROUP BY hour
ORDER BY hour
"""

# Run it
results = run_sql_to_ctr_predictions(sql)

# Print predictions
print("Predicted CTRs from SQL Query:")
print(results)
