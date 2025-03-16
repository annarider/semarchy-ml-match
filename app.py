import pandas as pd
from splink import Linker, DuckDBAPI, block_on, SettingsCreator, splink_datasets
from splink.datasets import splink_datasets
from splink import block_on, SettingsCreator
import splink.comparison_library as cl
from sqlalchemy import create_engine
import duckdb

db_api=DuckDBAPI()

# Read from xDM
pg_conn_str = "postgresql+psycopg2://macfarlane_demo:macfarlane_demo@localhost:5432/postgres"
pg_engine = create_engine(pg_conn_str)
query = "SELECT * FROM mi_item"  # Modify this with your actual table
input_df = pd.read_sql(query, pg_engine)
# Create pk from b_pubid + b_sourceid
input_df["unique_id"] = input_df["b_pubid"] + ":" + input_df["b_sourceid"]

# Data cleanup

# Remove rows with missing values for internal_name and description
input_df = input_df.dropna(subset=["internal_name"])
input_df = input_df.dropna(subset=["description"])

# Remove noise word 'tape' from df
input_df['description'] = input_df['description'].str.replace(r'tape', '', case=False, regex=True).str.strip()
input_df['internal_name'] = input_df['internal_name'].str.replace(r'tape', '', case=False, regex=True).str.strip()

df_cleaned = duckdb.sql("""
    SELECT 
        unique_id as unique_id,
        LOWER(TRIM(name)) AS name,
        LOWER(TRIM(description)) AS description,
        LOWER(TRIM(internal_name)) AS internal_name,
        name_part1,
        name_part2,
        name_part3,
        name_part4,
        name_part5,
        item_cat1,
        item_cat2,
        item_cat3,
        CONCAT(name_part1, name_part2, name_part3, name_part4, name_part5) as five_name_parts                
    FROM input_df
""").to_df()  # Convert result back to Pandas dataframe

blocking_rules = [
    block_on("name_part2", "name_part4", "name_part5"),
]

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=blocking_rules,
    comparisons=[
    cl.ExactMatch("item_cat1"),
    cl.ExactMatch("item_cat2"),
    cl.ExactMatch("item_cat3"),
    cl.LevenshteinAtThresholds("name", [3, 4,5]),
    cl.JaroAtThresholds("description").configure(term_frequency_adjustments=True),
    cl.JaroAtThresholds("internal_name").configure(term_frequency_adjustments=True),
    ],
    retain_intermediate_calculation_columns = True,
    retain_matching_columns= True,
)

linker = Linker(df_cleaned, settings, db_api)

# Training
linker.training.estimate_probability_two_random_records_match(blocking_rules, 0.25)
linker.training.estimate_u_using_random_sampling(max_pairs=1e9)

session_item_cat1 = linker.training.estimate_parameters_using_expectation_maximisation(block_on("substr(item_cat1, 1, 1)"))
session_item_cat2 = linker.training.estimate_parameters_using_expectation_maximisation(block_on("substr(item_cat2, 1, 2)"))
session_item_cat3 = linker.training.estimate_parameters_using_expectation_maximisation(block_on("substr(item_cat3, 1, 2)"))
session_name = linker.training.estimate_parameters_using_expectation_maximisation(block_on("substr(name, 1, 5)"))
session_description = linker.training.estimate_parameters_using_expectation_maximisation(block_on("substr(description, 1, 5)"))
session_internal_name = linker.training.estimate_parameters_using_expectation_maximisation(block_on("substr(internal_name, 1, 5)"))

# Create pairwise predictions
df_predictions = linker.inference.predict(threshold_match_probability=0.97)
# df_predictions.as_pandas_dataframe().to_csv('splink_predictions.csv')

# # Create clusters from predictions
clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    df_predictions, threshold_match_probability=0.97
)

df_clusters = clusters.as_pandas_dataframe()
df_clusters[['b_pubid', 'b_sourceid']] = df_clusters['unique_id'].str.split(':', expand=True)
df_clusters = df_clusters[df_clusters['cluster_id'].map(df_clusters['cluster_id'].value_counts()) >= 2]

df_clusters.sort_values(by="cluster_id").to_csv('splink_clusters.csv')

# Write back to xDM
df_sql_insert = df_clusters[['cluster_id', 'b_pubid', 'b_sourceid', 'name', 'description', 'internal_name']]
df_sql_insert['b_loadid'] = 4
df_sql_insert['b_classname'] = 'Item'
df_sql_insert.to_sql('sd_item', con=pg_engine, if_exists='append', index=False)

print('fin!')