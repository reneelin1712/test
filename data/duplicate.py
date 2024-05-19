import pandas as pd
import numpy as np

# Read the labeled paths CSV file
labeled_paths_df = pd.read_csv('labeled_path.csv')

# Create a new DataFrame with 100 rows by randomly duplicating rows
labeled_paths_100_df = labeled_paths_df.sample(n=100, replace=True, random_state=42)

# Reset the index of the new DataFrame
labeled_paths_100_df.reset_index(drop=True, inplace=True)

# Save the new DataFrame with 100 rows to a CSV file
labeled_paths_100_df.to_csv('labeled_paths_100.csv', index=False)

# Create a new DataFrame with 1000 rows by randomly duplicating rows
labeled_paths_1000_df = labeled_paths_df.sample(n=1000, replace=True, random_state=42)

# Reset the index of the new DataFrame
labeled_paths_1000_df.reset_index(drop=True, inplace=True)

# Save the new DataFrame with 1000 rows to a CSV file
labeled_paths_1000_df.to_csv('labeled_paths_1000.csv', index=False)