import pandas as pd

# Read the updated_path.csv file
df = pd.read_csv('updated_path.csv')

# Group the DataFrame by 'ori' and 'des'
grouped = df.groupby(['ori', 'des'])

# Initialize an empty list to store the extracted rows
extracted_rows = []

# Iterate over each group
for (ori, des), group in grouped:
    if len(group) > 1 and group['path'].nunique() > 1:
        # Find the row with the maximum number of primary roads
        max_primary_row = group.loc[group['primary'].idxmax()]
        
        # Find the row with the maximum number of secondary or tertiary roads
        max_secondary_tertiary_row = group.loc[group[['secondary', 'tertiary']].sum(axis=1).idxmax()]
        
        # Check if the paths are different
        if max_primary_row['path'] != max_secondary_tertiary_row['path']:
            # Append the extracted rows to the list
            extracted_rows.append(max_primary_row)
            extracted_rows.append(max_secondary_tertiary_row)

# Create a new DataFrame with the extracted rows
result_df = pd.DataFrame(extracted_rows)

# Save the result DataFrame to a new CSV file
result_df.to_csv('extracted_paths.csv', index=False)