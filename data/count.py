import pandas as pd

# Read the path.csv file
path_df = pd.read_csv('path.csv')

# Read the edge.txt file
edge_df = pd.read_csv('edge.txt')

# Create a dictionary to map n_id to highway type
n_id_to_highway = dict(zip(edge_df['n_id'], edge_df['highway']))

# Function to count highway types for each path
def count_highway_types(path):
    highway_types = ['primary', 'secondary', 'tertiary', 'residential', 'unclassified']
    counts = dict.fromkeys(highway_types, 0)
    
    for n_id in path.split('_'):
        highway = n_id_to_highway.get(int(n_id), 'unclassified')
        if highway not in highway_types:
            highway = 'unclassified'
        counts[highway] += 1
    
    return pd.Series(counts)

# Apply the count_highway_types function to each path
highway_counts = path_df['path'].apply(count_highway_types)

# Concatenate the highway counts with the original DataFrame
updated_path_df = pd.concat([path_df, highway_counts], axis=1)

# Save the updated DataFrame to a new CSV file
updated_path_df.to_csv('updated_path.csv', index=False)