import pandas as pd

# Read the path.csv file
path_df = pd.read_csv('path.csv')

# Read the extracted_paths.csv file
extracted_paths_df = pd.read_csv('extracted_paths.csv')

# Create a dictionary to store the labels for each (ori, des) pair
labels = {}

# Iterate over the rows in the extracted_paths_df
for _, row in extracted_paths_df.iterrows():
    ori = row['ori']
    des = row['des']
    path = row['path']
    primary_count = row['primary']
    
    if (ori, des) not in labels:
        labels[(ori, des)] = {}
    
    if primary_count > labels[(ori, des)].get('primary_count', 0):
        labels[(ori, des)]['primary'] = path
        labels[(ori, des)]['primary_count'] = primary_count
    else:
        labels[(ori, des)]['secondary'] = path

# Function to assign labels to each path
def assign_label(row):
    ori = row['ori']
    des = row['des']
    path = row['path']
    
    if (ori, des) in labels:
        if path == labels[(ori, des)].get('primary'):
            return 'primary'
        elif path == labels[(ori, des)].get('secondary'):
            return 'secondary'
    
    return 'unknown'

# Apply the assign_label function to each row in the path_df
path_df['label'] = path_df.apply(assign_label, axis=1)

# Save the updated path_df to a new CSV file
path_df.to_csv('labeled_path.csv', index=False)