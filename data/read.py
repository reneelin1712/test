import numpy as np

# Load the CSV file
data = np.loadtxt('transit.csv', delimiter=',')

print('run')
# Save the data to a .npy file
np.save('transit.npy', data)