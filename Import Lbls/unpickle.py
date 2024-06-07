'''
Download 'cropped_images.zip' from:
https://ieee-dataport.org/open-access/deepdarts-dataset
and use this script to unpickle it, and save it to a tab seperated format
'''

import pickle

# Load the pickled DataFrame
with open('dataset/labels.pkl', 'rb') as f:
    data = pickle.load(f)

data.to_csv('all_labels.tsv', sep='\t', index=False)   # tab-separated format with index
