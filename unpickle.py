import pickle

# Load the pickled DataFrame
with open('dataset/labels.pkl', 'rb') as f:
    data = pickle.load(f)

data.to_csv('all_labels.tsv', sep='\t', index=False)   # Space-separated format with index
