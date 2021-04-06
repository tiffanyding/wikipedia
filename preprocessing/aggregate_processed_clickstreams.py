import glob, os
import pandas as pd
import pickle

folder = "data/clickstream/cleaned"

# -------------------------

def load_pickle_file(path):
    with open(path, 'rb') as f:
        loaded_file = pickle.load(f)

    return loaded_file

# Get list of file names 
files = []
_, _, files_in_dir = next(os.walk(folder))
for f in files_in_dir:
    if f.startswith('20') and f.endswith('matrix.pkl'):
        files.append(f)
files = sorted(files)
print('NOTE: Please check that the following list contains files for all months that you selected! ' 
        'If there are missing months, rerun preprocessing/process_all_clickstream_files.sh')
print('File list:', files)

# Add matrices together
aggregated_matrix = load_pickle_file(os.path.join(folder, files[0]))
print('aggregated_matrix shape:', aggregated_matrix.shape)
for i in range(1,len(files)):
    m = load_pickle_file(os.path.join(folder, files[i]))
    # print('m', m.shape)
    aggregated_matrix = aggregated_matrix + m

# Save aggregated pages
save_to = os.path.join(folder, 'aggregated_matrix.pkl')
with open(save_to, 'wb') as f:
    pickle.dump(aggregated_matrix, f)
print(f'Saved aggregated matrix to {save_to}')
