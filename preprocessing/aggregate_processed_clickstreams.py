import glob, os
import numpy as np
import pandas as pd
import pathlib
import pickle
import scipy.sparse as ss
import time

from sklearn.preprocessing import normalize

# import sys; sys.path.append('..')

# from utils import load_pickle_file, save_to_pickle # For some reason, this is not working
def load_pickle_file(path):
    with open(path, 'rb') as f:
        loaded_file = pickle.load(f)

    return loaded_file

def save_to_pickle(obj, save_to, description=''):
    with open(save_to, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Saved {description} to {save_to}')


st = time.time()

# Folder with monthly cleaned clickstream data
load_folder = "data/clickstream/cleaned"

# year should match the files in load_folder
year = '2018'

# Folder to save pi, B, and C to
save_folder = "data/clickstream/final"

# -------------------------

# Make save folder if necessary
pathlib.Path(save_folder).mkdir(exist_ok=True)

## 1) Load data

# Get list of file names 
files = []
_, _, files_in_dir = next(os.walk(load_folder))
for f in files_in_dir:
    if f.startswith('20') and f.endswith('matrix.pkl'):
        files.append(f)
files = sorted(files)
print('NOTE: Please check that the following list contains files for all months that you selected! ' 
        'If there are missing months, rerun preprocessing/process_all_clickstream_files.sh')
print('File list:', files)

# Add matrices together
aggregated_matrix = load_pickle_file(os.path.join(load_folder, files[0]))
print('aggregated_matrix shape:', aggregated_matrix.shape)
for i in range(1,len(files)):
    m = load_pickle_file(os.path.join(load_folder, files[i]))
    # print('m', m.shape)
    aggregated_matrix = aggregated_matrix + m

## 2) Process data and save

# Compute pi (probablity of starting at each page)
pi = aggregated_matrix[-1,:] / aggregated_matrix[-1,:].sum()

save_to = os.path.join(save_folder, f'p_{year}.pkl')
save_to_pickle(pi, save_to, description='pi (probability of starting at each page)')

# Compute B (probability transition matrix that assumes surfer never exits)
clicks = aggregated_matrix[:-1,:]
B = normalize(clicks, norm='l1', axis=1) # normalize each row to sum to 1 (See https://stackoverflow.com/questions/12305021/efficient-way-to-normalize-a-scipy-sparse-matrix)

print('clicks row 10', clicks[10,:])
print('B row 10', B[10,:])

save_to = os.path.join(save_folder, f'B_{year}.pkl')
save_to_pickle(pi, save_to, 
        description='B (probability transition matrix that assumes surfer never exits)')

# Compute C (probability transition matrix that includes absorbing exit state)
# Row num_pages corresponds to exit state (indexing from 0)
num_pages = aggregated_matrix.shape[0]
total_page_views = aggregated_matrix.sum(axis=0).T
num_clicks_out = aggregated_matrix[:-1,:].sum(axis=1)
num_exit = num_clicks_out - total_page_views
# tmp = ss.csc_matrix((num_pages + 1, num_pages + 1))
tmp = aggregated_matrix[:-1,:]
tmp = ss.hstack([tmp, num_exit])
last_row = ss.csc_matrix(([1], ([0], [num_pages-1])), shape=(1, num_pages))
tmp = ss.vstack([tmp, last_row])


C = normalize(tmp, norm='l1', axis=1)
print('tmp row 5', tmp[5])
print('C row 5', C[5,:])

save_to = os.path.join(save_folder, f'C_{year}.pkl')
save_to_pickle(pi, save_to, 
        description='C (probability transition matrix that includes absorbing exit state)')

# Sanity check
print(B[10,:].sum())
print(C[10,:].sum())
print(B[1000,:].sum())
print(C[1000,:].sum())

print(f'Time taken: {(time.time() - st) / 60:.2f} min')