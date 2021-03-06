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

## 2) Compute pi, B, and C and save
# # print('type(aggregated_matrix))', type(aggregated_matrix))
# aggregated_matrix = ss.csr_matrix(aggregated_matrix)

# # Compute pi (probablity of starting at each page)
# pi = aggregated_matrix[-1,:] / aggregated_matrix[-1,:].sum()

# save_to = os.path.join(save_folder, f'pi_{year}.pkl')
# save_to_pickle(pi, save_to, description='pi (probability of starting at each page)')

# # Compute B (probability transition matrix that assumes surfer never exits)
# clicks = aggregated_matrix[:-1,:]
# # count number of rows with 0 outgoing clicks
# print(f'{(clicks.sum(axis=1) == 0).sum()} out of {clicks.shape[1]} pages have 0 outgoing clicks. '
#         'Adding self loops to these pages.' )
# diag_entries = np.squeeze(np.array(clicks.sum(axis=1) == 0)).astype(int)
# clicks = clicks + ss.diags(diag_entries) # Adding self loops to pages with 0 outgoing clicks
# print(f'{(clicks.sum(axis=1) == 0).sum()} out of {clicks.shape[1]} pages have 0 outgoing clicks after adding self loops')
# B = normalize(clicks, norm='l1', axis=1) # normalize each row to sum to 1 (See https://stackoverflow.com/questions/12305021/efficient-way-to-normalize-a-scipy-sparse-matrix)

# # print('clicks row 10', clicks[10,:])
# # print('B row 10', B[10,:])

# save_to = os.path.join(save_folder, f'B_{year}.pkl')
# save_to_pickle(B, save_to, 
#         description='B (probability transition matrix that assumes surfer never exits)')

# # Compute C (probability transition matrix that includes absorbing exit state)
# # Row num_pages corresponds to exit state (indexing from 0)
# num_pages = aggregated_matrix.shape[0]
# total_page_views = aggregated_matrix.sum(axis=0).T
# num_clicks_out = aggregated_matrix[:-1,:].sum(axis=1)
# num_exit = total_page_views - num_clicks_out
# tmp = aggregated_matrix[:-1,:]
# tmp = ss.hstack([tmp, num_exit])
# last_row = ss.csc_matrix(([1], ([0], [num_pages-1])), shape=(1, num_pages))
# tmp = ss.vstack([tmp, last_row])
# tmp = ss.csr_matrix(tmp)
# C = normalize(tmp, norm='l1', axis=1)

# save_to = os.path.join(save_folder, f'C_{year}.pkl')
# save_to_pickle(C, save_to, 
#         description='C (probability transition matrix that includes absorbing exit state)')

# Sanity checks (rows should sum to 1)
# print(B[10,:].sum())
# print(B[1000,:].sum())

# print(C[10,:].sum())
# print(C[1000,:].sum())
# print(C[-1,:].sum())

## 3) Compute page views
# Sum external and internal pageviews
pageviews_internal_and_external = aggregated_matrix.sum(axis=0)
save_to = f'results/pageviews_internal_and_external_{year}.pkl'
save_to_pickle(pageviews_internal_and_external, save_to, 
        description='pageviews (internal and external)')

# Sum internal pageviews only (clicks originating from other Wikipedia pages)
pageviews_internal = aggregated_matrix[:-1,:].sum(axis=0)
save_to = f'results/pageviews_internal_{year}.pkl'
save_to_pickle(pageviews_internal, save_to, 
        description='pageviews (internal)')

print(f'Total internal + external page views: {pageviews_internal_and_external.sum()}')
print(f'Total internal page views: {pageviews_internal.sum()} ({pageviews_internal.sum() / pageviews_internal_and_external.sum() * 100:.2f}%)')

print(f'Time taken: {(time.time() - st) / 60:.2f} min')