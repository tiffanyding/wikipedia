import argparse
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import scipy.sparse as ss
import time

st = time.time()

# Folder with clickstream data
clickstream_folder = 'data/clickstream'

year = '2002'

# Path to page title to index map
title_to_idx_path = f'data/wikilinkgraph/title_to_idx_{year}.pkl'


# Specify save locations
save_folder = 'results'

# Location to save pageviews csv to
save_csv_to = f'{save_folder}/pageviews_{year}.csv'

# Location to save pageviews np array to
save_array_to = f'{save_folder}/pageviews_{year}.npy'

# -------------------------------

# Make save folder if necessary
pathlib.Path(save_folder).mkdir(exist_ok=True)

# Read in title_to_idx map
with open(title_to_idx_path, 'rb') as f:
    title_to_idx = pickle.load(f)
print('title_to_idx keys sample:', list(title_to_idx.keys())[:10])

# Get list of clickstream file names 
files = []
_, _, files_in_dir = next(os.walk(clickstream_folder))
for f in files_in_dir:
    if f.endswith('tsv.gz'):
        files.append(f)
files = sorted(files)

counts = pd.DataFrame()
for zip_file in files:
    ## Can uncomment to test
    # df = pd.DataFrame({'curr': ['Anglo-Saxons', 'Neper', 'Alabama', 'Anglo-Saxons', 'bjrwgbbj'],
    #                     'n': [10, 20, 30, 100, 200]})

    # Read in files
    print('File: ', zip_file)
    df = pd.read_csv(os.path.join(clickstream_folder, zip_file), compression='gzip', sep='\t',
                # nrows=1000, # can uncomment to test
                names=['prev', 'curr', 'type', 'n'],
                usecols=['curr', 'n'])

    num_rows = len(df)
    print('Number of rows:', num_rows)

    ct = df.groupby('curr').sum()
    counts = counts.append(pd.DataFrame(ct).reset_index())

final_counts = pd.DataFrame(counts.groupby('curr').sum()).reset_index()

# Map page titles to index. Pages that did not appear in WikiLinkGraph
# are mapped to -1 and then removed
final_counts['page_idx'] = final_counts['curr'].apply(
                lambda x: title_to_idx.get(str(x).replace('_', ' '), -1))

# Remove counts for pages that did not appear in WikiLinkGraph
final_counts = final_counts[final_counts['page_idx']!=-1]

print(final_counts)

# Save csv
final_counts.to_csv(save_csv_to)
print(f'Saved pageviews csv to {save_csv_to}')

# Optionally load in csv
# final_counts = pd.read_csv(save_csv_to)

## Convert pageviews csv to np array and save
print('Converting csv to array...')
pageviews_arr = np.zeros((len(title_to_idx), 1))
for _, row in final_counts.iterrows():
    pageviews_arr[row['page_idx']] = row['n']

# Check number of pages with 0 views
print(f'{np.sum(pageviews_arr == 0)} out of {len(pageviews_arr)} pages have 0 views')
# Save
np.save(save_array_to, pageviews_arr)


print(f'Time taken: {(time.time() - st) / 60:.2f} min')


# Scratch
# df = pd.read_csv(path, compression='gzip', sep='\t',
#                 nrows=1000, # Can uncomment when debugging
#                 names=['prev', 'curr', 'type', 'n'],
#                 usecols=['curr', 'n'])