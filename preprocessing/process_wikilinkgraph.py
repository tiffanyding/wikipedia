import os
import pandas as pd
import pickle
import numpy as np
import time

from scipy.sparse import coo_matrix

st = time.time()

folder = 'data/wikilinkgraph/'

year = '2002' # Small dataset to debug with
# year = '2018'

wiki_file = f'enwiki.wikilink_graph.{year}-03-01.csv.gz' 

path = os.path.join(folder, wiki_file)


# Map page title to adjacency matrix index
title_to_idx = {}
idx = 0

# Row and column list that is used to build coo_matrix
row_list = []
col_list = []

## ---- ITERATE THROUGH CSV ----
df_reader = pd.read_csv(path, compression='gzip', sep='\t', chunksize=10000)
for i, df in enumerate(df_reader):
    print(f'Processing chunk {i}')
    for i, row in df.iterrows():
        if row['page_title_from'] not in title_to_idx:
            title_to_idx[row['page_title_from']] = idx
            idx += 1
        if row['page_title_to'] not in title_to_idx:
            title_to_idx[row['page_title_to']] = idx
            idx += 1
        
        row_list.append(title_to_idx[row['page_title_from']])
        col_list.append(title_to_idx[row['page_title_to']])

# Create sparse matrix
data = np.ones((len(row_list),))
A = coo_matrix((data, (row_list, col_list)), shape=(idx+1,idx+1))


## ---- SAVE THINGS ----
map_path = os.path.join(folder, f'title_to_idx_map_{year}.pkl')
A_path = os.path.join(folder, f'adjacency_matrix_{year}.pkl')

# Save map from page title to adjacency matrix index
with open(map_path, 'wb') as f:
    pickle.dump(title_to_idx, f)
print(f'Saved map from page title to index to {map_path}')

# Save adjacency matrix
with open(A_path, 'wb') as f:
    pickle.dump(A, f)
print(f'Saved adjacency matrix to {A_path}')

print(f'Time taken: {(time.time() - st) / 60:.2f} min')