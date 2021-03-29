import os
import pandas as pd
import pickle
import numpy as np
import scipy.sparse as ss
import time



st = time.time()

folder = 'data/wikilinkgraph/'

# year = '2002' # Small dataset to debug with
year = '2018'

wiki_file = f'enwiki.wikilink_graph.{year}-03-01.csv.gz' 

path = os.path.join(folder, wiki_file)

def exclude_redirection_pages(df):
    '''Filter out redirection pages'''

    nodes_1 = set(df.page_title_from.unique())
    nodes_2 = set(df.page_title_to.unique())
    nodes = list(nodes_1.union(nodes_2))

    indegree = df.groupby('page_title_to').count()
    without_incoming = set(nodes).difference(set(indegree.index))
    outdegree = df.groupby('page_title_from').count()
    one_outdegree = set(outdegree[outdegree['page_title_to']==1].index)
    
    redirections = without_incoming.intersection(one_outdegree)
    print('Number of redirection pages:', len(redirections))
    nodes_clean = set(nodes).difference(redirections)
    print('Number of non-redirection pages:', len(nodes_clean))

    df_clean = df[(df['page_title_from'].isin(nodes_clean)) & (df['page_title_to'].isin(nodes_clean))]

    return df_clean


def convert_df_to_coo_matrix(df):
    '''Returns sparse matrix in coordinate format.'''
    
    nodes_1 = set(df_clean.page_title_from.unique())
    nodes_2 = set(df_clean.page_title_to.unique())
    nodes_clean = list(nodes_1.union(nodes_2))

    idx_to_title = {i:n for i,n in enumerate(nodes_clean)}
    title_to_idx = {n:i for i,n in idx_to_title.items()}
    df_clean['id_from'] = df_clean['page_title_from'].apply(lambda x: title_to_idx[x])
    df_clean['id_to'] = df_clean['page_title_to'].apply(lambda x: title_to_idx[x])

    rows = df['id_from']  # Not a copy, just a reference.
    cols = df['id_to']
    ones = np.ones(len(rows), np.float32)
    matrix = ss.coo_matrix((ones, (rows, cols)))
    return ss.csc_matrix(matrix), idx_to_title, title_to_idx

## ---- PROCESS CSV ----
df = pd.read_csv(path, 
                 sep='\t', 
                 compression='gzip', 
                 usecols=['page_title_from','page_title_to'])
df_clean = exclude_redirection_pages(df)
A, idx_to_title, title_to_idx = convert_df_to_coo_matrix(df_clean)

## ---- SAVE ----
idx_to_title_path = os.path.join(folder, f'idx_to_title_{year}.pkl')
title_to_idx_path = os.path.join(folder, f'title_to_idx_{year}.pkl')
A_path = os.path.join(folder, f'adjacency_matrix_{year}.pkl')

# Save adjacency matrix
with open(A_path, 'wb') as f:
    pickle.dump(A, f)
print(f'Saved adjacency matrix to {A_path}')

# Save map from adjacency matrix index to page title 
with open(idx_to_title_path, 'wb') as f:
    pickle.dump(idx_to_title, f)
print(f'Saved map from index to page title to {idx_to_title_path}')

# Save map from page title to adjacency matrix index
with open(title_to_idx_path, 'wb') as f:
    pickle.dump(title_to_idx, f)
print(f'Saved map from page title to index to {title_to_idx_path}')



print(f'Time taken: {(time.time() - st) / 60:.2f} min')