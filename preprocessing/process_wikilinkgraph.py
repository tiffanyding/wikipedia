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

def exclude_redirection_pages_v0(df):
    '''Filters out pages with in-degree = 0 and out-degree = 1'''

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

def exclude_redirection_pages(df):
    '''
    Filters out pages not in https://figshare.com/articles/dataset/Topics_for_each_Wikipedia_Article_across_Languages/12127434.
    This function assumes that this file has been downloaded to data/page_list.csv.gz, which can be 
    done using scripts/download_page_list.sh
    '''
    d = pd.read_csv('data/page_list.csv.gz', 
                 compression='gzip',
                 usecols=['page_title', 'wiki_db'])

    # Filter for enwiki only
    d = d[d['wiki_db'] == 'enwiki']

    # Convert underscores in page titles to spaces
    print('Converting underscores in page titles to spaces...')
    d['page_title'] = d['page_title'].apply(lambda x: str(x).replace('_', ' '))

    # print('d:', d.head(30))
    
    nodes_clean = set(d['page_title'])
    print('Size of enwiki in April 15 2020:', len(nodes_clean))

    idx1 = (df['page_title_from'].isin(nodes_clean))
    idx2 = (df['page_title_to'].isin(nodes_clean))
    df_clean = df[idx1 & idx2]
    print(f'Excluded {len(df) - len(df_clean)} out of {len(df)} edges connected to pages not present in https://figshare.com/articles/dataset/Topics_for_each_Wikipedia_Article_across_Languages/12127434')
    print('Excluded titles:')
    print(set(df['page_title_from'][~idx1]))
    print(set(df['page_title_to'][~idx2]))

    # print('df_clean:', df_clean.head(30))
    return df_clean


def convert_df_to_coo_matrix(df):
    '''Returns sparse matrix in coordinate format.'''
    
    nodes_1 = set(df_clean.page_title_from.unique())
    nodes_2 = set(df_clean.page_title_to.unique())
    nodes_clean = list(nodes_1.union(nodes_2))

    print('Number of non-redirection pages:', len(nodes_clean))

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
                #  nrows=1000, # !!! Comment out when not debugging
                 compression='gzip', 
                 usecols=['page_title_from','page_title_to'])
print(df.head())
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


# d = pd.read_csv('data/page_list.csv.gz', 
#                  nrows=10,
#                  compression='gzip') 
# print(d.columns)           


print(f'Time taken: {(time.time() - st) / 60:.2f} min')