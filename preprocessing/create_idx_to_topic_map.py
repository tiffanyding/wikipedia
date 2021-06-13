import numpy as np
import os
import pandas as pd
import pickle
import scipy.sparse as ss
import time

from sklearn.preprocessing import normalize

st = time.time()

year = '2018'

with open(f'data/wikilinkgraph/title_to_idx_{year}.pkl', 'rb') as f:
    title_to_idx = pickle.load(f)

# Convert page list csv from https://figshare.com/articles/dataset/Topics_for_each_Wikipedia_Article_across_Languages/12127434
# into dict
df = pd.read_csv('data/page_list.csv.gz', 
                 compression='gzip',
                #  nrows=1000, # COMMENT OUT WHEN NOT DEBUGGING
                 usecols=['page_title', 'topic', 'probability', 'wiki_db'])
df = df[df['wiki_db']=='enwiki']

print('Number of rows:', len(df))

# Consider topic granularity to 2nd level (e.g.  Culture.Sports)
df['level2_topic'] = df['topic'].apply(lambda x: ".".join(x.split('.')[:2]))
print(df.head(20))

# Create a unique index for each level 2 topic
level2_topics = sorted(df['level2_topic'].unique())
level2topic_to_idx = {n:i for i,n in enumerate(level2_topics)}
idx_to_level2topic = {n:i for i,n in level2topic_to_idx.items()}

print('Topic map:')
print(level2topic_to_idx)

# Map titles -> index and topic -> index
df['page_idx'] = df['page_title'].apply(lambda x: title_to_idx.get(str(x).replace('_', ' '), -1))
df['level2_topic_idx'] = df['level2_topic'].apply(lambda x: level2topic_to_idx[x])

# Filter out pages not in title_to_idx (corresponds to pages not in WikiLinkGraph)
df = df[df['page_idx'] != -1]

print(df.head(20))

# Create sparse matrix
rows = df['page_idx']  # Not a copy, just a reference.
cols = df['level2_topic_idx']
data = df['probability']
matrix = ss.coo_matrix((data, (rows, cols)))
matrix = ss.csc_matrix(matrix)

print('matrix shape:', matrix.shape)

# Create normalized matrix where each row sums to 1
normalized_matrix = normalize(matrix, norm='l1', axis=1)

# Save matrix
matrix_path = f'data/topic_matrix_{year}.pkl'
with open(matrix_path, 'wb') as f:
    pickle.dump(matrix, f)
print(f'Saved title and topic matrix to {matrix_path}')

# Save normalized matrix
path = f'data/normalized_topic_matrix_{year}.pkl'
with open(path, 'wb') as f:
    pickle.dump(normalized_matrix, f)
print(f'Saved normalized title and topic matrix to {path}')

# Save map from topic to index
level2topic_to_idx_path = f'data/level2topic_to_idx_{year}.pkl'
with open(level2topic_to_idx_path, 'wb') as f:
    pickle.dump(level2topic_to_idx, f)
print(f'Saved map from topic to index to {level2topic_to_idx_path}')




print(f'Time taken: {(time.time() - st) / 60:.2f} min')