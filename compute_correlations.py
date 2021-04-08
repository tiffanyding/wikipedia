import argparse
import numpy as np
import os
import pandas as pd
import pickle
import scipy.sparse as ss
import time

st = time.time()

def load_pickle_file(path):
    with open(path, 'rb') as f:
        loaded_file = pickle.load(f)

    return loaded_file

def compute_corrcoeff(arr1, arr2):
    # if arr1.shape[1] 

    X = np.stack((np.squeeze(arr1), np.squeeze(arr2)), axis=1).T
    return np.corrcoef(X)[0][1]

year = '2018'
pageviews = np.load(f'results/pageviews_{year}.npy')

pageranks = load_pickle_file(f'results/pagerank_{year}.pkl')
rw2 = load_pickle_file(f'results/pagerank_{year}.pkl')

print('pageviews', pageviews.shape)
print('pageranks', pageranks.shape)
print('rw2', rw2.shape)

pageranks = np.array(pageranks).T

# print('pageranks', pageranks.shape)
# print('pageviews', pageviews.shape)

corr1 = compute_corrcoeff(pageviews, pageranks)
print(f'Correlation between page views and PageRank: {corr1:.3f}')

corr2 = compute_corrcoeff(pageviews, rw2)
print(f'Correlation between Model 2 and PageRank: {corr2:.3f}')



print(f'Time taken: {(time.time() - st) / 60:.2f} min')