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
    print('arr2 initially', arr2.shape, type(arr2), np.array(arr2).shape, np.array(arr2).T.shape)
    if arr1.ndim > 1 and arr1.shape[1] != 1:
        arr1 = arr1.T
    if arr2.ndim > 1 and arr2.shape[1] != 1:
        arr2 = arr2.toarray().T

    print('arr2 here', arr2.shape)
    arr1 = np.squeeze(arr1)
    arr2 = np.squeeze(arr2)

    print(arr1.shape, arr2.shape)
    # X = np.stack((np.squeeze(arr1), np.squeeze(arr2)), axis=1).T
    X = np.stack((arr1, arr2), axis=1).T
    print(X.shape)
    return np.corrcoef(X)[0][1]

year = '2018'
pageviews = np.load(f'results/pageviews_{year}.npy')

pi = load_pickle_file(f'data/clickstream/final/pi_{year}.pkl')
pageranks = load_pickle_file(f'results/pagerank_{year}.pkl')
rw2 = load_pickle_file(f'results/rw2_{year}.pkl')

print('pageviews', pageviews.shape)
print('pageranks', pageranks.shape)
print('rw2', rw2.shape)
print('pi', pi.shape)

# pageranks = np.array(pageranks).T

# print('pageranks', pageranks.shape)
# print('pageviews', pageviews.shape)

corr0 = compute_corrcoeff(pageviews, pi)
print(f'Correlation between page views and pi: {corr0:.5f}')

corr1 = compute_corrcoeff(pageviews, pageranks)
print(f'Correlation between page views and PageRank: {corr1:.5f}')

corr2 = compute_corrcoeff(pageviews, rw2)
print(f'Correlation between page views and Model 2: {corr2:.5f}')



print(f'Time taken: {(time.time() - st) / 60:.2f} min')