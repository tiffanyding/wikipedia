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
    if arr1.ndim > 1 and arr1.shape[1] != 1:
        if ss.issparse(arr1):
            arr1 = arr1.toarray().T
        else:
            arr1 = np.array(arr1).T
    if arr2.ndim > 1 and arr2.shape[1] != 1:
        if ss.issparse(arr2):
            arr2 = arr2.toarray().T
        else:
            arr2 = np.array(arr2).T

    arr1 = np.squeeze(arr1)
    arr2 = np.squeeze(arr2)

    X = np.stack((arr1, arr2), axis=1).T
    return np.corrcoef(X)[0][1]

year = '2018'
pageviews = np.load(f'results/pageviews_{year}.npy')
pageviews_internal_and_external = load_pickle_file(f'results/pageviews_internal_and_external_{year}.pkl')
pageviews_internal = load_pickle_file(f'results/pageviews_internal_{year}.pkl')

pi = load_pickle_file(f'data/clickstream/final/pi_{year}.pkl')
pageranks = load_pickle_file(f'results/pagerank_{year}.pkl')
weighted_pageranks = load_pickle_file(f'results/weighted_pagerank_{year}.pkl')
rw1_01 = load_pickle_file(f'results/rw1_p=0.1_{year}.pkl')
rw1_03 = load_pickle_file(f'results/rw1_p=0.3_{year}.pkl')
rw1_08 = load_pickle_file(f'results/rw1_p=0.8_{year}.pkl')
rw2 = load_pickle_file(f'results/rw2_{year}.pkl')

# print('pageviews', pageviews.shape)
# print('pageranks', pageranks.shape)
# print('rw2', rw2.shape)
# print('pi', pi.shape)

# pageranks = np.array(pageranks).T

# print('pageranks', pageranks.shape)
# print('pageviews', pageviews.shape)

corr0 = compute_corrcoeff(pageviews, pi)
print(f'Correlation between page views and pi: {corr0:.5f}')

corr1 = compute_corrcoeff(pageviews, pageranks)
print(f'Correlation between page views and PageRank: {corr1:.5f}')

corr2 = compute_corrcoeff(pageviews, weighted_pageranks)
print(f'Correlation between page views and weighted PageRank: {corr2:.5f}')

corr = compute_corrcoeff(pageviews, rw1_08)
print(f'Correlation between page views and Model 1: {corr:.5f}')

corr = compute_corrcoeff(pageviews, rw2)
print(f'Correlation between page views and Model 2: {corr:.5f}')


# Now break down into different types of page views

corr = compute_corrcoeff(pageviews_internal, pi)
print(f'Correlation between internal page views and pi: {corr:.5f}')

corr = compute_corrcoeff(pageviews_internal, pageranks)
print(f'Correlation between internal page views and PageRank: {corr:.5f}')

corr = compute_corrcoeff(pageviews_internal, weighted_pageranks)
print(f'Correlation between internal page views and weighted PageRank: {corr:.5f}')

corr = compute_corrcoeff(pageviews_internal, rw1_01)
print(f'Correlation between internal page views and Model 1 (p=0.1): {corr:.5f}')

corr = compute_corrcoeff(pageviews_internal, rw1_03)
print(f'Correlation between internal page views and Model 1 (p=0.3): {corr:.5f}')

corr = compute_corrcoeff(pageviews_internal, rw1_08)
print(f'Correlation between internal page views and Model 1 (p=0.8): {corr:.5f}')

corr = compute_corrcoeff(pageviews_internal, rw2)
print(f'Correlation between internal page views and Model 2: {corr:.5f}')



print(f'Time taken: {(time.time() - st) / 60:.2f} min')