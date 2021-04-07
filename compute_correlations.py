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
    return np.corrcoef(arr1, arr2)[0][1]

year = '2018'
pageranks = load_pickle_file(f'results/pagerank_{year}.pkl')
pageviews = np.load(f'results/pageviews_{year}.npy')

pageranks = pageranks.T

print('pageranks', pageranks.shape)
print('pageviews', pageviews.shape)

corr1 = compute_corrcoeff(pageranks, pageviews)
print(f'Correlation between page views and PageRank: {corr1:.3f}')



print(f'Time taken: {(time.time() - st) / 60:.2f} min')