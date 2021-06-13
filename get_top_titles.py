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

def get_top_n(arr, idx_to_title=None, n=10):
    '''
    Gets indices of the n largest entries of arr and maps them to the corresponding page titles
    '''
    if idx_to_title is None:
        print('Using default idx_to_title for 2018')
        year = 2018
        idx_to_title = load_pickle_file(f'data/wikilinkgraph/idx_to_title_{year}.pkl')

    if ss.isspmatrix(arr):
        arr = arr.todense()
    arr = np.squeeze(np.array(arr))
    # print('arr', arr)
    max_n_idx = (-arr).argsort()[:n]
    # print(max_n_idx)
    titles = [idx_to_title[i] for i in max_n_idx]
    for i, title in enumerate(titles):
        print(f'{i+1}. {title}')
    print()
    return titles
    
if __name__ == '__main__':
    year = '2018'

    idx_to_title = load_pickle_file(f'data/wikilinkgraph/idx_to_title_{year}.pkl')

    pageviews = np.load(f'results/pageviews_{year}.npy')
    pageviews_internal_and_external = load_pickle_file(f'results/pageviews_internal_and_external_{year}.pkl')
    pageviews_internal = load_pickle_file(f'results/pageviews_internal_{year}.pkl')

    pi = load_pickle_file(f'data/clickstream/final/pi_{year}.pkl')
    pageranks = load_pickle_file(f'results/pagerank_{year}.pkl')
    weighted_pageranks = load_pickle_file(f'results/weighted_pagerank_{year}.pkl')
    rw1_00001 = load_pickle_file(f'results/rw1_p=0.01_{year}.pkl')
    rw1_001 = load_pickle_file(f'results/rw1_p=0.01_{year}.pkl')
    rw1_005 = load_pickle_file(f'results/rw1_p=0.05_{year}.pkl')
    rw1_01 = load_pickle_file(f'results/rw1_p=0.1_{year}.pkl')
    rw1_03 = load_pickle_file(f'results/rw1_p=0.3_{year}.pkl')
    rw1_08 = load_pickle_file(f'results/rw1_p=0.8_{year}.pkl')
    rw2 = load_pickle_file(f'results/rw2_{year}.pkl')
    uniform_08 = load_pickle_file(f'results/uniform_p=0.8_{year}.pkl')

    n = 10
    print('Top pages by total page views:')
    get_top_n(pageviews, idx_to_title, n=n)
    print('Top pages by external+internal page views:')
    get_top_n(pageviews_internal_and_external, idx_to_title, n=n)
    print('Top pages by internal page views:')
    get_top_n(pageviews_internal, idx_to_title, n=n)
    print('Top pages by external page views (pi):')
    get_top_n(pi, idx_to_title, n=n)
    print('Top pages by PageRank:')
    get_top_n(pageranks, idx_to_title, n=n)
    print('Top pages by weighted PageRank:')
    get_top_n(weighted_pageranks, idx_to_title, n=n)
    print('Top pages by random walk (Model 1, p=0.3):')
    get_top_n(rw1_03, idx_to_title, n=n)
    print('Top pages by random walk (Model 1, p=0.8):')
    get_top_n(rw1_08, idx_to_title, n=n)
    print('Top pages by random walk (Model 2):')
    get_top_n(rw2, idx_to_title, n=n)
    print('Top pages by uniform random walk (p=0.8):')
    get_top_n(uniform_08, idx_to_title, n=n)

    print(f'Time taken: {(time.time() - st) / 60:.2f} min')