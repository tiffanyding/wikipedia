import numpy as np
import os
import pandas as pd
import pickle
import scipy.sparse as ss
import time

from scipy.stats import kendalltau

st = time.time()

def load_pickle_file(path):
    with open(path, 'rb') as f:
        loaded_file = pickle.load(f)

    return loaded_file

def compute_corrcoeff(arr1, arr2, use_kendalltau=True):
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

    # If use_kendalltau=True, use kendall tau correlation measure
    if use_kendalltau:
        corr, pvalue = kendalltau(arr1, arr2)
    # Else, comput Pearson correlation coeffcient:
    else:
        X = np.stack((arr1, arr2), axis=1).T
        corr = np.corrcoef(X)[0][1]
    return corr

def get_corrcoeff_matrix(arr_list, use_kendalltau=True, print_latex=False, latex_names=[]):
    print('Computing correlation matrix')
    num_arrs = len(arr_list)
    corrs = np.zeros((num_arrs, num_arrs))
    for i in range(num_arrs):
        for j in range(i): 
            corrs[i][j] = compute_corrcoeff(arr_list[i], arr_list[j], use_kendalltau=use_kendalltau)
            print(f'Corr {i} {j} = {corrs[i][j]:.3f}')

    if print_latex:
        print('&', ' & '.join(latex_names), '\\\\')
        for i in range(num_arrs):
            print(latex_names[i], '&', ' & '.join([f'{x:.3f}' if x != 0 else '' for x in corrs[i]]), '\\\\')

    return corrs

if __name__ == "__main__":
    year = '2018'
    pageviews = np.load(f'results/pageviews_{year}.npy')
    pageviews_internal_and_external = load_pickle_file(f'results/pageviews_internal_and_external_{year}.pkl')
    pageviews_internal = load_pickle_file(f'results/pageviews_internal_{year}.pkl')

    pi = load_pickle_file(f'data/clickstream/final/pi_{year}.pkl')
    pageranks = load_pickle_file(f'results/pagerank_{year}.pkl')
    weighted_pageranks = load_pickle_file(f'results/weighted_pagerank_{year}.pkl')

    uniform_08 = load_pickle_file(f'results/uniform_p=0.8_{year}.pkl')

    rw1_00001 = load_pickle_file(f'results/rw1_p=0.01_{year}.pkl')
    rw1_001 = load_pickle_file(f'results/rw1_p=0.01_{year}.pkl')
    rw1_005 = load_pickle_file(f'results/rw1_p=0.05_{year}.pkl')
    rw1_01 = load_pickle_file(f'results/rw1_p=0.1_{year}.pkl')
    rw1_03 = load_pickle_file(f'results/rw1_p=0.3_{year}.pkl')
    rw1_05 = load_pickle_file(f'results/rw1_p=0.5_{year}.pkl')
    rw1_08 = load_pickle_file(f'results/rw1_p=0.8_{year}.pkl')
    rw2 = load_pickle_file(f'results/rw2_{year}.pkl')

    exit_prob = load_pickle_file(f'data/clickstream/final/C_{year}.pkl')[:-1,-1].T

    # print('pageviews', pageviews.shape)
    # print('pageranks', pageranks.shape)
    # print('rw2', rw2.shape)
    # print('pi', pi.shape)

    # pageranks = np.array(pageranks).T

    # print('pageranks', pageranks.shape)
    # print('pageviews', pageviews.shape)

    print("Using default correlation coefficient (Kendall's tau')")

    # corr0 = compute_corrcoeff(pageviews, pi)
    # print(f'Correlation between page views and pi: {corr0:.5f}')

    # corr1 = compute_corrcoeff(pageviews, pageranks)
    # print(f'Correlation between page views and PageRank: {corr1:.5f}')

    # corr2 = compute_corrcoeff(pageviews, weighted_pageranks)
    # print(f'Correlation between page views and weighted PageRank: {corr2:.5f}')

    # corr = compute_corrcoeff(pageviews, rw1_01)
    # print(f'Correlation between page views and Model 1 (p=0.1): {corr:.5f}')

    # corr = compute_corrcoeff(pageviews, rw1_03)
    # print(f'Correlation between page views and Model 1 (p=0.3): {corr:.5f}')

    # corr = compute_corrcoeff(pageviews, rw1_05)
    # print(f'Correlation between page views and Model 1 (p=0.5): {corr:.5f}')

    # corr = compute_corrcoeff(pageviews, rw1_08)
    # print(f'Correlation between page views and Model 1 (p=0.8): {corr:.5f}')

    # corr = compute_corrcoeff(pageviews, rw2)
    # print(f'Correlation between page views and Model 2: {corr:.5f}')


    # # Now break down into different types of page views

    # corr = compute_corrcoeff(pageviews_internal, pi)
    # print(f'Correlation between internal page views and pi: {corr:.5f}')

    # corr = compute_corrcoeff(pageviews_internal, pageranks)
    # print(f'Correlation between internal page views and PageRank: {corr:.5f}')

    # corr = compute_corrcoeff(pageviews_internal, weighted_pageranks)
    # print(f'Correlation between internal page views and weighted PageRank: {corr:.5f}')

    # corr = compute_corrcoeff(pageviews_internal, rw1_00001)
    # print(f'Correlation between internal page views and Model 1 (p=0.0001): {corr:.5f}')

    # corr = compute_corrcoeff(pageviews_internal, rw1_001)
    # print(f'Correlation between internal page views and Model 1 (p=0.01): {corr:.5f}')

    # corr = compute_corrcoeff(pageviews_internal, rw1_005)
    # print(f'Correlation between internal page views and Model 1 (p=0.05): {corr:.5f}')

    # corr = compute_corrcoeff(pageviews_internal, rw1_01)
    # print(f'Correlation between internal page views and Model 1 (p=0.1): {corr:.5f}')

    # corr = compute_corrcoeff(pageviews_internal, rw1_03)
    # print(f'Correlation between internal page views and Model 1 (p=0.3): {corr:.5f}')

    # corr = compute_corrcoeff(pageviews_internal, rw1_05)
    # print(f'Correlation between internal page views and Model 1 (p=0.5): {corr:.5f}')

    # corr = compute_corrcoeff(pageviews_internal, rw1_08)
    # print(f'Correlation between internal page views and Model 1 (p=0.8): {corr:.5f}')

    # corr = compute_corrcoeff(pageviews_internal, rw2)
    # print(f'Correlation between internal page views and Model 2: {corr:.5f}')

    # Compute correlation matrix
    arr_list = [pageviews_internal_and_external, pi, pageviews_internal, exit_prob,
                pageranks,
                uniform_08,
                rw1_03, rw1_08, rw2]
    names = ['\\textbf{Total PV}', '\\textbf{External PV}', '\\textbf{Internal PV}', '\\textbf{Exit prob.}',
            '\\textbf{PR}',
            '\\textbf{Uniform}$^{(0.8)}$',
            '\\textbf{M1}$^{(0.3)}$', '\\textbf{M1}$^{(0.8)}$', '\\textbf{M2}'] # UPDATE!
    print('names:', names)
    get_corrcoeff_matrix(arr_list, use_kendalltau=True, print_latex=True, latex_names=names)

    print(f'Time taken: {(time.time() - st) / 60:.2f} min')