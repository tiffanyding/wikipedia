# import pandas as pd
import numpy as np
import pathlib
import pickle
import scipy.sparse as ss
import time

from scipy.sparse.linalg import inv

def compute_unweighted_pagerank(A, d=.85, tol=1e-6):
    '''
    Inputs:
        A: adjacency matrix of 1s and 0s
        d: damping factor. (1-d) is probability that surfer jumps to a random page
        tol: threshold for PageRank convergence
    '''
    n = A.shape[0]

    # Compute uniform transition probability matrix
    degree_seq = np.sum(A, 1)
    D = ss.diags(degree_seq, format='csc')
    P = inv(D) * A

    # Iterate until convergence of l1 norm
    pr = (1 / n) * np.matrix(np.ones((1,n)))
    prev_pr = np.matrix(np.ones((1,n)))
    dist = np.inf
    iter = 0
    while dist > tol:
        iter += 1
        prev_pr = pr
        pr = (1 - d) / n + (d * prev_pr * P)
        dist = np.sum(np.abs(pr - prev_pr))
        print(f'Iteration {iter}. Change in l1 norm = {dist:.7f}')
    return pr

def compute_weighted_pagerank(P):
    # TODO (use clickstream graph)
    pass


if __name__ == '__main__':

    year = '2018'

    # ---------
    A_path = f'data/wikilinkgraph/adjacency_matrix_{year}.pkl'

    # # Uncomment to test code by creating random matrix of 0s and 1s
    # num_pages = 1000
    # A = np.random.rand(num_pages, num_pages)

    with open(A_path, 'rb') as f:
        A = pickle.load(f)

    pr = compute_unweighted_pagerank(A, d=.85, tol=1e-6)

    ## Save results
    save_folder = 'data/results'
    save_to = f'{save_folder}/pagerank_{year}.pkl'
    # Make folder if necessary
    pathlib.Path(save_folder).mkdir(exist_ok=True)
    with open(save_to, 'wb') as f:
        pickle.dump(pr, f)
    print(f'Saved {year} PageRanks to {save_to}')