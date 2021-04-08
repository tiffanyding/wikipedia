# import pandas as pd
import numpy as np
import pathlib
import pickle
import scipy.sparse as ss
import time

from scipy.sparse.linalg import inv
from scipy.stats import geom

from utils import load_pickle_file, save_to_pickle

def compute_unweighted_pagerank(adjacency_matrix, d=.85, tol=1e-6):
    '''
    Inputs:
        adjacency_matrix: adjacency matrix of 1s and 0s
        d: damping factor. (1-d) is probability that surfer jumps to a random page
        tol: threshold for PageRank convergence
    '''
    print('Computing unweighted PageRank...')
    A = adjacency_matrix
    n = A.shape[0]

    # Compute uniform transition probability matrix
    degree_seq = np.squeeze(np.asarray(A.sum(axis=1)))
    # Add 1 to each degree so that no vertex has out-degree zero
    degree_seq += 1
    inv_D = ss.diags(1 / degree_seq, format='csc')
    P = inv_D * A

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

def compute_weighted_pagerank(B, d=.85, tol=1e-6):
    print('Computing weighted PageRank...')
    # Uses clickstream graph to determine transition probabilities
    n = B.shape[0]
    P = B
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

def random_walk_model1(pi, B, p=.8, max_len=20):
    '''
    p = parameter of Geometric distribution
    '''
    print('Performing random walk (Model 1)...')
    print(f'Geometric({p}):', [geom.pmf(x, p) for x in range(max_len)])

    # Keep track of "number" of visits to each page over time 
    # (not an actual number because it can be fractional)
    # Weight probability distribution at each time step t by probability 
    # that random walk has not ended before t (= 1 - cdf(t))
    num_visits = np.zeros(np.shape(pi))

    # Probability distribution over pages
    curr_locs = pi
    for i in range(max_len):
        weight = geom.cdf(i, p)
        print('Weight:', weight)
        num_visits += weight * curr_locs 
        curr_locs = curr_locs * B

    return num_visits

def random_walk_model2(pi, C, max_len=30):
    '''
        max_len: Number of steps to simulate random walk for
    '''
    print('Performing random walk (Model 2)...')
    pi = pi.todense()
    pi = np.append(np.array(pi).T, 0) # Add 0 probability of starting at external page

    # Keep track of "number" of visits to each page over time 
    # (not an actual number because it can be fractional)
    num_visits = np.zeros(np.shape(pi))

    # Probability distribution over pages
    curr_locs = pi
    for i in range(max_len):
        num_visits += curr_locs
        curr_locs = curr_locs * C

    # Exclude visits to external 
    num_visits = num_visits[:-1]

    return num_visits
        

if __name__ == '__main__':

    year = '2018'

    save_folder = 'results'

    # ---------
    # Make save_folder if necessary
    pathlib.Path(save_folder).mkdir(exist_ok=True)

    ## 1) Load data
    A_path = f'data/wikilinkgraph/adjacency_matrix_{year}.pkl'
    B_path = f'data/clickstream/final/B_{year}.pkl'
    C_path = f'data/clickstream/final/C_{year}.pkl'
    pi_path = f'data/clickstream/final/pi_{year}.pkl'

    # # Uncomment to test code by creating random matrix of 0s and 1s
    # num_pages = 1000
    # A = np.random.rand(num_pages, num_pages)
    # A = ss.csc_matrix(A)

    A = load_pickle_file(A_path)
    B = load_pickle_file(B_path)
    C = load_pickle_file(C_path)
    pi = load_pickle_file(pi_path)

    ## 2) Simulate random walks to estimate proportion of time spent at each page
    #     and save results
    # (a) PageRank (unweighted)
    # pr = compute_unweighted_pagerank(A, d=.85, tol=1e-6)
    # save_to = f'{save_folder}/pagerank_{year}.pkl'
    # save_to_pickle(pr, save_to, description=f'{year} PageRanks')

    # (b) PageRank (weighted)
    weighted_pr = compute_weighted_pagerank(B, d=.85, tol=1e-6)
    save_to = f'{save_folder}/weighted_pagerank_{year}.pkl'
    save_to_pickle(weighted_pr, save_to, description=f'{year} weighted PageRanks')


    # (c) Random Walk Model 1
    rw1 = random_walk_model1(pi, B, p=.8, max_len=20)
    save_to = f'{save_folder}/rw1_{year}.pkl'
    save_to_pickle(rw1, save_to, description=f'{year} random walk (Model 1)')

    # (d) Random Walk Model 2
    rw2 = random_walk_model2(pi, C, max_len=10)
    save_to = f'{save_folder}/rw2_{year}.pkl'
    save_to_pickle(rw2, save_to, description=f'{year} random walk (Model 2)')



    
