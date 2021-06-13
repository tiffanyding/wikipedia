# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle
import scipy.sparse as ss
import time

from scipy.sparse.linalg import inv
from scipy.stats import geom

from compute_correlations import compute_corrcoeff
from get_top_titles import get_top_n
from map_to_topic import get_topic_distribution, plot_topic_distr
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

def random_walk_model1(pi, B, p=.8, max_len=15):
    '''
    p = parameter of Geometric distribution
    '''
    print(f'Performing random walk (Model 1, p={p})...')
    print(f'Geometric({p}):', [geom.pmf(x, p) for x in range(max_len)])

    # Keep track of "number" of visits to each page over time 
    # (not an actual number because it can be fractional)
    # Weight probability distribution at each time step t by probability 
    # that random walk has not ended before t (= 1 - cdf(t))
    num_visits = np.zeros(np.shape(pi))

    # Probability distribution over pages
    curr_locs = pi
    for i in range(max_len):
        weight = 1 - geom.cdf(i, p)
        print(f'Step {i}, weight:', weight)
        num_visits += weight * curr_locs 
        curr_locs = curr_locs * B

    return num_visits

def random_walk_model2(pi, C, max_len=15, print_top10=False, return_frac_sink=False,
                        compute_correlation_with=None, return_topic_distr=False):
    '''
    Inputs
        - pi: Initial distribution
        - C: transition probability matrix (including a sink node)
        - max_len: Number of steps to simulate random walk for
        - print_top10: If True, print the titles of the 10 pages a user is most 
            likely to be at after each step
        - return_frac_sink: If True, return list of fraction of vertices at sink node
            over time
        - compute_correlation_with: None, or an array of length num_pages. At each step,
          we compute the Kendall's tau correlation of the Model 2 probability distribution 
          with this array
    '''
    print('Performing random walk (Model 2)...')
    pi = pi.todense()
    pi = np.append(np.array(pi).T, 0) # Add 0 probability of starting at external page

    # Keep track of "number" of visits to each page over time 
    # (not an actual number because it can be fractional)
    num_visits = np.zeros(np.shape(pi))

    if return_frac_sink:
        frac_sink = [0] # Fraction of vertices at sink node is initially 0
    
    if compute_correlation_with is not None:
        correlations = [compute_corrcoeff(pi[:-1], compute_correlation_with, use_kendalltau=True)]

    if return_topic_distr:
        topic_distr = [get_topic_distribution(pi[:-1])]

    # Probability distribution over pages (last entry represents sink node)
    curr_locs = pi
    for i in range(max_len):
        num_visits += curr_locs
        curr_locs = curr_locs * C
        print(f'Total fraction at sink node after {i+1} steps: {curr_locs[-1]:.3f}')
        frac_sink.append(curr_locs[-1])

        if print_top10:
            print(f'Top pages after {i+1} steps:')
            get_top_n(curr_locs[:-1], n=10) 
        
        if compute_correlation_with is not None:
            correlations.append(compute_corrcoeff(curr_locs[:-1], compute_correlation_with, use_kendalltau=True))

        if return_topic_distr:
            normalized_distr = curr_locs[:-1] / curr_locs[:-1].sum() # Ignore sink node probability
            topic_distr.append(get_topic_distribution(normalized_distr))

    # Exclude visits to external 
    num_visits = num_visits[:-1]

    # TODO: fix this so that we don't have to return everything
    return num_visits, np.array(frac_sink), np.array(correlations), np.array(topic_distr)

    # if return_frac_sink and compute_correlation_with is None:
    #     return num_visits, np.array(frac_sink)
    # elif not return_frac_sink and compute_correlation_with is not None:
    #     return num_visits, np.array(correlations)
    # elif return_frac_sink and compute_correlation_with is not None:
    #     return num_visits, np.array(frac_sink), np.array(correlations)
    # else:
    #     return num_visits

def uniform_random_walk(pi, adjacency_matrix, p=0.8, max_len=15, return_topic_distr=False,
                            compute_correlation_with=None):
    '''
    Similar to random_walk_model1, but with uniform transition probability
    '''

    print(f'Performing uniform random walk with p={p}...')
    A = adjacency_matrix
    n = A.shape[0]

    if return_topic_distr:
        topic_distr = [get_topic_distribution(pi)]

    # Compute uniform transition probability matrix
    degree_seq = np.squeeze(np.asarray(A.sum(axis=1)))
    # Add 1 to each degree so that no vertex has out-degree zero
    degree_seq += 1
    inv_D = ss.diags(1 / degree_seq, format='csc')
    P = inv_D * A
        
    # Keep track of "number" of visits to each page over time 
    # (not an actual number because it can be fractional)
    # Weight probability distribution at each time step t by probability 
    # that random walk has not ended before t (= 1 - cdf(t))
    num_visits = np.zeros(np.shape(pi))

    if compute_correlation_with is not None:
        correlations = [compute_corrcoeff(pi, compute_correlation_with, use_kendalltau=True)]

    # Probability distribution over pages
    curr_locs = pi
    for i in range(max_len):
        weight = 1 - geom.cdf(i, p)
        print(f'Step {i}, weight:', weight)
        num_visits += weight * curr_locs 
        curr_locs = curr_locs * P

        if return_topic_distr:
            topic_distr.append(get_topic_distribution(curr_locs))

        if compute_correlation_with is not None:
            correlations.append(compute_corrcoeff(curr_locs, compute_correlation_with, use_kendalltau=True))

    if return_topic_distr and compute_correlation_with is None:
        return num_visits, np.array(topic_distr)
    elif not return_topic_distr and compute_correlation_with is not None:
        return num_visits, np.array(correlations)
    elif return_topic_distr and compute_correlation_with is not None:
        return num_visits, np.array(topic_distr), np.array(correlations)
    else:
        return num_visits

if __name__ == '__main__':

    st = time.time()

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

    # 2) Simulate random walks to estimate proportion of time spent at each page
    #     and save results
    # (a) PageRank (unweighted)
    pr = compute_unweighted_pagerank(A, d=.85, tol=1e-6)
    save_to = f'{save_folder}/pagerank_{year}.pkl'
    save_to_pickle(pr, save_to, description=f'{year} PageRanks')

    # (b) PageRank (weighted)
    weighted_pr = compute_weighted_pagerank(B, d=.85, tol=1e-6)
    save_to = f'{save_folder}/weighted_pagerank_{year}.pkl'
    save_to_pickle(weighted_pr, save_to, description=f'{year} weighted PageRanks')


    # (c) Random Walk Model 1
    p = 0.5
    rw1 = random_walk_model1(pi, B, p=p, max_len=15)
    save_to = f'{save_folder}/rw1_p={p}_{year}.pkl'
    save_to_pickle(rw1, save_to, description=f'{year} random walk (Model 1)')

    # (d) Random Walk Model 2
    # Load in PR so I can compute correlation with it
    pr = load_pickle_file(f'{save_folder}/pagerank_{year}.pkl')
    rw2, frac_sink, rw2_prcorrs, topic_distrs = random_walk_model2(pi, C, max_len=15, print_top10=True, 
                                        return_frac_sink=True,
                                        compute_correlation_with=pr)
    save_to = f'{save_folder}/rw2_{year}.pkl'
    save_to_pickle(rw2, save_to, description=f'{year} random walk (Model 2)')
    save_to = f'{save_folder}/rw2_fracsink_{year}.pkl'
    save_to_pickle(frac_sink, save_to, description=f'{year} Model 2 fraction at sink node')
    save_to = f'{save_folder}/rw2_PRcorr_{year}.pkl'
    save_to_pickle(rw2_prcorrs, save_to, description=f'{year} Model 2 correlation with PageRank')
    save_to = f'{save_folder}/rw2_topic_distr_{year}.pkl'
    save_to_pickle(topic_distrs, save_to, description=f'{year} Model 2 topic distributions')


    # # (e) Uniform random walk
    p = 0.8
    uniform_rw, uniform_topic_distrs, uniform_prcorrs = uniform_random_walk(pi, A, p=p, max_len=15, return_topic_distr=True,
                                        compute_correlation_with=pr)
    save_to = f'{save_folder}/uniform_p={p}_{year}.pkl'
    save_to_pickle(uniform_rw, save_to, description=f'{year} uniform random walk')
    save_to = f'{save_folder}/uniform_topic_distr_{year}.pkl'
    save_to_pickle(uniform_topic_distrs, save_to, description=f'{year} Uniform random walk topic distributions')
    save_to = f'{save_folder}/uniform_PRcorr_{year}.pkl'
    save_to_pickle(uniform_prcorrs, save_to, description=f'{year} Uniform random walk correlation with PageRank')


    ## 3) Make additional plots

    # --- PLOT 1 ----
    # Plot Model 2 fraction at sink over time
    frac_sink = load_pickle_file(f'{save_folder}/rw2_fracsink_{year}.pkl')
    # # Version 1
    # plt.plot(range(len(frac_sink)), frac_sink, '-o')
    # plt.xlabel('Number of steps')
    # plt.ylabel('Fraction of users that have exited')
    # plt.xlim(0, 10)
    # save_to = 'figs/model2_frac_exited.jpg'
    # plt.savefig(save_to)
    # print(f'Saved Model 2 fraction exited plot to {save_to}')

    # Version 2
    fig, ax = plt.subplots(figsize=(8,5))
    percent_left = (1 - np.array(frac_sink)) * 100
    print(percent_left)
    print('sum frac_sink', np.sum(frac_sink))
    print('sum percent_left', np.sum(percent_left))
    percent_left = percent_left[:11] # Select steps 0-10

    ax.bar(range(len(percent_left)), percent_left)
    ax.set_xlabel('Number of clicks (t)', fontsize=12)
    ax.set_ylabel('Percentage of users remaining (%)', fontsize=12)
    ax.set_title('Percentage of users that remain on Wikipedia after t clicks', fontsize=12)
    ax.set_xticks(range(len(percent_left)))

    # Add bar labels
    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for idx, i in enumerate(ax.patches):
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x()-.03, i.get_height()+.5, f'{percent_left[idx]:.2f}%', fontsize=10,
                    color='dimgrey')

    save_to = 'figs/model2_frac_exited_bar.jpg'
    plt.savefig(save_to)
    print(f'Saved Model 2 fraction exited bar plot to {save_to}')

    # --- PLOT 2 ---
    # Plot correlation of Model 2 and Uniform(0.8) with PageRank at each step
    rw2_prcorrs = load_pickle_file(f'{save_folder}/rw2_PRcorr_{year}.pkl')
    uniform_prcorrs = load_pickle_file(f'{save_folder}/uniform_PRcorr_{year}.pkl')
    plt.plot(range(len(rw2_prcorrs)), rw2_prcorrs, '-o', label='Model 2')
    plt.plot(range(len(uniform_prcorrs)), uniform_prcorrs, '-o', label='Uniform$^{(0.8)}$')
    plt.legend()
    plt.xlabel('Number of clicks (t)')
    plt.ylabel("Correlation with PageRank")
    plt.xticks(range(len(rw2_prcorrs)))
    plt.title('Centrality of pages visited at each click')

    save_to = 'figs/model2_and_uniform_centrality.jpg'
    plt.savefig(save_to)
    print(f'Saved Model 2 and Uniform(0.8) centrality plot to {save_to}')

    # --- PLOT 3 ---
    # Plot Model 2 topic distributions over time
    topic_distrs = np.array(load_pickle_file(f'{save_folder}/rw2_topic_distr_{year}.pkl'))
    # Only consider first 5 clicks
    topic_distrs = topic_distrs[:5]
    names = [f'Click {i}' for i in range(len(topic_distrs))]
    plot_topic_distr(topic_distrs, names=names, save_to='figs/rw2_topic_distr.png')

    # --- PLOT 4 ---
    # Plot Uniform topic distributions over time
    uniform_topic_distrs = np.array(load_pickle_file(f'{save_folder}/uniform_topic_distr_{year}.pkl'))
    # Only consider first 5 clicks
    uniform_topic_distrs = uniform_topic_distrs[:5]
    names = [f'Click {i}' for i in range(len(uniform_topic_distrs))]
    plot_topic_distr(uniform_topic_distrs, names=names, save_to='figs/uniform_topic_distr.png')

    print(f'Time taken: {(time.time() - st) / 60:.2f} min')




    
