'''
See jun20_analysis.py

Goals: Rebuild it with uniform transition probabilities and quitting prob from clickstream (averaged over longer period) - check how kendal works
Use the model to predict the pageviews (from step 1) for each page and compare it with the internal visits. To make the numbers more comparable, initialize the walks according to the external probabilities of pages (which will correspond to the initial walk probability of real numbers)
How well do we do? Which pages with higher differences? Can we characterize them? (If you have not time for the last question it is fine!)
Relation of centrality (like PageRank) and total pageviews (if I’m not wrong you did it already using Kendall as metric)
'''

import matplotlib.pyplot as plt
import os
import pandas as pd
# import sys
import time

from random_walks import *

if __name__ == '__main__':

    st = time.time()

    year = '2018'

    save_folder = 'results'
    fig_folder = 'figs/jun18'

    # ---------
    # Make folders if necessary
    pathlib.Path(save_folder).mkdir(exist_ok=True)
    pathlib.Path(fig_folder).mkdir(exist_ok=True)

    ## 1) Load data
    A_path = f'data/wikilinkgraph/adjacency_matrix_{year}.pkl'
    C_path = f'data/clickstream/final/C_{year}.pkl'
    pi_path = f'data/clickstream/final/pi_{year}.pkl'

    A = load_pickle_file(A_path)
    C = load_pickle_file(C_path)
    pi = load_pickle_file(pi_path)
    exit_probs = C[:,-1]

    ## 2) Simulate uniform random walk w/ exit probability
    max_len = 15
    save_to = f'{save_folder}/uniform_wexit_{year}.pkl'
#     uniform_wexit_rw = uniform_random_walk2(pi, A, exit_probs, max_len=max_len, return_topic_distr=False,
#             compute_correlation_with=None)
#     save_to_pickle(uniform_wexit_rw, save_to, description=f'{year} uniform random walk with exit probability')

    ## 3) Plot scatterplot between page visits predicted by uniform random walk and ground truth page views
    uniform_wexit_rw = load_pickle_file(save_to)
    pageviews_internal_and_external = load_pickle_file(f'results/pageviews_internal_and_external_{year}.pkl')
    pageviews_internal_and_external = np.squeeze(np.array(pageviews_internal_and_external))

    # Scale by total number of external pageviews / max_len. This makes it so the number of walks at click 0 
    # is equal to number of external pageviews
    pageviews_internal = load_pickle_file(f'results/pageviews_internal_{year}.pkl')
    num_external_pageviews = np.sum(pageviews_internal_and_external) - np.sum(pageviews_internal)
    print(f'Total number of external pageviews: {num_external_pageviews}')
    uniform_wexit_rw = num_external_pageviews * uniform_wexit_rw

#     plt.scatter(uniform_wexit_rw, pageviews_internal_and_external)
    # plt.ylim(0, 1000000)
    # plt.xlim(0, 0.00004)
    plt.figure(figsize=(6,6))
    plt.scatter(np.log(uniform_wexit_rw), np.log(pageviews_internal_and_external))
    plt.xlabel('Log expected number of page visits according to uniform RW')
    plt.ylabel('Log ground truth page views')
    plt.ylim(0, 23)
    plt.xlim(0, 23)
    plt.plot([0,23],[0,23], label='y=x', color='red', zorder=10)
    plt.legend()

    save_to = f'{fig_folder}/uniformrw_vs_pageviews_loglog.jpg'
    plt.savefig(save_to)
    print(f'Saved scatter plot of uniform RW with exit probabilitis vs. ground truth page visits to {save_to}')
    plt.clf()

    ## 4) Plot distributions of differences between expected and actual page views
    ## (a) Raw difference: expected - actual
    raw_diff = uniform_wexit_rw - pageviews_internal_and_external 
    bins = np.linspace(-50, 150, 50)
    plt.figure(figsize=(6,4))
    plt.hist(raw_diff, bins=bins)
    plt.xlabel('Expected - actual page views')
    plt.ylabel('Count')
    plt.title('Distribution of raw difference between expected and actual PV')
    plt.tight_layout()
    save_to = f'{fig_folder}/raw_diff_zoomedin.jpg'
    plt.savefig(save_to)
    print(f'Saved raw expected - actual page views plot to {save_to}')
    plt.clf()
    print('Descriptive statistics:')
    df = pd.DataFrame({'raw_diff': raw_diff})
    print(df.describe())   

    ## (b) Normalized difference: (expected - actual) / actual
    normed_diff = (uniform_wexit_rw - pageviews_internal_and_external) / pageviews_internal_and_external
    # Filter out pages with 0 actual PV
    nonzero_pv_idx = ~(pageviews_internal_and_external==0)
    normed_diff = normed_diff[nonzero_pv_idx]
    print(f'Excluding {(pageviews_internal_and_external==0).sum()} pages with 0 actual pageviews from histogram')
    bins = np.linspace(-3, 5, 100)
    plt.hist(normed_diff, bins=bins)
    plt.xlabel('(Expected - actual PV) / actual PV')
    plt.ylabel('Count')
    plt.title('Distribution of normalized difference between expected and actual PV')
    plt.tight_layout()
    save_to = f'{fig_folder}/normed_diff_zoomedin.jpg'
    plt.savefig(save_to)
    print(f'Saved normalized difference of expected - actual page views plot to {save_to}')
    plt.clf()
    print('Descriptive statistics:')
    df = pd.DataFrame({'normed_diff': normed_diff})
    print(df.describe())

    ## Plot correlations
    raw_diff_and_pv_corr = np.corrcoef(raw_diff, pageviews_internal_and_external)
    normed_diff_and_pv_corr = np.corrcoef(normed_diff, pageviews_internal_and_external[nonzero_pv_idx])
    print(f'Correlation between raw diff and PV: {raw_diff_and_pv_corr[0][1]:.3f}')
    print(f'Correlation between normed diff and PV: {normed_diff_and_pv_corr[0][1]:.3f}')

    print(f'Time taken: {(time.time() - st) / 60:.2f} min')