'''
If we break down into pages with external > internal and vice-versa, do we see something interesting in characterizing the pages? We can split the pages in those whose internal pages are more than the external and vice versa. The we can look at:
what’s the degree? 
What is the probability of being in that page click after click? 
How much this probability changes the the initial probabilities are defined by the external probability visits?
Narrow down the set of pages to those that have a “steady” number of pageviews over time and repeat the analysis
We can look at the variance of the number of pages and exclude the articles beyond the variance
Or we can make the page-views steady removing the “event”s
'''
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import time

st = time.time()

fig_folder = 'page_views_analysis/figs/'
pathlib.Path(fig_folder).mkdir(parents=True, exist_ok=True)

def load_pickle_file(path):
    with open(path, 'rb') as f:
        loaded_file = pickle.load(f)

    return loaded_file

# Adapted from random_walks.py:random_walk_model2()
def model2_distr_after_t_clicks(pi, C, t):
    '''
     Inputs
        - pi: Initial distribution
        - C: Transition probability matrix (including a sink node)
        - t: Number of clicks

    Output:
        - array of length (num_pages + 1) where entry i is the probability a random surfer
        is at page i after 1 clicks. The last index is the probability of being at the sink
        node
    '''

    print(f'Performing random walk (Model 2) for {t} steps ...')
    pi = pi.todense()
    pi = np.append(np.array(pi).T, 0) # Add 0 probability of starting at external page

    # Probability distribution over pages (last entry represents sink node)
    curr_locs = pi
    for i in range(t):
        curr_locs = curr_locs * C
        print(f'Total fraction at sink node after {i+1} steps: {curr_locs[-1]:.3f}')

    return curr_locs

if __name__ == '__main__':

    ### 1) Load data
    year = '2018'

    ## Page views 
    pv_path = f'page_views_analysis/data/external_and_internal_page_views_{year}.csv' # !!! TODO: Update
    print(f'Reading in {pv_path}')
    month_df = pd.read_csv(pv_path)
    print('Number of rows:', len(month_df))
    print(month_df)

    ## Adjacency matrix
    A_path = f'data/wikilinkgraph/adjacency_matrix_{year}.pkl'
    A = load_pickle_file(A_path)

    ## Initial distribution
    pi_path = f'data/clickstream/final/pi_{year}.pkl'
    pi = load_pickle_file(pi_path)

    ## Transition probability matrix (including sink node)
    C_path = f'data/clickstream/final/C_{year}.pkl'
    C = load_pickle_file(C_path)

    ### 2) Prepare data

    ## Sum over months
    df = month_df.groupby(['title', 'idx'], as_index=False).sum()
    print(df.head(5))

    ## Add column with fraction of views that are internal
    df['frac_internal_pv'] = df['internal_pv'] / (df['internal_pv'] + df['external_pv'])

    ## Add column with in-degree of each page
    in_degrees = np.array(A.sum(axis=0)).squeeze()
    df['in_degree'] = [in_degrees[idx] for idx in df['idx']] 

    ## Add columns with P(at page) after t clicks according to Model 2
    num_clicks_list = [0, 1, 2, 3]
    pi = load_pickle_file(pi_path)
    for t in num_clicks_list:
        probs = model2_distr_after_t_clicks(pi, C, t)
        df[f'prob_at_page_after_{t}_clicks'] = [probs[idx] for idx in df['idx']]

    ### 3) Analysis I 

    df['total_pv'] =  df['external_pv'] + df['internal_pv']

    more_external_pv = df[df['external_pv'] > df['internal_pv']]
    # NOTE: Also includes external == internal case
    more_internal_pv = df[df['external_pv'] <= df['internal_pv']]

    ## Compute fraction of pages with more internal pvs
    frac_internal = len(more_internal_pv) / len(df)
    print(f'Fraction of pages with external <= internal pvs: {frac_internal*100:.2f}%')

    ## Plot distribution of frac_internal_pv
    save_to = os.path.join(fig_folder, 'frac_internal.jpg')
    plt.hist(df['frac_internal_pv'])
    plt.ylabel('Count')
    plt.xlabel('Fraction of page views that are internal')
    plt.title('Distribution of internal page view fraction')
    plt.savefig(save_to)
    print(f'Saved plot of distribution of fraction of internal page views to {save_to}')
    plt.clf()

    ## Plot in-degree distributions
    save_to = os.path.join(fig_folder, 'in_degree.jpg')
    plt.style.use('seaborn-deep')

    x = more_external_pv['in_degree']
    y = more_internal_pv['in_degree']
    bins = np.linspace(0, 10000, 30)

    fig, axs = plt.subplots(2,1)
    ax = axs[0]
    ax.hist(x, bins, alpha=0.5, label='external > internal pv', color='green')
    ax.legend(loc='upper right')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Count')
    ax.set_title(f'Average in-degree={x.mean():.1f} (sd={x.std():.1f})')

    ax = axs[1]
    ax.hist(y, bins, alpha=0.5, label='external <= internal pv', color='blue')
    ax.legend(loc='upper right')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Count')
    ax.set_title(f'Average in-degree={y.mean():.1f} (sd={y.std():.1f})')

    # plt.hist([x, y], label=['external > internal pv', 'external <= internal pv'], density=True)
    plt.tight_layout()
    plt.savefig(save_to)
    print(f'Saved in-degree distribution plot to {save_to}')
    plt.clf()

    ## What is the probability of being at ext > int vs. int >= ext pages click after click? 
    save_to = os.path.join(fig_folder, 'ext_vs_int_probs_over_clicks.jpg')
    prob_at_mostly_extpv_pages = [more_external_pv[f'prob_at_page_after_{t}_clicks'].sum() for t in num_clicks_list]
    prob_at_mostly_intpv_pages = [more_internal_pv[f'prob_at_page_after_{t}_clicks'].sum() for t in num_clicks_list]

    plt.plot(prob_at_mostly_extpv_pages, label='Pages with more external page views', marker='o')
    plt.plot(prob_at_mostly_intpv_pages, label='Pages with more internal page views', marker='o')
    plt.xlabel('Number of clicks (t)')
    plt.ylabel('Probability of being at given page type after t clicks')
    plt.title('Probability of being at ext > int pv vs. int >= ext pv pages')
    plt.legend()
    plt.ylim(0,1)
    plt.savefig(save_to)
    print(f'Saved plot of probabilities of ext > int pv vs. int >= ext pv pages click after click to {save_to}')
    plt.clf()

    # Version 2: Normalize by prevalence
    save_to = os.path.join(fig_folder, 'ext_vs_int_probs_over_clicks_v2.jpg')
    prob_at_mostly_extpv_pages = [more_external_pv[f'prob_at_page_after_{t}_clicks'].sum() / (1-frac_internal) for t in num_clicks_list]
    prob_at_mostly_intpv_pages = [more_internal_pv[f'prob_at_page_after_{t}_clicks'].sum() / frac_internal for t in num_clicks_list]

    plt.plot(prob_at_mostly_extpv_pages, label='Pages with more external page views', marker='o')
    plt.plot(prob_at_mostly_intpv_pages, label='Pages with more internal page views', marker='o')
    plt.xlabel('Number of clicks (t)')
    plt.ylabel('Normalized probability of being at given page type after t clicks')
    plt.title('Normalized probability of being at ext > int pv vs. int >= ext pv pages')
    plt.legend()
    # plt.ylim(0,1)
    plt.savefig(save_to)
    print(f'Saved plot of normalized probabilities of ext > int pv vs. int >= ext pv pages click after click to {save_to}')
    plt.clf()

    ## Compare average page views of the two types of pages
    print(f"Total page views for pages with mostly internal pv: mean={more_internal_pv['total_pv'].mean():.1f}, sd={more_internal_pv['total_pv'].std():.1f}")
    print(f"Total page views for pages with mostly external pv: mean={more_external_pv['total_pv'].mean():.1f}, sd={more_external_pv['total_pv'].std():.1f}")

    ### 4) Analysis II - Analyze steadiness of pageviews
    save_to = os.path.join(fig_folder, 'mean_pv_vs_sd.jpg')
    month_df['total_pv'] = month_df['internal_pv'] + month_df['external_pv']
    avg_df = month_df.groupby(['title', 'idx'], as_index=False)['total_pv'].mean()
    sd_df = month_df.groupby(['title', 'idx'], as_index=False)['total_pv'].std()
    df2 = pd.merge(avg_df, sd_df, on=['title', 'idx'])
    df2.rename(columns={'total_pv_x': 'total_pv_mean', 'total_pv_y': 'total_pv_sd'}, inplace=True)
    
    print(df2[['total_pv_mean', 'total_pv_sd']].cov())
    
    # plt.scatter(df2['total_pv_mean'], df2['total_pv_sd']) 
    # plt.title('Mean page views vs. standard deviation')
    # plt.xlabel('Mean monthly page views')
    # plt.ylabel('Standard deviation')
    # plt.xlim(0, 10000) # Exclude outliers
    # plt.savefig(save_to)
    # print(f'Saved plot of mean page views vs. std dev to {save_to}')
    # plt.clf()

    # Normalize std dev by dividing by mean. This yields the "coefficient of variation" (Is there a better way?)
    
    df2['total_pv_norm_sd'] = df2['total_pv_sd'] / df2['total_pv_mean'] 
    print(df2[['total_pv_mean', 'total_pv_norm_sd']].cov())
    # Check that there is no longer a correlation
    # save_to = os.path.join(fig_folder, 'mean_pv_vs_norm_sd.jpg')
    # plt.scatter(df2['total_pv_mean'], df2['total_pv_norm_sd'])
    # plt.title('Mean page views vs. coefficient of variation')
    # plt.xlabel('Mean monthly page views')
    # plt.ylabel('standard deviation / mean')
    # plt.xlim(0, 10000) # Exclude outliers
    # plt.savefig(save_to)
    # print(f'Saved plot of mean page views vs. normalized std dev to {save_to}')
    # plt.clf()

    print(df2[['total_pv_mean', 'total_pv_sd']].corr())
    print(df2[['total_pv_mean', 'total_pv_norm_sd']].corr())

    # Measure page view steadiness as percentage change compared to prev month
    # RQ: Are increases in page views at all driven by internal page views? 
    #   Or is it almost all due to an increase in external page views


    # RQ: Do pages with int > ext pv get less page views overall?


    print(f'Time taken: {(time.time() - st) / 60:.2f} min')