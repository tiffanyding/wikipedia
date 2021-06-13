import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import scipy.sparse as ss
import time

from utils import load_pickle_file, save_to_pickle

def get_topic_distribution(arr, norm_topic_matrix=None):
    '''
    arr = vector of probabilities that user is at each page

    Multiply P(at page j) by (fraction of page j's content that belongs to topic x)

    Returns vector of length num_topics
    '''
    if norm_topic_matrix is None:
        print('Using default normalized topic_matrix for 2018')
        year = 2018
        norm_topic_matrix = load_pickle_file(f'data/normalized_topic_matrix_{year}.pkl')
        # level2topic_to_idx_2018 = load_pickle_file(f'data/level2topic_to_idx_{year}.pkl')

        # print(level2topic_to_idx_2018)

    # Convert array to 1 x num_pages
    if ss.issparse(arr):
        arr = arr.toarray()
    arr = arr.reshape((1, -1))

    # Make sure arr sums to 1
    arr = arr / np.sum(arr)

    print('arr', arr.shape, arr.sum())
    print('norm_topic_matrix', norm_topic_matrix.shape)
    topic_distr = arr * norm_topic_matrix

    return np.squeeze(np.array(topic_distr))

def plot_topic_distr(distr_matrix, names, idx_to_topic=None, save_to=None):
    '''
    Each row of distr_matrix is a distribution over topics
    '''
    distr_matrix = np.array(distr_matrix)
    num_distr, num_topics = np.shape(distr_matrix)

    print('distr', distr_matrix.shape)
    if idx_to_topic is None:
        print('Using default topic to index map for 2018')
        year = 2018
        idx_to_topic = load_pickle_file(f'data/idx_to_level2topic_{year}.pkl')

        print('idx_to_topic', idx_to_topic)

    # Create color palette
    import random
    random.seed(0)
    num_colors = num_topics
    colors = []
    for i in range(num_colors):
        color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        colors.append(color)
    
    ind = np.arange(num_distr)
    plt.figure(figsize=(12,10))
    for i in range(num_topics):
        plt.bar(ind, distr_matrix[:,i], bottom=1-distr_matrix[:,:i+1].sum(axis=1), label=idx_to_topic[i], color=colors[i])
    
    plt.legend(fontsize=12)
    plt.ylabel('Page topic', fontsize=18)
    plt.xticks(ind, names, rotation=45, fontsize=16)
    plt.xlim(-0.5, num_distr+5)
    plt.title('Topic distribution', fontsize=18)
    plt.tight_layout()

    if save_to is not None:
        plt.savefig(save_to)
        print(f'Saved topic distribution plot to {save_to}')

if __name__ == '__main__':
    # Test functionality

    year = '2018'

    arr = np.load(f'results/pageviews_{year}.npy')
    arr = arr / arr.sum()
    topic_distr1 = get_topic_distribution(arr, norm_topic_matrix=None)
    print(topic_distr1, np.sum(topic_distr1))

    arr = load_pickle_file(f'results/pageviews_internal_{year}.pkl')
    arr = arr / arr.sum()
    topic_distr2 = get_topic_distribution(arr, norm_topic_matrix=None)
    print(topic_distr2, np.sum(topic_distr2))

    
    plot_topic_distr([topic_distr1, topic_distr2], ['Total pageviews', 'External pageviews'], 
                            save_to='figs/topic_distribution.jpg')