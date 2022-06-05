#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import os
import ufekt
from sklearn.neighbors import NearestNeighbors


def nn(mat, dis_th, _n_neighbors=5, _algorithm='ball_tree'):
    '''
    A function of the nearest neighbor algorithm
    Returns
    ----------
    distance : 
        distance from every data point to its nearest neighbor
        1st column indicates distance to itself, i.e., zero
        2nd column indicates distance to the nearest neighbor
        3rd column indicates distance to the 2nd nearest neighbor
        ...
    out : 
        ...
    '''
    np.random.seed(0)
    nn = NearestNeighbors(n_neighbors=_n_neighbors, algorithm=_algorithm).fit(mat)
    distances, indices = nn.kneighbors(mat)
    dis = distances[:,1]
    out = np.where(dis > dis_th)
    return(distances, out)

def sample(fname_input, max_rank, min_rank, window_size, sigma, knn_k, od):
    # ------------------------------------------------------------------------
    # Parameters
    UFEKT_rank_opt_th = min_rank
    UFEKT_window_size = window_size
    UFEKT_sigma = sigma
    UFEKT_knn_k = knn_k
    print('------------------------------------------------------------------------')
    print('file=\"' + fname_input + '\", min_rank=' + str(min_rank) + ', max_rank=' + str(max_rank) + 
        ', window_size=' + str(window_size) + ', sigma=' + str(sigma) + ', knn_k=' + str(knn_k) + ', od=' + str(od))

    # ------------------------------------------------------------------------
    # read a csv file of datasets
    file = fname_input
    x = np.loadtxt(file, delimiter=',', dtype='float64')
    UFEKT_rank_range = [x.shape[1]-1, max_rank, max_rank]

    # --------------------------------------------------------------------
    # execute UFEKT to extract features from multivariate time series
    print('Executing UFEKT')
    core,factors,tensor,ranks,df_opt = ufekt.ufekt(
        x[:,0:], width=UFEKT_window_size, sigma=UFEKT_sigma, rank_range=UFEKT_rank_range, rank_th=UFEKT_rank_opt_th, seed=0)

    print('Saving the reuslts')
    base_fname = os.path.basename(file)
    base_fname = os.path.splitext(base_fname)[0]
    fname = base_fname + '_factors_1.csv'
    np.savetxt(fname, factors[1], fmt='%0.8f', delimiter=',')

    # --------------------------------------------------------------------
    # execute kNN for outlier detection
    if od:
        print('kth-Nearest Neighbor for outlier detection')
        score_pred,outlier_rows = nn(factors[1], dis_th=0.01, _n_neighbors=UFEKT_knn_k, _algorithm='ball_tree')

        print('Saving the reuslts')
        fname = base_fname + '_scores.csv'
        np.savetxt(fname, score_pred[:,(UFEKT_knn_k-1)].T, fmt='%0.8f', delimiter=',')

    return(True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str,
        help='set a csv file of time series')
    parser.add_argument('--max_rank', default=50, type=int,
        help='set a maximum rank to specify search range to find the best rank for outlier detection [default value: 50]')
    parser.add_argument('--min_rank', default=10, type=int,
        help='set a minimum rank to specify search range to find the best rank for outlier detection [default value: 10]')
    parser.add_argument('-w', '--window_size', default=2, type=int, 
        help='set a window size or the length of subsequence of time series')
    parser.add_argument('-s', '--sigma', default=1.0, 
        help='set a sigma used in RBF kernel [default value: 1.0]')
    parser.add_argument('-k', '--knn_k', default=5, type=int, 
        help='set the number of "k" used in k-th nearest neighbor algorithm [default value: 5]')
    parser.add_argument('-od', '--od', action='store_true', 
        help='outlier detection is executed using obtained feature vectors from UFEKT and k-th nearest neighbor algorithm if true')
    args = parser.parse_args()
    return(args)

def main():
    args = get_args()
    sample(fname_input = args.filename, 
        max_rank = args.max_rank, 
        min_rank = args.min_rank, 
        window_size = args.window_size, 
        sigma = args.sigma, 
        knn_k = args.knn_k,
        od = args.od)
    return(True)

if __name__ == '__main__':
    main()
    print('Completed.')


