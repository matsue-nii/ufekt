# UFEKT
UFEKT (Unsupervised Feature Extraction uisng Kernel Method and Tucker Decomposition)

## Overview
UFEKT can be used for extracting fetatures from multivariate time series. The resulting vectors from UFEKT are representing features of the multivariate time series and they can be used for variety of applications such as outlier detection and clustering.

## Keywords
multivariate time series, unsupervised feature extraction, tensor decomposition, outlier detection.

## For more details
Please see the following paper for more details: 
- Matsue, K., Sugiyama, M.: **Unsupervised Tensor based Feature Extraction and Outlier Detection for Multivariate Time Series**, 2021 IEEE 8th International Conference on Data Science and Advanced Analytics (DSAA), the paper is available from *[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9564117)*.
```
@inproceedings{Matsue2021Unsupervised,
    Author = {Matsue, K. and Sugiyama, M.},
    Title = {Unsupervised Tensor based Feature Extraction and Outlier Detection for Multivariate Time Series},
    Booktitle = {Proceedings of the 8th IEEE International Conference on Data Science and Advanced Analytics},
    Address = {Virtual Event},
    Month = {October},
    Year = {2021}}
```

## Usage
You can perform UFEKT by importing the module, ***ufekt***, in your source code. Furthermore, you can also execute outlier detection task using the provided sample code using feature vectors obtained from UFEKT. Please see `sample.py` file. In our sceinario, k-th Nearest Neighbor algorithm is employed for outlier detection task. 

Below is an example of how to execute the sample code when multivariate time seires dataset is given as `datasets_sample1.csv`. In this case, the feature vectors obtaind from UFKET is output to `datasets_sample1_factors_1.csv` file.

```sh
$ python3 sample.py <dataset_filename> [ --max_rank | --min_rank | -w | -s | -k | --od ]
```

```
$ python3 sample.py datasets_sample1.csv
------------------------------------------------------------------------
file="datasets_sample1.csv", min_rank=10, max_rank=50, window_size=2, sigma=1.0, knn_k=5, od=False
Executing UFEKT
making a tensor, 0 1 optimizing ranks, 50
0
5
10
15
20
25
30
35
40
45
Tucker decomposition.
Saving the reuslts
Completed.
$ 
```

If you would like to perform outlier detection, please add ***-od*** option in the command line. The results would be output to a score file. If a threshold value for outlier detection is determined, you can identify outliers from the score file. Note that the k-th Nearest Neighbors algorithm is only employed for outlier detection in our sample code.
```
$ python3 sample.py datasets_sample1.csv -od
------------------------------------------------------------------------
file="datasets_sample1.csv", min_rank=10, max_rank=50, window_size=2, sigma=1.0, knn_k=5, od=True
Executing UFEKT
making a tensor, 0 1 optimizing ranks, 50
0
5
10
15
20
25
30
35
40
45
Tucker decomposition.
Saving the reuslts
kth-Nearest Neighbor for outlier detection
Saving the reuslts
Completed.
$ 
```

## A csv file format
Dataset must be given as a csv file. It must compose of $T$ by $P$ matrix where $T$ and $P$ indicate the length of time series and the number of vairables, respectively. Please see `datasets_sample1.csv` file as an example.

## Command-line arguments
Some values of parameters can be changed in the command-line options:
- `--max_rank`: set a maximum rank which is used for search range to find the best rank for outlier detection [default value: 50]
- `--min_rank`: set a minimum rank which is used for search range to find the best rank for outlier detection [default value: 10]
- `-w`: set a window size or the length of subsequence of time series [default value: 2]
- `-s`: set a sigma used in RBF kernel, $exp\{-\sum(x_i - x_j)^2/\sigma^2\}$, hence, do not set to zero [default value: 1.0]
- `-k`: set the number of "k" used in k-th Nearest Neighbor algorithm [default value: 5]
- `--od`: execute outlier detection using k-th Nearest Neighbor algorithm and feature vectors obtained from UFEKT

## Output
The sample code generates one or two csv files such as `datasets_sample1_factors_1.csv` and `datasets_sample1_scores.csv`, when `datasets_sample1.csv` is given as a dataset file. 
- The `datasets_sample1_factors_1.csv` include feature vectors obtained from UFEKT. Its size of a matrix would be $(T-w+1)$ by $R$, where $T$, $w$ and $R$ indicate the length of time series, a window size, and a rank, respectively. 
- The `datasets_sample1_scores.csv` includes scores of results of outlier detection using ***k-th Nearest Neighbor*** algorithm. The number of rows would be $T-w+1$. 

## Environment
We use Python 3.7.6 to execute our sample code and some packages are called in the code. Please check the packages and their versions below.

- numpy 1.18.4
- pandas 1.1.4
- scikit-learn 0.23.1
- tensorly 0.5.0

## Contact
- Author: Kiyotaka Matsue
- Affiliation: National Institute of Informatics, Tokyo, Japan  
- E-mail: matsue@nii.ac.jp


