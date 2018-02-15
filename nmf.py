import csv
import sys
import pickle

import numpy as np

from sklearn.decomposition import NMF


def log(*args, **kwargs):
    kwargs['file'] = sys.stderr
    return print(*args, **kwargs)


def main(watched_matrix_file, feature_count):
    log("Reading matrix from {}".format(watched_matrix_file))
    arr = pickle.load(watched_matrix_file)
    log("Read in a matrix of dimension {} (movies, users)".format(arr.shape))
    arr = arr.transpose()
    log("Transpose the matrix to {} (users, movies)".format(arr.shape))
    # log(arr)
    # log(arr.shape)

    # W x H = V; V is the original matrix
    #  V is a user-movie matrix,
    #  W is a user-feature matrix, and
    #  H is a feature-movie matrix.

    model = NMF(n_components=feature_count, init='random', random_state=0)
    W = model.fit_transform(arr)
    H = model.components_

    log("W shape: {}".format(W.shape))
    log("H shape: {}".format(H.shape))

    for user in range(arr.shape[0]):
        parts = []
        for feature in range(feature_count):
            parts.append(str(W[user][feature]))
        log('\t'.join(parts))


if __name__ == '__main__':
    import argparse
    from patharg import PathType
    parser = argparse.ArgumentParser()
    parser.add_argument('watched_matrix_file', type=argparse.FileType('rb'))

    args = parser.parse_args()
    main(args.watched_matrix_file, 5)
