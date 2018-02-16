import math
import pickle
import os
import shutil
import sys

import pandas as pd

RANDOM_STATE = 1101


def log(*args, **kwargs):
    kwargs['file'] = sys.stderr
    return print(*args, **kwargs)


def read_movies(movielens_dir):
    result = pd.read_csv(os.path.join(movielens_dir, 'movies.csv'))
    result['movieId'] = result['movieId'].apply(pd.to_numeric)
    return result


def read_ratings(movielens_dir, train_or_test=None):
    if train_or_test == 'train':
        train_file = os.path.join(movielens_dir, 'ratings-train.pickle')
        with open(train_file, 'rb') as f:
            return pickle.load(f)
    elif train_or_test == 'test':
        test_file = os.path.join(movielens_dir, 'ratings-test.pickle')
        with open(test_file, 'rb') as f:
            return pickle.load(f)

    ratings_file = os.path.join(movielens_dir, 'ratings.csv')
    result = pd.read_csv(ratings_file)
    result['movieId'] = result['movieId'].apply(pd.to_numeric)
    result['userId'] = result['userId'].apply(pd.to_numeric)
    result['rating'] = result['rating'].apply(pd.to_numeric)
    return result


def main():
    import argparse, patharg
    parser = argparse.ArgumentParser()
    parser.add_argument('movielens_dir', type=patharg.PathType(exists=True, type='dir'))
    args = parser.parse_args()

    train_file = os.path.join(args.movielens_dir, 'ratings-train.pickle')
    test_file = os.path.join(args.movielens_dir, 'ratings-test.pickle')
    if os.path.exists(train_file):
        parser.error('{} exists, not clobbering'.format(train_file))
    if os.path.exists(test_file):
        parser.error('{} exists, not clobbering'.format(test_file))

    log('Reading in ratings...')
    all_ratings = read_ratings(args.movielens_dir)
    log('Read {} ratings'.format(all_ratings.size))

    train = all_ratings.sample(frac=0.8, random_state=RANDOM_STATE)
    test = all_ratings.drop(train.index)
    log('Split into training set with {} ratings and test set of {} ratings'
        .format(train.size, test.size))

    with open(train_file, 'xb') as f:
        pickle.dump(train, f)
        log('Training ratings saved into {}'.format(train_file))
    with open(test_file, 'xb') as f:
        pickle.dump(test, f)
        log('Testing ratings saved into {}'.format(test_file))


if __name__ == '__main__':
    main()
