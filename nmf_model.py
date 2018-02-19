import collections
import csv
import math
import operator
import os
import pickle
import random
import sys
import warnings

from datetime import datetime as dt

import numpy as np

from sklearn.decomposition import NMF
from scipy.spatial import distance
import matplotlib.pyplot as plt

import movielens_util


Model = collections.namedtuple('Model', ['movies', 'movieIdToIndex', 'W', 'H', 'V'])


def log(*args, **kwargs):
    kwargs['file'] = sys.stderr
    return print(*args, **kwargs)


def distances(embedding_matrix, index, distance_function=distance.cosine):
    # jaccard, euclidean, ...
    distances = []
    for i in range(embedding_matrix.shape[0]):
        if i == index:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            distances.append((i, distance_function(embedding_matrix[index],
                                                   embedding_matrix[i])))
    return distances


def nearest_neighbors(embedding_matrix, index, count):
    return sorted(distances(embedding_matrix, index),
                  key=operator.itemgetter(1))[0:count]


def build_model(movielens_dir, feature_count, user_count, movie_count):
    start = dt.now()

    movies = movielens_util.read_movies(movielens_dir)
    movieIdToIndex = {}
    for index, row in enumerate(movies.itertuples()):
        movieIdToIndex[row.movieId] = index

    ratings = movielens_util.read_ratings(movielens_dir, 'train')

    data_mangled = dt.now()
    log("Read in data, took {}s".format((data_mangled - start).total_seconds()))

    # log('Movies:')
    # log(movies.head())
    # log('Ratings:')
    # log(ratings.head())

    # W x H = V; V is the original matrix
    #  V is a user-movie matrix,
    #  W is a user-feature matrix, and
    #  H is a feature-movie matrix.

    V = np.zeros((ratings.userId.max(), len(movies)))
    log("V shape:", V.shape)
    for index, rating in enumerate(ratings.itertuples()):
        V[(rating.userId - 1, movieIdToIndex[rating.movieId])] = rating.rating
        # V[(rating.userId - 1, movieIdToIndex[rating.movieId])] = 1.0

    if user_count > -1 and movie_count > -1:
        log("Indexing V with :{}:, :{}:".format(user_count, movie_count))
        V = V[:user_count:, :movie_count:]
        log("V shape:", V.shape)

    included_movieIds = []
    for movie in movies.itertuples():
        if movieIdToIndex[movie.movieId] < movie_count:
            included_movieIds.append(movie.movieId)

    movies = movies[movies.movieId.isin(included_movieIds)]

    v_formed = dt.now()
    log("V formed in {}s".format((v_formed - data_mangled).total_seconds()))
    model = NMF(n_components=feature_count, init='random', random_state=0)
    log("Starting model fitting...")
    W = model.fit_transform(V)
    log("Model fitted in {}s".format((dt.now() - v_formed).total_seconds()))
    H = model.components_

    log("W shape: {} (user-feature)".format(W.shape))
    log("H shape: {} (feature-movie)".format(H.shape))
    log("V shape: {} (user-movie)".format(V.shape))

    m = Model(movies=movies, movieIdToIndex=movieIdToIndex, W=W, H=H, V=V)

    log('Model tuple contains the following types:')
    for key in m._asdict().keys():
        value = getattr(m, key)
        log('', key, type(value), sep='\t')

    return m


def evaluate(model, movielens_dir):
    train_ratings = movielens_util.read_ratings(movielens_dir, 'train')
    test_ratings = movielens_util.read_ratings(movielens_dir, 'test')
    inferred_V = np.dot(model.W, model.H)

    # Random
    squared_errors_sum = 0
    squared_errors_count = 0
    for i in range(inferred_V.shape[0]):
        for j in range(inferred_V.shape[1]):
            model_rating = inferred_V[(i-1, j-1)]
            # random_rating = random.choice([0.0, 1.0]) # binary
            random_rating = random.choice([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            squared_errors_sum += (model_rating - random_rating)**2
            squared_errors_count += 1
    mse = squared_errors_sum / squared_errors_count
    log('RMSE (random)', math.sqrt(mse), sep='\t')

    # First for train data
    user_size, movie_size = model.V.shape
    squared_errors_sum = 0
    squared_errors_count = 0
    for rating in train_ratings.itertuples():
        user_idx = rating.userId - 1
        movie_idx = model.movieIdToIndex[rating.movieId]

        if user_idx >= user_size or movie_idx >= movie_size:
            continue

        model_rating = inferred_V[(user_idx, movie_idx)]
        # test_rating = 1.0 # binary
        test_rating = rating.rating
        squared_errors_sum += (model_rating - test_rating)**2
        squared_errors_count += 1
    mse = squared_errors_sum / squared_errors_count
    log('RMSE (train)', math.sqrt(mse), sep='\t')

    # Then for test data
    user_size, movie_size = model.V.shape
    squared_errors_sum = 0
    squared_errors_count = 0
    for rating in test_ratings.itertuples():
        user_idx = rating.userId - 1
        movie_idx = model.movieIdToIndex[rating.movieId]

        if user_idx >= user_size or movie_idx >= movie_size:
            continue

        model_rating = inferred_V[(user_idx, movie_idx)]
        # test_rating = 1.0 # binary
        test_rating = rating.rating
        squared_errors_sum += (model_rating - test_rating)**2
        squared_errors_count += 1
    mse = squared_errors_sum / squared_errors_count
    log('RMSE (test)', math.sqrt(mse), sep='\t')
    

def similar(model, vis=False):
    movie_feature_embedding_matrix = np.transpose(model.H)
    movie = model.movies.sample(1)
    log()
    log(movie.iloc[0].title, '\t', movie.iloc[0].genres)
    from_index = movie.index.data[0]

    vis_vecs = []
    vis_vecs.append(movie_feature_embedding_matrix[from_index])
    for index, distance in nearest_neighbors(movie_feature_embedding_matrix, from_index, 10):
        m = model.movies.iloc[index]
        log("", m.title, m.genres, distance, sep='\t')
        vis_vecs.append(movie_feature_embedding_matrix[index])

    if vis:
        arr = np.array(vis_vecs)
        imshow = plt.imshow(arr)
        plt.title('Similar movies; first one is the one computing similarities for')
        # set the limits of the plot to the limits of the data
        # plt.axis([x.min(), x.max(), y.min(), y.max()])
        plt.colorbar(imshow)
        plt.show()


def sample_embeddings(model, embedding_count=250):
    movie_feature_embedding_matrix = np.transpose(model.H)
    movie = model.movies.sample(1)

    embeddings = movie_feature_embedding_matrix[
        np.random.randint(movie_feature_embedding_matrix.shape[0], size=embedding_count), :]
    embeddings = np.log(embeddings)

    arr = np.array(embeddings)
    imshow = plt.imshow(arr)
    plt.title('Movie embeddings, sample of {}, values log()'.format(embedding_count))
    plt.colorbar(imshow)
    plt.show()


if __name__ == '__main__':
    import argparse, patharg
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')

    build_parser = subparsers.add_parser('build', help='build help')
    build_parser.add_argument('movielens_dir', type=patharg.PathType(exists=True, type='dir'))
    build_parser.add_argument('model_file', type=argparse.FileType('xb'))
    build_parser.add_argument('--user-count', type=int, default=-1)
    build_parser.add_argument('--movie-count', type=int, default=-1)
    build_parser.add_argument('--feature-count', type=int, default=150)

    similar_parser = subparsers.add_parser('similar', help='similar help')
    similar_parser.add_argument('model_file', type=argparse.FileType('rb'))

    similar_vis_parser = subparsers.add_parser('similar-vis', help='similar-vis help')
    similar_vis_parser.add_argument('model_file', type=argparse.FileType('rb'))

    sample_embeddings_parser = \
      subparsers.add_parser('sample-embeddings', help='similar- help')
    sample_embeddings_parser.add_argument('model_file', type=argparse.FileType('rb'))

    evaluate_parser = subparsers.add_parser('evaluate', help='evaluate help')
    evaluate_parser.add_argument('model_file', type=argparse.FileType('rb'))
    evaluate_parser.add_argument('movielens_dir', type=patharg.PathType(exists=True, type='dir'))

    args = parser.parse_args()
    if args.command == 'build':
        model = build_model(args.movielens_dir,
                            args.feature_count, args.user_count, args.movie_count)
        log('Writing model to file {}'.format(args.model_file))
        pickle.dump(model, args.model_file)
        os.fsync(args.model_file.fileno())
    elif args.command == 'evaluate':
        model = pickle.load(args.model_file)
        evaluate(model, args.movielens_dir)
    elif args.command == 'similar':
        model = pickle.load(args.model_file)
        similar(model)
    elif args.command == 'similar-vis':
        model = pickle.load(args.model_file)
        similar(model, vis=True)
    elif args.command == 'sample-embeddings':
        model = pickle.load(args.model_file)
        sample_embeddings(model)
    else:
        parser.error('unknown command "{}"'.format(args.command))
