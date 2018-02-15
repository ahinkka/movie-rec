import collections
import csv
import operator
import os
import pickle
import sys

from datetime import datetime as dt

import numpy as np

from sklearn.decomposition import NMF
from scipy.spatial import distance

import movielens_util


Model = collections.namedtuple('Model', ['movies', 'movieIdToIndex', 'W', 'H', 'V'])


def log(*args, **kwargs):
    kwargs['file'] = sys.stderr
    return print(*args, **kwargs)


def distances(embedding_matrix, index):
    distances = []
    for i in range(embedding_matrix.shape[0]):
        if i == index:
            continue
        # log(index, i)
        # log('\t', embedding_matrix[index])
        # log('\t', embedding_matrix[i])
        # distances.append((i, distance.euclidean(embedding_matrix[index],
        #                                         embedding_matrix[i])))
        distances.append((i, distance.cosine(embedding_matrix[index],
                                             embedding_matrix[i])))
        # distances.append((i, distance.jaccard(embedding_matrix[index],
        #                                       embedding_matrix[i])))
    return distances


def nearest_neighbors(embedding_matrix, index, count):
    return sorted(distances(embedding_matrix, index),
                  key=operator.itemgetter(1))[0:count]


def build_model(movielens_dir, training_percentage, feature_count, user_count, movie_count):
    total_user_count = 138492 # TODO: make this not fixed...
    start = dt.now()

    movies = movielens_util.read_movies(movielens_dir)
    movieIdToIndex = {}
    for index, row in enumerate(movies.itertuples()):
        movieIdToIndex[row.movieId] = index

    ratings = movielens_util.read_ratings(movielens_dir,
                                          skip_percentage=0,
                                          include_percentage=training_percentage)

    # make ratings binary
    rating_mapping = { 0.0: 0,
                       1.0: 1, 2.0: 1, 3.0: 1, 4.0: 1, 5.0: 1, }
    ratings['rating'] = ratings['rating'].map(rating_mapping)

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

    V = np.zeros((total_user_count, len(movies)))
    log("V shape:", V.shape)
    for index, rating in enumerate(ratings.itertuples()):
        V[(rating.userId - 1, movieIdToIndex[rating.movieId])] = 1.0

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


def recommend(model_file):
    model = pickle.load(model_file)

    # log("W shape: {} (user-feature)".format(model.W.shape))
    # log("H shape: {} (feature-movie)".format(model.H.shape))
    # log("V shape: {} (user-movie)".format(model.V.shape))

    movie_feature_embedding_matrix = np.transpose(model.H)
    # log(movie_feature_embedding_matrix.shape)
    # log(model.movies)
    # return
    # log(movie_feature_embedding_matrix)
    movie = model.movies.sample(1)
    log()
    log(movie.iloc[0].title, '\t', movie.iloc[0].genres)
    index = movie.index.data[0]
    for index, distance in nearest_neighbors(movie_feature_embedding_matrix, index, 10):
        m = model.movies.iloc[index]
        log("", m.title, m.genres, distance, sep='\t')


if __name__ == '__main__':
    import argparse, patharg
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')

    build_parser = subparsers.add_parser('build', help='build help')
    build_parser.add_argument('movielens_dir', type=patharg.PathType(exists=True, type='dir'))
    build_parser.add_argument('model_file', type=argparse.FileType('xb'))
    build_parser.add_argument('--training-set-percentage', type=int, default=80)
    build_parser.add_argument('--user-count', type=int, default=-1)
    build_parser.add_argument('--movie-count', type=int, default=-1)
    build_parser.add_argument('--feature-count', type=int, default=150)

    query_parser = subparsers.add_parser('query', help='query help')
    query_parser.add_argument('model_file', type=argparse.FileType('rb'))
    query_parser.add_argument('function', choices=['recommend'])
    args = parser.parse_args()

    if args.command == 'build':
        model = build_model(args.movielens_dir, args.training_set_percentage,
                            args.feature_count, args.user_count, args.movie_count)
        log('Writing model to file {}'.format(args.model_file))
        pickle.dump(model, args.model_file)
        os.fsync(args.model_file.fileno())
    elif args.command == 'query':
        recommend(args.model_file)
    else:
        parser.error('unknown command')
