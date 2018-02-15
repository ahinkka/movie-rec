import collections
import csv
import operator
import os
import pickle
import sys

from datetime import datetime as dt

import numpy as np
import pandas as pd

from sklearn.decomposition import NMF
from scipy.spatial.distance import cosine as cosine_distance
from scipy.spatial.distance import jaccard as jaccard_distance


Model = collections.namedtuple('Model', ['movies', 'W', 'H', 'V'])


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
        distances.append((i, cosine_distance(embedding_matrix[index],
                                             embedding_matrix[i])))
        # distances.append((i, jaccard_distance(embedding_matrix[index],
        #                                       embedding_matrix[i])))
    return distances


def nearest_neighbors(embedding_matrix, index, count):
    return sorted(distances(embedding_matrix, index),
                  key=operator.itemgetter(1), reverse=True)[0:count]


def build_and_write(movies_file, ratings_file, model_file,
                    movie_count, user_count, feature_count):
    start = dt.now()
    all_movies = pd.read_csv(movies_file)
    all_movies['movieId'] = all_movies['movieId'].apply(pd.to_numeric)
    if movie_count != -1:
        movies = all_movies.head(movie_count)
    else:
        movies = all_movies
    include_movies = list(movies['movieId'])

    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    all_ratings = pd.read_csv(ratings_file)
    if user_count != -1:
        include_users = list(np.sort(all_ratings['userId'].unique())[0:user_count])
    else:
        include_users = list(np.sort(all_ratings['userId'].unique()))

    ratings = all_ratings[all_ratings.movieId.isin(include_movies)]
    ratings = ratings[ratings.userId.isin(include_users)]

    # Because we don't include all users, some movies are completely without
    # ratings and hence we end up without these movies; filter out movies with
    # no ratings. We should probably include them, but for now it's like this.
    movies = movies[movies.movieId.isin(ratings.movieId.unique())]
    log(movies)

    # make ratings binary
    rating_mapping = { 0.0: 0,
                       1.0: 1, 2.0: 1, 3.0: 1, 4.0: 1, 5.0: 1, }
    ratings['rating'] = ratings['rating'].map(rating_mapping)

    data_mangled = dt.now()
    log("Read in data, took {}s".format((data_mangled - start).total_seconds()))
    log('Movies:')
    log(movies.head())
    log('Ratings:')
    log(ratings.head())

    # W x H = V; V is the original matrix
    #  V is a user-movie matrix,
    #  W is a user-feature matrix, and
    #  H is a feature-movie matrix.

    V = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    log("Data massaged into NMF format, took {}s".format((dt.now() - data_mangled).total_seconds()))
    log("NMF format:")
    log(V.head())

    model = NMF(n_components=feature_count, init='random', random_state=0)
    W = model.fit_transform(V)
    H = model.components_

    log("W shape: {} (user-feature)".format(W.shape))
    log("H shape: {} (feature-movie)".format(H.shape))
    log("V shape: {} (user-movie)".format(V.shape))

    m = Model(movies=movies, W=W, H=H, V=V)
    log('Writing model to file {}'.format(model_file))
    pickle.dump(m, model_file)
    os.fsync(model_file.fileno())

    log('Model tuple contains the following types:')
    for key in m._asdict().keys():
        value = getattr(m, key)
        log('', key, type(value), sep='\t')


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
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')

    build_parser = subparsers.add_parser('build', help='build help')
    build_parser.add_argument('movies_file', type=argparse.FileType('r'))
    build_parser.add_argument('ratings_file', type=argparse.FileType('r'))
    build_parser.add_argument('model_file', type=argparse.FileType('xb'))
    build_parser.add_argument('--movie-count', type=int, default=-1)
    build_parser.add_argument('--user-count', type=int, default=-1)
    build_parser.add_argument('--feature-count', type=int, default=50)

    query_parser = subparsers.add_parser('query', help='query help')
    query_parser.add_argument('model_file', type=argparse.FileType('rb'))
    query_parser.add_argument('function', choices=['recommend'])
    args = parser.parse_args()

    if args.command == 'build':
        build_and_write(args.movies_file, args.ratings_file, args.model_file,
                        args.movie_count, args.user_count, args.feature_count)
    elif args.command == 'query':
        recommend(args.model_file)
    else:
        parser.error('unknown command')
