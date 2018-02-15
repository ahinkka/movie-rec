"""

- Read in movies.
- Read in ratings and convert to a binary "watched" for each rating.
- Return a movie-to-id mapping (name to numerical id) and a binary matrix.
- The output of this script is a mapping from movies to user watches.

- Read only a subset of the data, i.e. --movie-count 10 and --user-count 10.

"""
import csv
import sys
import pickle
import collections

import numpy as np


def log(*args, **kwargs):
    kwargs['file'] = sys.stderr
    return print(*args, **kwargs)


def main(movies_file, ratings_file, watched_matrix_file, movie_count, user_count):
    movie_ids = {}
    movie_reader = csv.reader(movies_file)
    max_movie_id = -1
    for row in movie_reader:
        if row[0] == 'movieId':
            continue

        movie_id = int(row[0])

        if movie_count > 0 and movie_id > movie_count:
                break

        if movie_id > max_movie_id:
            max_movie_id = movie_id
        movie_ids[movie_id] = row[1]
    log("Read in movies")

    rating_reader = csv.reader(ratings_file)
    max_user_id = -1
    for idx, row in enumerate(rating_reader):
        if row[0] == 'userId':
            continue
        user_id = int(row[0])

        if user_count > 0 and user_id > user_count:
                break

        if user_id > max_user_id:
            max_user_id = user_id
    log("Resolved user count")

    arr = np.zeros((max_movie_id + 1, max_user_id + 1), dtype = 'bool')

    ratings_file.seek(0)
    for idx, row in enumerate(rating_reader):
        # log(row)
        if row[0] == 'userId':
            continue

        if idx % 100000 == 0:
            log("Read in {} ratings...".format(idx))


        user_id = int(row[0])
        movie_id = int(row[1])

        # log('m', movie_count, movie_id)
        # log('u', user_count, user_id)
        if movie_count > 0 and movie_id > movie_count:
            continue
        if user_count > 0 and user_id > user_count:
            break # ratings are ordered by userId

        arr[movie_id][user_id] = True
        # log()

    # np.delete(arr,3,axis=0) | Deletes row on index 3 of arr
    # np.delete(arr,4,axis=1) | Deletes column on index 4 of arr

    log("Removing the empty first row and first column...")
    arr = np.delete(arr, 0, axis=0)
    arr = np.delete(arr, 0, axis=1)

    log("Read in all ratings into a matrix of dimension {}".format(arr.shape))
    pickle.dump(arr, watched_matrix_file)
    log("Matrix written to {}".format(watched_matrix_file))


if __name__ == '__main__':
    import argparse
    from patharg import PathType
    parser = argparse.ArgumentParser()
    parser.add_argument('movies_file', type=argparse.FileType('r'))
    parser.add_argument('ratings_file', type=argparse.FileType('r'))
    parser.add_argument('watched_matrix_file', type=argparse.FileType('wb'))

    parser.add_argument('--movie-count', type=int, default=-1)
    parser.add_argument('--user-count', type=int, default=-1)

    args = parser.parse_args()
    main(args.movies_file, args.ratings_file, args.watched_matrix_file,
         args.movie_count, args.user_count)
