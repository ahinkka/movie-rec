import math
import os

import pandas as pd


# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html


def read_movies(movielens_dir):
    result = pd.read_csv(os.path.join(movielens_dir, 'movies.csv'))
    result['movieId'] = result['movieId'].apply(pd.to_numeric)
    return result


def read_ratings(movielens_dir, skip_percentage=0, include_percentage=100):
    ratings_file = os.path.join(movielens_dir, 'ratings.csv')
    ratings_mtime = os.stat(ratings_file).st_mtime

    # don't do this if the file is untouched; useless to count lines
    if ratings_mtime == 1427836442.0:
        line_count = 20000264
    else:
        line_count = 0
        with open(ratings_file, 'r') as f:
            for line in f:
                line_count += 1

    skip_rows = math.floor(skip_percentage * 0.01 * line_count)
    take_rows = math.floor(include_percentage * 0.01 * line_count)
    result = pd.read_csv(ratings_file,
                         skiprows=skip_rows, nrows=take_rows)
    result['movieId'] = result['movieId'].apply(pd.to_numeric)
    result['userId'] = result['userId'].apply(pd.to_numeric)
    result['rating'] = result['rating'].apply(pd.to_numeric)
    return result
