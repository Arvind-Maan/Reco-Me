#this is a prototype to see how Collabrative Filtering could work in the context with movies
#From this prototype, we want to find out,
# 1. how we can improve recommendation
# 2. how a program like this could run
# 3. get a better interpretation of the algorithm

import pandas as pd

"""
initialize the data set
"""
# user data 
# take the first 4 columns cause the 5th column is postal code, which we don't really care about.
user_cols = ['user_id', 'age', 'sex', 'occupation']
users = pd.read_csv('data/u.user', sep='|', names=user_cols, usecols=range(4), encoding='latin-1')

#ratings
rate_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('data/u.data', sep='\t', names=rate_cols, encoding='latin-1')

# the movies file contains columns indicating the movie's genres
# a plan to optimize is to perhaps change the columns we get. For example, the IMDB URL is not very important for RECOMMENDING
# Perhaps we can pass our information for the IMDB URL and others when we actually get a movie?
# for testing purposes, i am choosing the first 5 columns
movs_cols = ['movie_id', 'title', 'release_year', 'video_release_date', 'URL']
movies = pd.read_csv('data/u.item', sep='|', names=movs_cols, usecols=range(5), encoding='latin-1')

# create one merged DataFrame
movie_ratings = pd.merge(movies, ratings)
lens = pd.merge(movie_ratings, users)
most_rated = lens.groupby('title').size().sort_values(ascending=False)[:25]


filtered_movie_ratings = movie_ratings.drop(columns=['URL','video_release_date', 'release_year', 'timestamp'])
movie_features = filtered_movie_ratings.pivot(index='user_id',columns='movie_id',values='rating').fillna(0)
# At this point, we have a filtered movie DataFrame
# we want to iterate through this dataset and accumulate the ratings as one sole-rating.
from scipy.sparse import csr_matrix

arr_movie_features = csr_matrix(movie_features.values)

