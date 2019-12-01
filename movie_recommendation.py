#this is a prototype to see how Collabrative Filtering could work in the context with movies
#From this prototype, we want to find out,
# 1. how we can improve recommendation
# 2. how a program like this could run
# 3. get a better interpretation of the algorithm

import pandas as pd
import numpy as np
import math

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
movie_features = pd.merge(movie_ratings, users).pivot(index='user_id',columns='movie_id',values='rating').fillna(0)
mu_matrix = np.array(movie_features.values, dtype=int)

"""
Euclidean Distance
Determines the "Distance" or Simularity between 2 users by checking each of their common rankings and what they ranked them
Then returning the ratio of how similar they are
Parameters : the users we are comparing
"""
def euclidean_distance(user1,user2):
    #each user is an index in the movie_features dataset, check every movie rating and compare them
    # get the 2 users row, keeping in mind that each column is a movie and each value is the ranking of that movie
    # mu_matrix is the movie-user matrix, every row in the matrix (indexed like mu_matrix[x]) is a user and his ratings.
    # so each user is their index in the matrix :d
    # find all the mutual rankings and store the difference between those 2 rankings ^2
    mutual_rankings = [(mu_matrix[user1][i] - mu_matrix[user2][i])**2 for i in range(0,len(mu_matrix[user1])-1) if mu_matrix[user1][i] > 0 and mu_matrix[user2][i] > 0]
    return math.sqrt(sum(mutual_rankings))

def pearson_similarity(user1,user2):
    # get the mutually ranked as pairs (user1 ranking, user2 ranking)
    mutual_rankings1 = [mu_matrix[user1][i] for i in range(0,len(mu_matrix[user1])-1) if mu_matrix[user1][i] > 0 and mu_matrix[user2][i] > 0]
    mutual_rankings2 = [mu_matrix[user2][i] for i in range(0,len(mu_matrix[user1])-1) if mu_matrix[user1][i] > 0 and mu_matrix[user2][i] > 0]
    # get the variables needed for the algorithm
    n = len(mutual_rankings2) # both rankings SHOULD be the same size, so choose either to be n
    # sum of both users
    sum_x = sum([item for item in mutual_rankings1])
    sum_y = sum(mutual_rankings2)
    # sum of both users squared
    sum_x_sqr = sum(square(mutual_rankings1))
    sum_y_sqr = sum(square(mutual_rankings2))
    # sum of the products of paired rankings
    sum_xy = sum([mutual_rankings1[i] * mutual_rankings2[i] for i in range (0, n)])
    # we have everything we need, find the correlation coefficient!
    denom = math.sqrt((n * sum_x_sqr - (sum_x**2)) * (n * sum_y_sqr - (sum_y**2)))
    return ((n * sum_xy - (sum_x - sum_y))/denom) if denom != 0 else 0

def square(list):
    return [i**2 for i in list]
