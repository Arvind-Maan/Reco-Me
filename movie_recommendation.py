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
lens = pd.merge(movie_ratings, users)

movie_features = lens.pivot(index='user_id',columns='movie_id',values='rating').fillna(-1)

mu_matrix = np.array(movie_features.values, dtype=int)
def euclidean_distance(user1, user2):
    #each user is an index in the movie_features dataset, check every movie rating and compare them
    user1_row = mu_matrix[user1]
    user2_row = mu_matrix[user2]
    # find all the mutual rankings and add them as a pair where (user1 rank, user2 rank)
    distance = []
    for i in range(0,len(user1_row)):
        if(user1_row[i] > 0 and user2_row[i] > 0):
            distance.append((user1_row[i] - user2_row[i])**2)

    return 1 / (1 + sum(distance))


min_val = -1
for i in range(0,943):
    if(euclidean_distance(15,i) > min_val and i != 15):
        min_user = i
        min_val = euclidean_distance(15,i)

print("The Closest User to User 1 is: User " + str(min_user) + " with value: " + str(min_val))
