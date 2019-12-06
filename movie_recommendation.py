#this is a prototype to see how Collabrative Filtering could work in the context with movies
#From this prototype, we want to find out,
# 1. how we can improve recommendation
# 2. how a program like this could run
# 3. get a better interpretation of the algorithm

import pandas as pd
import numpy as np
import math
import random

# user data 
# take the first 4 columns cause the 5th column is postal code, which we don't really care about.
user_cols = ['user_id', 'age', 'sex', 'occupation']
users = pd.read_csv('data/u.user', sep='|', names=user_cols, usecols=range(4), encoding='latin-1')

#ratings
rate_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('data/u.data', sep='\t', names=rate_cols, encoding='latin-1')

# the movies file contains columns indicating the movie's genres
# a plan to optimize is to perhaps change the columns we get.
# For example, the IMDB URL is not very important for RECOMMENDING
# Perhaps we can passgi our information for the IMDB URL and others when we actually get a movie?
# for testing purposes, i am choosing the first 5 columns
movs_cols = ['movie_id', 'title', 'release_year', 'video_release_date', 'URL']
movies = pd.read_csv('data/u.item', sep='|', names=movs_cols, usecols=range(5), encoding='latin-1')

# create one merged DataFrame
movie_ratings = pd.merge(movies, ratings)
movie_features = pd.merge(movie_ratings, users).pivot(index='user_id',columns='movie_id',values='rating').fillna(0)
mu_matrix = np.array(movie_features.values, dtype=int)


"""
add_user
add yourself to the matrix
"""
def add_user():
    print("\n\nIn order for us to accurately recommend you a movie, we need to get an understanding of some of the movies you like.")
    print("You must rank at least 10 movies so we can get an understanding of your taste :)")

    # we need to create an array of rankings
    user_row = [0 for i in range(0, len(mu_matrix[0]))]
    num = 0 # number of current movies ranked
    while num < 10:
        # pick a random movie and make the user rank it. WE NEED 10 rankings.
        movie_index = random.randint(0,len(mu_matrix[0])-1)
        if(user_row[movie_index] <= 0):
            movie_x = (movies.fillna(-1)).loc[movie_index] #get the movie summary
            print("~~~What do you think of this movie?~~~")
            #print the movie
            to_string = "Movie \n\t"
            for val in movie_x.values:
                if val != -1: # ignore nan values
                    to_string += str(val) + " | " 
            print(to_string)
            #if the users answer is valid
            valid = False
            while not valid: #until we get a valid answer, we cannot continue
                x = input("Please enter a rating 1-5, 0 if no opinion: ")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                if( x.isdigit() and int(x) >= 0 and int(x) <= 5):
                    user_row[movie_index] = int(x)
                    num += 1 if int(x) != 0 else 0 # if its 0, we don't increment the ratings we've gotten
                    valid = True
                else: 
                    print("~Invalid Input~")
                    valid = False
    # we have to return the updated array
    # since if we try to change mu_matrix in this method we will get an error.
    # not this does not add the user to our user database or rating database. this is a bandaid fix and only stores the user for runtime
    convert_row = [[item for item in user_row]]
    copy_matrix = np.append(mu_matrix, convert_row, axis = 0)
    return copy_matrix
    

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
    common_rankings = [(mu_matrix[user1][i] - mu_matrix[user2][i])**2 for i in range(0,len(mu_matrix[user1])-1) if mu_matrix[user1][i] > 0 and mu_matrix[user2][i] > 0]
    return 1 / (1 + sum(common_rankings))

"""
Pearson Simularity
returns the value of the correlation coefficient in the pearson simularity algorithm
Parameters: the users we are comparing
"""
def pearson_similarity(user1,user2):
    # get the mutually ranked
    common_rankings1 = [mu_matrix[user1][i] for i in range(0,len(mu_matrix[user1])-1) if mu_matrix[user1][i] > 0 and mu_matrix[user2][i] > 0]
    common_rankings2 = [mu_matrix[user2][i] for i in range(0,len(mu_matrix[user1])-1) if mu_matrix[user1][i] > 0 and mu_matrix[user2][i] > 0]
    # get the variables needed for the algorithm
    n = len(common_rankings2) # both rankings SHOULD be the same size, so choose either to be n
    # sum of both users squared
    sum_x_sqr = sum(square(common_rankings1))
    sum_y_sqr = sum(square(common_rankings2))
    # sum of the products of paired rankings
    sum_xy = sum([common_rankings1[i] * common_rankings2[i] for i in range (0, n)])
    # we have everything we need, find the correlation coefficient!
    denom = math.sqrt((n * sum_x_sqr - (sum(common_rankings1)**2)) * (n * sum_y_sqr - (sum(common_rankings2)**2)))
    return ((n * sum_xy - (sum(common_rankings1) - sum(common_rankings2)))/denom) if denom != 0 else 0

"""
Cosine Similarity
computes the cosine simularity function on the users and returns the value
PArameters: the users we are comparing
"""
def cosine_similarity(user1, user2):
    # get the mutually ranked
    common_rankingsA = [mu_matrix[user1][i] for i in range(0,len(mu_matrix[user1])-1) if mu_matrix[user1][i] > 0 and mu_matrix[user2][i] > 0]
    common_rankingsB = [mu_matrix[user2][i] for i in range(0,len(mu_matrix[user1])-1) if mu_matrix[user1][i] > 0 and mu_matrix[user2][i] > 0]
    #get n
    n = len(common_rankingsA)
    #sum of a*b
    sum_ab =  sum([common_rankingsA[i] * common_rankingsB[i] for i in range (0, n)])
    #sum of a and b squared
    sum_a_sqr = sum(square(common_rankingsA))
    sum_b_sqr = sum(square(common_rankingsB))
    # get denominator so we can check division by 0
    denom = (math.sqrt(sum_a_sqr)*math.sqrt(sum_b_sqr))
    return (sum_ab / denom) if denom != 0 else 0


"""
Jaccard Similarity
computes the jaccard simularity function on the users and returns the value
PArameters: the users we are comparing
"""
def jaccard_similarity(user1, user2):
    # jacard similarity is the intersection of the 2 / the union of the 2 
    # in other words, the common rankings / all ranked
    # we just care about length, so don't worry about maintaing indices
    union_ab = []
    intersection_ab = []
    for i in range (0, len(mu_matrix[user1]-1)):
        if mu_matrix[user1][i] > 0 and mu_matrix[user2][i] > 0:
            intersection_ab.append(mu_matrix[user1][i])
        else: # can't be both
            if mu_matrix[user1][i] > 0:
                union_ab.append(mu_matrix[user1][i])
            elif mu_matrix[user2][i] > 0:
                union_ab.append(mu_matrix[user2][i])    
    # typically jaccard only factors in set difference
    # for the sake of the movie example, we will use the actual ratings to see if this makes it more accurate
    return 1 - (sum(intersection_ab)/sum(union_ab)) if sum(union_ab) != 0 else 0

"""
Square
returns the list with every single value square'd
parameters: the list 
"""
def square(list):
    return [i**2 for i in list]

"""
recommend
Recommends [k] number of movies using [fn] with [depth] as the depth and on user [user].
Parameters:
    user: the user we are recommending to
    depth: how many users are we comparing to this user
    fn: the function used for calculating simularity and distance
    k: the number of recommendations 
"""
def recommend (user, clusters, fn, k):
    print("############ RECOMMENDATION FOR USER %d ############" %(user))
    print("Using %s for calculating similarity/distance with %d clusters" %(fn,clusters))
    # we want to use our simularity functions to develop a list of ratings that our user might rate a movie as.
    # this allows us to think what our user will like the most in comparison
    distances = simularity_matrix(user,fn,clusters)
    #for each distance 
    all_unseen_movies = {}
    for dist,o in distances: 
        # each movie is an index, the number of movies is the LENGTH of the items row
        for i in range(0, len(mu_matrix[o])-1):
            predicted_rank = int(dist*mu_matrix[o][i])
            # update recommendations i to be the value of the similarity predicted rank 
            if (mu_matrix[o][i] > 0 and mu_matrix[user][i] <= 0): #if they ranked it but we didn't
                # if we are already planning to recommend it, add more weight to the prediction and increase similarity!
                all_unseen_movies[i] = (dist, [predicted_rank]) if i not in all_unseen_movies else (all_unseen_movies[i][0] + dist, all_unseen_movies[i][1] + [predicted_rank])   
    # update all the values
    for m in all_unseen_movies: 
        all_unseen_movies[m] = sum(all_unseen_movies[m][1])/all_unseen_movies[m][0] if all_unseen_movies[m][0] != 0 else 0 
    # at this point we have a dictionary of all possible recommendations with the format:
    # { [movie_id] : [predicted_rating]}
    # lets pick the top 5!
    recommend_movies = filter_results(all_unseen_movies,k)
    # recommend movies outputs an array formatted like so: [ (movie_id, predicted rank)]
    # lets print it in a nice way:
    print_recommendations(recommend_movies)
    print("############ END OF RECOMMENDATION FOR USER %d ############" %(user))
    return recommend_movies

"""
simularity matrix
    returns a matrix of users that are similar to our user and how similar they are.
    [ (user similarity, user), (user similarity, user), .... , ]
    usings [fn] to calculate simularity
    and limits the matrix to size clusters
"""
def simularity_matrix(user, fn, clusters):
    # get the distances 
    # sort and reverse the data (closest -> least closest) and attach a bound 
    toReturn = [(fn(user, other),other) for other in range(0,len(mu_matrix)-1) if other != user]
    toReturn.sort(reverse=True)
    toReturn = toReturn[0:clusters] 
    return toReturn

import operator,collections

"""
filter_results
sorts all the recommended movies and returns an array of the top (num_of_recommendations)
parameters:
    all_outputs: a dictionary consisting of ALL recommendations
    num_of_recommendations: the number of recommendations we want
"""
def filter_results(all_outputs, num_of_recommendations):
    # sort the recommedations by possible rankings
    sorted_recommendations = sorted(all_outputs.items(), key=operator.itemgetter(1),reverse=True)
    # return the top 10
    sorted_recommendations = sorted_recommendations[0:num_of_recommendations]
    return sorted_recommendations

"""
print_recommendations 
Prints all the recommendations passed
Parameters:
    to_recommend: the list of recommendations to print
"""
def print_recommendations(to_recommend):
    i = 1 # What recommendation is this, starts at 1.
    for x,y in to_recommend: # for every recommendation
        movie_x = (movies.fillna(-1)).loc[x-1] #get the row: IGNORE nan values. the reason it's x-1 is because the index is the movie_id-1 since movie_id starts at 1 not 0 and index starts at 0 not 1
        print("----Recommendation %d----" %(i))
        to_string = ""
        for val in movie_x.values:
            if val != -1: # ignore nan values
                to_string += str(val) + " | " 
        print(to_string)
        print("Predicted Rating: %f" %(y))  
        print("-------------------------")
        i += 1 #increment the recommendation
    return

"""
print_user_summary
Prints the users top 5 movies
Parameters:
    the user
"""
def print_user_summary(user):
    # the header
    print("*****USER SUMMARY FOR USER %d*****" %(user))
    user_x = (users.fillna(-1)).loc[user-1]
    print(user_x)
    print("==Users Top 5 Favourite Movies==")
    # for each movie, get the most ranked and sort them by their rank. then pick the top 5
    # this is a slightly inaccurate representation of favourite movies, a more accurate description would be:
    #   5 of this users top movies
    movies_seen = [(mu_matrix[user][x],x) for x in range (0,len(mu_matrix[user])-1)]
    movies_seen.sort()
    movies_seen.reverse()
    movies_seen = movies_seen[0:5]
    i = 1 # movie index
    # for each movie, print a formatted description
    for rating,movie in movies_seen:
        movie_x = (movies.fillna(-1)).loc[movie-1]
        to_string = "Movie #%d\n\t" %(i)
        for val in movie_x.values:
            if val != -1: # ignore nan values
                to_string += str(val) + " | " 
        print(to_string + "User Ranking: %d" %(rating))
        i += 1 #increment the movie
    # print the footer
    print("*********************************\n")
    return #end


# sample entries 
print("\n\n____________________Welcome to Project: Reco-ME____________________")
print("You have the option between recommending a movie to yourself or to an existing user in the database")
answer = input("Would you like to recommend to yourself? Please enter (Y/N): ")
if(answer == "Y" or answer == "y"):
    mu_matrix = add_user()
    user = len(mu_matrix)-1
else:
    user = int(input("Please enter the ID of the existing user you'd like to recommend to [Note: Max is 943](Y/N): "))

print("Please enter the number of clusters you'd like to test with: (a good one to try is 10)")
bound = int(input("[Note: Max is 943]"))

k = 5

print("Recommendations with Euclidean Distance")
recommend(user, bound, euclidean_distance,k)
print("\nRecommendations with Jaccard Similarity")
recommend(user, bound, jaccard_similarity,k)
print("\nRecommendations with Pearson Similarity")
recommend(user, bound, pearson_similarity,k)
print("\nRecommendations with Cosine Similarity")
recommend(user, bound, cosine_similarity,k)
print("\n\n____________________END OF SIMULATION____________________")