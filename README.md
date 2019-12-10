# Reco-Me
How to Run:
If python3 is already installed simply run:
  python3 movie_recommendation.py
and follow the user prompts :)

This project was built for the class Introduction to Artifical Intelligence at the UofM.
We had to choose a machine learning / artificial intelligence problem and provide a solution to it, my problem was movie recommendation with collabrative filtering as the solution to it.

The project gave freedom for which language you wanted to do it in, so I used this as an opportunity to learn Python. 

The project implements several different ways of calculating simularity and distance:
- Euclidean Distance
- Pearson Similarity
- Cosine Similarity
- Jaccard Distance

We use those methods to calculate a user's "distance" from another user.

Taking the closest users to our user, we predict possible movie ratings for movies we haven't seen using a 
weighted value of the opposing user's rating.

This allows us to guess what we would rate every movie that we haven't yet seen, then we order this and return a top 5!

Libraries Used:
- Built in Libraries: (math, collections, operator)
- Pandas
- Numpy

How to download Pandas:
- https://pandas.pydata.org/getpandas.html
  
How to download Numpy: 
- https://docs.scipy.org/doc/numpy/user/install.html
