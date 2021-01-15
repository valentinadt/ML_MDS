#!/usr/bin/env python
# coding: utf-8

# * Authors: Andrea Jiménez Zuñiga, Valentina Díaz Torres e Isabel Afán de Ribera Olaso
# * Date: 15/01/2020
# * Institution: CUNEF

# # 05. Collaborative Filtering
# 

# The collaborative filtering methodology is applied to the recommendation systems to optimize their operation and mitigate the problems of information that can be generated in a digital environment. Internet users can obtain millions of data, but only a few are of interest to them, and this can generate negative experiences and significant loss of time. Thanks to collaborative filtering, valuable information is selected, processed and built from this information, a set of suggestions and recommendations that are in accordance with user expectations.
# 
# 
# Here we are going to use the Memory-Based Collaborative filtering to make recommendations to movie users. The idea is based on the fact that users that are similar to me can be used to predict how much I would like a movie that they have liked and I have not watched. 
# 
# There are two types of Collaborative Filtering: 
# 
# 1. __User-User Collaborative Filtering:__ Its aim is to find user's look-alike.
# 
# 2. __Item-Item Collaborative Filtering:__ Its aim is to find movie's look-alike.
# 
# 

# ![Captura%20de%20pantalla%202021-01-07%20a%20las%202.15.15%20p.%C2%A0m..png](attachment:Captura%20de%20pantalla%202021-01-07%20a%20las%202.15.15%20p.%C2%A0m..png)

# If we either proceed to do a user-user collaborative filtering or an item-item one, we need to build a similarity matrix. For the first one, this matrix will consist of distance metrics that measure the similarity between any two pair of users. On the other hand, for the item-similarity matrix, this will measure the similarity between any two pair of items.
# 
# This distance similarity metrics are 3: 
# 
# 1. ___Jaccard Similarity___
# 2. ___Cosine Similarity___
# 3. ___Pearson Similarity___
# 
# In this case we are going to use the **Pearson Similarity**, which its similarity refers to the Pearson coefficient between the two vectors.

# ## Import Libraries
# 

# In[1]:


import pandas as pd 
import numpy as np


# In this case we only need two datasets: _movies.csv_ and _ratings.csv_.

# In[2]:


ratings = pd.read_csv('../../data/ratings.csv')
movies = pd.read_csv('../../data/movies.csv')


# In[3]:


# We are just interested in the ratings and users so we drop genres and timestamp

ratings = pd.merge(movies,ratings).drop(['genres','timestamp'], axis = 1)


# In[5]:


ratings.head()


# In[4]:


ratings = ratings.head(10000000)


# In[5]:


# Now we use pivot method in pandas. In the values for each column we want the ratings that each user gives 
# to a particular movie.

user_ratings = ratings.pivot_table(index = ['userId'], columns = ['title'],
                                  values = 'rating')
user_ratings.head()


# There are a lot of NaN values so we need to make a decision. We should drop a few movies from our dataframe which dont have a lot of users, as it might create noise in our system. As a result, we are going to drop those movies that have less than 10 users.
# 

# In[6]:


user_ratings = user_ratings.dropna(thresh = 10, axis = 1).fillna(0)
user_ratings.head() # See with how many movies we are left with (# of columns)


# Now we are going to build our similarity matrix. There are 3 methods: Jaccard Similarity, Cosine Similarity and Pearson Similarity. In this case we are going to use Pearson Similarity, being such similarity the coefficient between the 2 vectors.
# 

# In[7]:


item_similarity_df = user_ratings.corr(method = 'pearson')
item_similarity_df.head(50)


# * __Top Action Movie Recommendations:__

# Now we make recommendations based on the model that we have created: 

# In[8]:


# To make recommendations based on the model that we have created.
# This method will take the movie name and the rating 
# Will return a similarity score for all the movies that are similar to this particular movie. 

def get_similar_movies(movie_name, user_rating): 
    similar_score = item_similarity_df[movie_name]*(user_rating - 2.5) # We use the df we just created. We 
    # first get the particular movie that this user has already seen and we scale it by the rating that the 
    # user has given to that particular movie. (if it gives a 5 for that movie all will be multiplied by 5)
    # If a user rates bad a movie, we want all the similar movies to go down in the list. So we substract by the 
    # mean (2.5), only if the ratings are above 3 will appear at the top of the list. 
    similar_score = similar_score.sort_values(ascending = False) # I want it in descending order 
    
    return similar_score 


# We proceed to test how well our recommendation system is working.

# In[9]:


# The rating I give depends on the action that the movie has, that is, if it's fast and furious it would be 5 
# and if I put it's a romantic movie it would be a lower rating, a rating of 2 or 1. 
# For example an action user has rated these movies bellow:

action_lover = [('Broken Arrow (1996)', 5),
                ('Eye for an Eye (1996)', 5), 
                ('Dead Presidents (1995)', 4),
                ('Father of the Bride Part II (1995)', 2)]


similar_movies = pd.DataFrame()

# I want to get similar movies 
for movie, rating in action_lover: 
    similar_movies = similar_movies.append(get_similar_movies(movie,rating), ignore_index = True)
    # ignore index, so the indexes are not automatically created. 
    

similar_movies.head()




# The recommended movies selected are:

# In[10]:


similar_movies.sum().sort_values(ascending = False).head(20)


# * __Top Comedy Movie Recommendations:__

# For a **comedy lover** for example: 

# In[13]:


comedy_lover = [("Clueless (1995)",5),("Father of the Bride Part II (1995)",4),
                ("Dangerous Minds (1995)",1),
                ("Flirting With Disaster (1996)",5)]

similar_movies = pd.DataFrame()

for movie, rating in comedy_lover:
    
    similar_movies = similar_movies.append(get_similar_movies(movie,rating), ignore_index = True)


similar_movies.head(10)


# The recommended movies selected are:
