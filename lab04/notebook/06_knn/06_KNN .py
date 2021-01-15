#!/usr/bin/env python
# coding: utf-8

# ## 0.6 Item-based recommender using Nearest Neighbor method

# ## Import libraries

# In[2]:


import os
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split # data splitting 
import warnings
warnings.filterwarnings("ignore")
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors #knn
from sklearn.metrics import mean_squared_error #sqrt operations
from sklearn.metrics import pairwise_distances
import pickle


# ### Load data

# In this case, we only need movies and ratings datasets. 

# In[4]:


# Load data

movies = pd.read_csv('../../data/movies.csv', sep= ",")
ratings = pd.read_csv('../../data/ratings.csv', sep= ",")


# In[5]:


movies.head()


# In[6]:


ratings.head()


# In[7]:


# delete the year value to the title column because we are going to recommend movies and it looks better if it's just the name of the movie
movies['title'] = movies['title'].str.replace(r"\(.*\)","")
movies.head()


# In[8]:


# Merged data by movieId

data_movies = pd.merge(ratings, movies, on='movieId').drop(['timestamp','genres'],axis=1)
# We use a subsample
data_movies = data_movies.head(10000000)
# data_movies.head()


# First, we are going to work with raw data in order to find the movies user have not watched yet, using nan values and then we fill these values with 0.
# 

# In[9]:


# Missing values in rating
# It's necessary to pivot the dataframe to the wide format with movies as rows and users as columns
nan_table = data_movies.pivot_table(values = 'rating', index = 'userId', columns = 'title')


# The nan values represent the movies that have not been rated yet, so we don't know the rating. Those will be used later for recommendation since the user might want to discover them.

# In[10]:


# Identify the movies each user have rated

rated_movies = {}  # a dictionary for movies rated by users
rows_indexes = {}
for i, row in nan_table.iterrows():  # loop that runs through the rows to detect na´s
    rows = [x for x in range(0,len(nan_table.columns))] # to get the lenght of the columns
    combined = list(zip(row.index, row.values, rows)) # iterator of tuples where the elements of each iterator is paired together
                                                     # iterators: row.index, row.values, rows
    rated = [(x,z) for x,y,z in combined if str(y) != 'nan'] # choosing movies which values are not nan
    # for rated combined x, y and z values when (y) is not a null value
    index = [i[1] for i in rated]  # take from column 1 rated movies
    row_names = [i[0] for i in rated] # take from column 0 rated movies
    rows_indexes[i] = index
    rated_movies[i] = row_names


# #### User-item Matrix

# Now it's time to find the movies the users have not rated yet, we don't want to recommed movies that the users have already watched.
# 
# Users might be probably looking for movies that they haven’t watched or just different movies. If the system recomends movies with diverse topics it would allow users to explore and discover different tastes and keeps user engaged with the recommender product. We need to keep in mind that lack of diversity will make users get bored and less engaged with the product and we don't want that.

# In[15]:


# Creating a new pivot table, but this time we fill nan values with 0 to be able to measure the vectors and evaluate he similarity between items
# From now on we are going to work with this dataset since we have already trated the nan values and
# We want to recommed movies not watched yet to the users

noNan_table = data_movies.pivot_table(values = 'rating', index = 'userId', columns = 'title').fillna(0)
noNan_table.head()


# In[17]:


noNan_table = noNan_table.apply(np.sign)


# In[18]:


# Identify the movies that users have not rated yet

not_rated_movies = {} # we are creating a dictionary again but this time for the films that have not been rated
not_rated_indexes= {} # to get the movies that users have not rated
for i, row in noNan_table.iterrows(): # loop that runs through the rows of pivot_table to detect no rated movies
    rows = [x for x in range(0,len(nan_table.columns))] # to get the lenght of the columns 
    combined = list(zip(row.index, row.values, row))  # iterator of tuples where the elements of each iterator is paired together
    idx_row = [(idx,col) for idx, val, col in combined if not val > 0] # combine idx, val, col when value is not > 0 that means the movie has not been rated
    idx_2 = [i[1] for i in idx_row]  # take from column 1 rated movies
    row_names = [i[0] for i in idx_row] # take from column 1 rated movies
    not_rated_indexes[i] = idx_2
    not_rated_movies[i] = row_names
    


# In[50]:


# dictionary with userId as key and not rated movie title as value
# Here we can see the first 10 movies for the first user

print(not_rated_movies[1][0:10])


# ###  Nearest Neighbor Recommender Model (KNN)
# 

# Now we are ready to build an unsupervised KNN model. KNN separates the dataset into different clusters. It calculates the distance between the target movie and other movies in the dataset. Then it ranks its distances and returns the nearest neighbor movies (k) as the most similar movie recommendations.

# In[57]:


from IPython.display import Image
Image(filename='knn.png', width = '800')


# In[40]:


n = 5 #the number of neighbors
knn = NearestNeighbors(n_neighbors = n,algorithm = 'brute', metric = 'cosine')  # brute is the lgorithm used to compute the nearest neighbors
#we use the cosine similarity for nearest neighbor search
knn_fit = knn.fit(noNan_table.T.values)  # fit the nearest neighbors estimator from the training dataset
i_distances, i_indices = knn_fit.kneighbors(noNan_table.T.values) # with T function we transpose the table


# #### Item-based collaborative filtering

# 
# Item-based recommender filtering is the best way of doing that. It measures the similarities between items by using the user's ratings os those items.
# 

# In[45]:


# here we have a dictionary where the key is the rated movie and the values are the movies associated 
i_dic = {} # dictionary for recomendation based on similar movies
for i in range(len(noNan_table.T.index)): # loop to inspect the index of the transpose pivot table
    item_idx = i_indices[i]
    col_names = noNan_table.T.index[item_idx].tolist() # to get the actual index name and turned it to a list
    i_dic[noNan_table.T.index[i]] = col_names


# In[58]:


top_Recm = {}
for k,v in rows_indexes.items(): 
    item_idx = [j for i in i_indices[v] for j in i]  # for each movie and item indexes find all the indexes that are most similar to the rated movies
    item_dist = [j for i in i_distances[v] for j in i] # and the same for the distances
    zip_1 = list(zip(item_dist, item_idx))
    combine_idx = {i:d for d, i in zip_1 if i not in v} # combine when index is not in v because we want not watched movies 
    zip_3 = list(zip(combine_idx.keys(),combine_idx.values()))
    sort = sorted(zip_3, key = lambda x: x[1]) # order from the most similar to the least similar 
    recommendation = [(noNan_table.columns[i],d) for i, d in sort] # get columns names from the original pivot table
    top_Recm[k]= recommendation

#k = users 


# In[59]:


# function to get recommendations for a particular user

def getrecommendations(user, number_of_recs = 30):
    if user > len(noNan_table.index):   # in case you choose a user number that is not in the table
        print('Out of range, there are only {} users, try again!'.format(len(noNan_table.index)))
    else:
         # we choose a user number from the list then I get back the movies that the user has seen
        print("These are all the movies you have viewed in the past: \n\n{}".format('\n'.join(rated_movies[user])))
        print()
        # and returns some similar movies that the user has not seen including the similarity
        print("We recommed to view these movies too:\n")
    for k,v in top_Recm.items():
        if user == k:
            for i in v[:number_of_recs]:
                print('{} with similarity: {:.4f}'.format(i[0], 1 - i[1]))


# In[60]:


# Recommendations for 500 users

getrecommendations(10)


# ### Evaluation

# #### Rating predictions for the movies users had not watched before

# In[61]:


i_distances = 1 - i_distances


# In[62]:


recommender_predictions = i_distances.T.dot(noNan_table.T.values)/np.array([np.abs(i_distances.T).sum(axis = 1)]).T


# In[63]:


argsort_fun = noNan_table.T.values[i_distances.argsort()[0]]


# #### Evaluating the recommendation model

# For the evaluation, we use Root Mean Square Error(rmse) in order to make good predictions by using the sklearn.metrics package.

# In[64]:


# function for RMSE that measure how well this recommendation system is

def rmse(prediction, argsort_fun):
    prediction = prediction[argsort_fun.nonzero()].flatten()
    argsort_fun = argsort_fun[argsort_fun.nonzero()].flatten()
    return sqrt (mean_squared_error(prediction,argsort_fun)) #squrt=squared root


# In[65]:


error_result = rmse(recommender_predictions, argsort_fun)
print("Acuraccy: {:.3f}".format(100 - error_result))
print("RMSE: {:.5f}".format(error_result))


# ## Conclusion

# One of the main shortcomings in item based collaborative filtering is that it's that this system is most likely to recommed the most popular movies instead of the unknown ones. We found the same issue if we look at the new movies, those have less interactios and for this reason they might be less likely to be recommended.
# 
# However, the accuraccy of the model is 99%, so it's pretty high and we could conclude that this system really works in real life.

# In[66]:


model = knn
filename = 'modelo_knn.pkl'
pickle.dump(model,open(filename,'wb'))

