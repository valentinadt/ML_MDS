#!/usr/bin/env python
# coding: utf-8

# * Authors: Andrea Jiménez Zuñiga, Valentina Díaz Torres e Isabel Afán de Ribera Olaso
# * Date: 15/01/2020
# * Institution: CUNEF

# # 04. Collaborative Filtering Recommender Systems

# As mentioned above, collaborative filtering is a recommendation system based on user characteristics using "mass" information to identify similar profiles and learn from the data to recommend products individually.

# There are two classifications within collaborative filtering:
# 
# * __Memory-based methods__: use similarity metrics to determine the similarity between a pair of users. They calculate the items that have been voted on by both users and compare these votes to calculate similarity.
# 
#     - User-based filtering: similar users are identified.
#     - Item-based filtering: calculate the similarity between items.
# 
# 
# * __Model-based methods__: they use the voting matrix to create a model through which to establish the set of users similar to the active user. An example of this type is matrix decomposition based on the mathematical technique of SVD.

# ![recommendation.png](attachment:recommendation.png)

# In this notebook we will implement the __Singular Value Decomposition__ technique from __model-based method__.

# ## 04. Singular value decomposition (SVD)

# From a mathematical point of view, we can say given an actual m×n matrix A, there are orthogonal matrices U (of order m) and V (of order n) and a diagonal matrix Σ (of size m×n). This factorization of A is called Singular Value Decomposition,

# ![A.png](attachment:A.png)

# The diagonal elements of Σ are known as the singular values of the A. The columns of U are known as the left-singular vectors and are the eigenvectors of A*A. The columns of V are known as the right-singular vectors and are the eigenvectors of A*A.
# 
# The diagonal matrix Σ is uniquely determined by A. The nonzero singular values of A are the square roots of the eigenvalues A*A.

# ![svd.png](attachment:svd.png)

# * __A__: Input data matrix.
# * __U__: Left singular vectors.
# * __V__: Right singular vectors.
# * __Σ__: Singular values.

# When it comes to dimensionality reduction, the Singular Value Decomposition (SVD) is a popular method in linear algebra for matrix factorization in machine learning. Such a method shrinks the space dimension from N-dimension to K-dimension (where K<N) and reduces the number of features. SVD constructs a matrix with the row of users and columns of items and the elements are given by the users’ ratings. Singular value decomposition decomposes a matrix into three other matrices and extracts the factors from the factorization of a high-level (user-item-rating) matrix.
# 

# In particular, in the case of recommendation systems Singular value decomposition (SVD) is a collaborative filtering method for item recommendation. For this technique, the interactions between users and products are stacked into a large matrix (the rating matrix), which has as many rows as users, and as many columns as products. And the aim for the code implementation is to provide users with recommendations from the latent features of item-user matrices.

# Now we are going to implement this technique in our film recommendation system.

# ### Import Libraries

# In[67]:


import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # data splitting 
from scipy.sparse.linalg import svds # SVD model
import warnings
warnings.filterwarnings("ignore")


# ### Load data

# For this purpose we only need the databases of movies and ratings.

# In[68]:


df_movies = pd.read_csv('../../data/movies.csv', sep= ",")
df_ratings = pd.read_csv('../../data/ratings.csv', sep= ",")


# We joined the two databases to facilitate the work.

# In[69]:


# Merged data by movieId
data_movies = pd.merge(df_movies, df_ratings, on='movieId', how='inner')
data_movies.head()


# In[70]:


# we eliminate the column 'timestamp' for efficiency reasons as it includes information we do not need
data_movies.drop('timestamp', axis=1, inplace=True)


# In[42]:


# shape of the data, 25.000.095 rows and 5 columns

print(data_movies.shape)


# ### Splitting the data into train and test

# Here we split the data randomnly into test and train datasets 70% train and 30% test for model training and prediction.

# In[71]:


# we extract a sample of the data for use in the model as our database is very large

data_movies = data_movies.head(10000000)


# In[72]:


data_train, data_test = train_test_split(data_movies, test_size = 0.3, random_state = 10)
data_train.head()


# ### Singular Value Decomposition Model

# Now we have to pivot the matrix to get a ratings matrix. We set the columns as title, the rows as userId and the values in the matrix will be the ratings.

# In[73]:


# Pivoting the data for train and test
# the rating matrix will be full of null values, as most of the user's item interactions are unknown so we will replace 
# these null values with zeros

train_matrix = pd.pivot_table(data_train, index = 'userId', columns = 'title', values = 'rating').fillna(0)
test_matrix = pd.pivot_table(data_train, index = 'userId', columns = 'title', values = 'rating').fillna(0)


# In[74]:


train_matrix.head()


# And we normalize the data by each users mean and convert it from a dataframe to a numpy array to be able to carry out the decomposition of the matrix.

# In[75]:


d_m = train_matrix.to_numpy()
avg_ratings = np.mean(d_m, axis = 1)


# We are now ready to apply the matrix factorization with the Singular value decomposition (SVD) algorithm.

# In[80]:


# Applying Singular Value Decomposition with the Scipy function svds

U, sigma, Vt = svds(train_matrix, k = 20) # k is the number of singular values and vectors to compute


# In[77]:


print(U.shape) # orthogonal matrix containing the left single vectors of A


# In[50]:


print(Vt.shape) # transposed matrix whose values are the unique vectors rights of A


# In[81]:


# diagonal array in SVD, Σ diagonal matrix whose values are the singular values of matrix A ordered in decreasing value

sigma = np.diag(sigma)
print(sigma)


# In[82]:


print(sigma) # the singular values


# In[87]:


# and this is the final matrix

ratings_predict = np.dot(np.dot(U, sigma), Vt)
print(ratings_predict)


# In[84]:


# adding the user averages back to get the actual ratings prediction

ratings_predict  = ratings_predict + avg_ratings.reshape(-1, 1)
print(ratings_predict)


# ### Predictions

# In[88]:


# making predictions with the prediction matrix for every user using train data

predictions_df = pd.DataFrame(ratings_predict, columns = train_matrix.columns)
predictions_df.head()


# We create a function to return the movies with the highest predicted rating that the user we have randomly selected from our database has not yet rated.

# In[89]:


def recommend_movies(userID, df_predict, train_matrix, num_recommend):
    
    user_index = userID - 1 # user index starts at 0
    
    # Get and sort the user's ratings and predictions
    sorted_user_ratings = train_matrix.iloc[user_index].sort_values(ascending = False) # sort the user ratings
    sorted_user_predicts = df_predict.iloc[user_index].sort_values(ascending = False) # sort the user predicts

    
     # Return recommendations of movies with the highest predicted rating that the user hasn't seen yet
    recom_df = pd.concat([sorted_user_ratings, sorted_user_predicts], axis = 1) # concat ratings and predicts
    recom_df.index.name = 'Recommended Items'   # title for the recommendation list
    recom_df.columns = ['u_ratings', 'u_predicts'] # user ratings and user predictions will be the columns
    
    recom_df = recom_df.loc[recom_df.u_ratings == 0] 
    recom_df = recom_df.sort_values('u_predicts', ascending = False) # sort values by user predictions 
    print('Here we have the recommended items for user {0}.'.format(userID))
    print(recom_df.head(num_recommend))


# In order for the model to make the predictions we establish the number of users to whom we want to make recommendations and the number of recommendations we want to make 

# In[90]:


# Here we give value to the parameters of the function 

userID = 152
num_recommend = 10
recommend_movies(userID, predictions_df, train_matrix, num_recommend)


# ### Evaluating the model

# In[91]:


# Actual average rating for each item
# using test

test_matrix.mean().head()


# In[92]:


# Singular Value Decomposition
U_t, sigma_t, Vt_t = svds(test_matrix, k = 20)

# Construct diagonal array in SVD
sigma_t = np.diag(sigma_t)


# In[93]:


# Final matrix
test_predicted_ratings = np.dot(np.dot(U_t, sigma_t), Vt_t) 

# Predicted ratings
test_predict = pd.DataFrame(test_predicted_ratings, columns = test_matrix.columns)
test_predict.head()


# In[94]:


# Predicted average rating for each item

test_predict.mean().head()


# To evaluate the model we will use the mean square error as a measure of accuracy, a standard way to measure the error of a model in predicting quantitative data. It is  a measure of the differences between values (sample or population values) predicted by a model or an estimator and the values observed

# ![rsme.PNG](attachment:rsme.PNG)

# In[95]:


# Dataframe to calculate RMSE
df_rmse = pd.concat([test_matrix.mean(), test_predict.mean()], axis = 1) # concat test_matrix.mean and test_predict.mean to compare actual and predicted ratings
df_rmse.columns = ['average_actual_ratings', 'average_predicted_ratings']
print(df_rmse.shape)

# ordered by the index of the item
df_rmse['item_index'] = np.arange(0, df_rmse.shape[0], 1)
df_rmse.head()


# In[96]:


# Root mean square error: measure of the differences between values predicted by a model or an estimator and the values observed

RMSE = round((((df_rmse.average_actual_ratings - df_rmse.average_predicted_ratings) ** 2).mean() ** 0.5), 4)
print('\nThe RMSE of the SVD Model is = {} \n'.format(RMSE))


# As can be seen in the table above and checked with the result of the error the model is quite adequate as there is very little difference between the current average scores and the predicted average scores for the recommended films.

# Now let's train on the data set and get the predictions for a random user.

# In[97]:


# Singular Value Decomposition
U, sigma, Vt = svds(train_matrix, k = 20)

# Construct diagonal array in SVD
sigma = np.diag(sigma)


# In[65]:


pre_ratings = np.dot(np.dot(U, sigma), Vt) 

# Predicted ratings
preds_df = pd.DataFrame(pre_ratings, columns = train_matrix.columns)


# In[98]:


# Here we give value to the parameters of the function 

userID = 1356
num_recommend = 10
recommend_movies(userID, preds_df, train_matrix, num_recommend)


# # Conclusion

# There are two problems in applying the SVD. The first one is how to deal with those items that the user has not evaluated, because if that item is evaluated with a value of 0 the prediction will be of a value very close to 0 and that is not the expected result. The second problem that arises is the dispersion that the voting matrixes have; that is, the user only votes a very small percentage of the items that are in the system, so you have to work with widely dispersed voting matrices.
# 
