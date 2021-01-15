#!/usr/bin/env python
# coding: utf-8

# * Authors: Andrea Jiménez Zuñiga, Valentina Díaz Torres e Isabel Afán de Ribera Olaso
# * Date: 15/01/2021
# * Institution: CUNEF

# # 01. EDA Movie Recommendation System

# ## Introduction

# One of the most widely used machine learning tools in the business world today are recommendation systems as they allow predictions to be made about articles or content that may be useful or valuable to the user.  These systems select data provided by the user directly or indirectly, and proceed to analyse and process information from the user's history in order to transform this data into recommendation knowledge. In such a way that an advanced network of complex connections between products and users is created. There are three types of connection:
# 
# * User-product relationship: based on individual user preferences.
# * User-user relationship: based on similar people who are likely to have similar product preferences.
# * Product-product relationship: based on similar or complementary products. 
# 
# When configured correctly, these systems can significantly increase sales, revenue, click rates, conversions and other important business metrics. This is because customising recommendations to each individual user creates a very positive effect on customer satisfaction.
# 
# In order to learn more about this powerful Machine Learning technique, we are going to test different algorithms that will allow us to draw beneficial conclusions for the business. To do this we will use a set of databases provided by GroupLens through its web recommendation system "MovieLens" with information about movies and scores given by users.
# 

# ![movies.jpg](attachment:movies.jpg)

# ## Import Libraries

# In[1]:


import sys
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import re # expresiones regulares, para eliminar el paréntesis en la columna titulo
pd.options.display.max_columns = None #para poder visualizar todas las columnas sin puntos suspensivos
pd.options.display.max_rows = None #para poder visualizar todas las filas sin puntos suspensivos
import wordcloud
from wordcloud import WordCloud, STOPWORDS
pd.options.display.max_columns = None #para poder visualizar todas las columnas sin puntos suspensivos
pd.options.display.max_rows = None #para poder visualizar todas las filas sin puntos suspensivos


# ## Data Gathering and Importing

# At first we imported the entire data set provided by MovieLens which includes 25.000.095 ratings and 1.093.360 tag applications across 62.423 movies. Later on, we will work only with the most relevant ones for the analysis.

# In[2]:


df_scores = pd.read_csv('../../data/genome-scores.csv', sep= ",")
df_g_tags = pd.read_csv('../../data/genome-tags.csv', sep= ",")
df_movies = pd.read_csv('../../data/movies.csv', sep= ",")
df_ratings = pd.read_csv('../../data/ratings.csv', sep= ",")
df_links = pd.read_csv('../../data/links.csv', sep= ",")
df_tags = pd.read_csv("../../data/tags.csv", sep =",")


# First, we will see how the different datasets are organised.

# In[3]:


# First 5 rows  of the scores dataset

df_scores.head()


# In[4]:


# First 5 rows  of the genome tags dataset

df_g_tags.head()


# In[5]:


# First 5 rows  of the tags dataset

df_tags.head()


# In[6]:


# First 5 rows  of the links dataset

df_links.head()


# In[7]:


# First 5 rows  of the movies dataset

df_movies.head()


# In[8]:


# First 5 rows  of the movies dataset

df_ratings.head()


# In[9]:


# Merge the movies and ratings datasets by movieId

data_movies = pd.merge(df_movies, df_ratings, on='movieId', how='inner')

# Because got so many memory errors in visualization, we are going to work with 10.000.000 million of observations for some graphics.
# We rename our data_movies with a subsample from the original dataset. However, for the most of our work here, in the EDA notebook
# we will work with the whole dataset (25 millions).

data_movies2 = data_movies.head(10000000)

# Visualization

data_movies.head()


# ## Basic Data Analysis

# We check the dimensions of our dataset. The dataset measures 25,000,095  rows and 6 columns.

# In[10]:


# Number of columns and rows

print(f'Number of rows: {data_movies.shape[0]}')
print(f'Number of columns: {data_movies.shape[1]}')


# We now check the names of the 6 columns found in the merged dataset. 

# In[11]:


# Column names

data_movies.columns


# Our dataset is a pandas dataframe with 25,000,095 rows and 6 columns of the integer, object and float type.

# In[12]:


# basic information

data_movies.info()


# And we check the statistical position measurements of all our variables.

# In[13]:


# Summary statistics

data_movies.describe(include='all')


# We have checked the duplicated data and we have verified that there is no duplicate data.

# In[14]:


duplicated_rows = data_movies[data_movies.duplicated()]
print('number of duplicated rows: ', duplicated_rows.shape) #No duplicated data 


# ## Exploring the data

# There is information of 162.541 different users and 59.047 different movies in the dataset.

# In[15]:


# Number of unique users and movies

print("Unique users: ", data_movies.userId.nunique())
print("Unique movies: ",data_movies.movieId.nunique())


# If we want to know what the minimum, mean and maximum ratings are, we need to do the following: 

# We can see that the minimum rating is 0.5, the mean rating is 3.53 and the maximum rating that a user can give is 5.0.

# In[20]:


# Minimum of Rating
print("The minimum rating is:", np.min(data_movies.rating))
# Mean of Rating
print("The mean rating is:", round(np.mean(data_movies.rating),2))
# Maximum of Rating
print("The maximum rating is:", np.max(data_movies.rating))


# In[21]:


# Display distribution of rating
sns.distplot(data_movies2['rating'].fillna(data_movies2['rating'].median()))


# The frequence of each score can be calculated. It is seen that the most used score is _4.0_ and the least used score is _0.5_.

# In[22]:


# Frequency of scores

print(df_ratings.rating.value_counts())


# In[23]:


plt.hist(df_ratings.rating,bins=8)  


# It can be noticed in the histogram above, that score 4 is the most frequent, followed by score 3 and 5.

# We want to check the different genres that are found in the movies dataset. In order to achieve that, we have proceeded with a loop in order to separate the movies by genres. This gives us a list of the different genres found in the movies dataset. 

# In[24]:


# Loop for separating the movies by genres

genres_list = []
for index, row in df_movies.iterrows():
    try:
        genres = row.genres.split('|') # First we split the genres that are grouped by a pipeline ("|"). 
        genres_list.extend(genres)
        
    except: 
        
        genres_list.append(row.genres)
        
genres_list = list(set(genres_list)) 

# This gives us a list of unique genres found in the movies dataset.


# This is the genres list. We can see how also "no genres listed" is also a category where  are the most unknown movies.

# In[25]:


# Resulting genres list

genres_list


# In[26]:


plt.figure(figsize=(20,7))
listado = data_movies2['genres'].apply(lambda list_gener : str(list_gener).split("|"))
geners_count = {}

for list_gener in listado:
    for gener in list_gener:
        if(geners_count.get(gener,False)):
            geners_count[gener] = geners_count[gener] + 1
        else:
            geners_count[gener] = 1       
plt.bar(geners_count.keys(),geners_count.values(), color = 'plum')


# Here we can see that the most common genre are Drama and Comedy.

# * #### Most common words in movie titles

# There is a way to check if certain words are used more commonly in titles than others. This can be done as follows: 

# In[27]:


get_ipython().run_cell_magic('time', '', '# Create a wordcloud of the movie titles\ndata_movies2[\'title\'] = data_movies2[\'title\'].fillna("").astype(\'str\')\ntitle_corpus = \' \'.join(data_movies2[\'title\'])\ntitle_wordcloud = WordCloud(stopwords=STOPWORDS, background_color=\'black\', height=2000, width=4000).generate(title_corpus)\n\n# Plot the wordcloud\nplt.figure(figsize=(16,8))\nplt.imshow(title_wordcloud)\nplt.axis(\'off\')\nplt.show()')


# In[28]:



# Create a wordcloud of the tags
df_tags['tag'] = df_tags['tag'].fillna("").astype('str')
title_corpus = ' '.join(df_tags['tag'])
title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', height=2000, width=4000).generate(title_corpus)

# Plot the wordcloud
plt.figure(figsize=(16,8))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.show()


# * #### List of top 25 rated movies

# It is also possible to see what the top 25 movies are. To do that, it is needed to group the first 25 values of the dataset _"data_movies"_ by rating and sort them in a descending order. 
# 
# We can see that __Forrest Gump__ is the top rated movie, with a total of 81,491 rates. 

# In[29]:


# List of top 25 rated movies

top25 = data_movies.groupby('title')['rating'].count().sort_values(ascending = False)[:25]
top25


# For a better visualization, we can represent these top 25 movies with a graph.

# In[30]:


# Representation of the top 25 rated movies

top25.plot(kind ='barh', color = 'turquoise', figsize = (10,9))
plt.xlabel('Rating')
plt.ylabel('Movie')
plt.title('Top 25 rated movies')
plt.show


# * #### Correlation: Year of the movie vs Rating by Genre of the movie
# 

# In order to be able to calculate the correlation of the year of the movie vs the rating by genre of the movie it is needed to create a new column in the dataset _data_movies_ named ___year___ which represents the year the movie was made. For that, we need to split the _title_ column in order to have first the column of the title of the movie and in another column the year made. 
# 
# As done earlier we know that the different genres are: 
# * 'Crime',
# * 'Animation',
# * 'Mystery',
# * 'Documentary',
# * 'Horror',
# * 'Film-Noir',
# * 'Action',
# * 'Adventure',
# * 'Sci-Fi',
# * 'War',
# * 'Romance',
# * 'Drama',
# * 'Children',
# * 'Thriller',
# * 'Western',
# * 'Musical',
# * 'Comedy',
# * 'Fantasy'
# 
# Taking this into consideration, we are going to create a new dataset named _movies_2_ that contains the movies with those genres, followed by creating another object that will calculate the mean, std and count, taking into consideration the ratings as well. 

# In[31]:


# Creating a new column called 'year' by using  the year values from the 'title' column

data_movies['year'] = data_movies.title.str.extract('\s\((\d+)', expand=True)

# Dropping the years to the 'title' column
data_movies['title'] = data_movies['title'].str.replace(r"\(.*\)","")


# In[32]:


movies_2 = data_movies.loc[data_movies['genres'].isin(
    ['Comedy', 'Crime', 'Documentary', 'Fantasy', 'Western','War', 'Mystery', 'Romance', 'Animation',  
        'Musical', 'Action', 'Children','Film-Noir','Drama','Adventure','Horror','Sci-Fi','Thriller'])]

mean_ratings = movies_2.groupby([ 'genres', 'year'], 
                               as_index=False)['rating'].agg(['mean', 'std', 'count'])


# In[33]:


genres = ['Comedy', 'Crime', 'Documentary', 'Fantasy', 'Western','War', 'Mystery', 'Romance', 'Animation',  
        'Musical', 'Action', 'Children','Film-Noir','Drama','Adventure','Horror','Sci-Fi','Thriller']

data = pd.DataFrame()

for f in genres:
    f1 = mean_ratings.loc[f]
    g1 = f1.loc[:]['mean']
    g1 = pd.Series(g1)
    data = pd.concat([data,g1], axis=1)
    
data.columns = genres
datay = data.fillna(data.mean())
cory = datay.corr()
plt.subplots(figsize=(15,7))
sns.heatmap(cory, annot=True)


# In this correlation we can see that some genres have higher correlation than others, as it is expected. 
# We can see that for example the genre _Thriller_ has the highest correlation with the genre _Crime_ , this means that, a movie whose genre is thriller it is very likely that it is also classified as a crime movie. 
# 
# On the other hand, we can see that the genre _Documentary_ has a negative correlation with the genre _Horror_ , this expresses the opposite of what happens with the genres Thriller and Crime.

# In[34]:


# Turning genres to onehot encoding
onehot_genres = pd.concat([data_movies2.drop('genres', axis=1), data_movies2.genres.str.get_dummies(sep='|')], axis=1)


# In[35]:


thriller = onehot_genres[ onehot_genres['Thriller'] == 1]
crime = onehot_genres[ onehot_genres['Crime'] == 1]
thriller_crime = onehot_genres[ (onehot_genres['Thriller'] == 1) & (onehot_genres['Crime'] == 1)]

print ('# Thriller: ' + str(len(thriller)) )
print ('# Crime: ' + str(len(crime)) )
print ('# Thriller/Crime: ' + str(len(thriller_crime)) )


# In the correlation above, we used fillna( ) with mean, and in the one shown below we use dropna( ).

# In[36]:


data.dropna()

data.columns = genres
corry = data.corr()
mask = np.zeros_like(corry)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
        plt.subplots(figsize=(15,10))
        sns.heatmap(corry, mask=mask, annot=True);


# We can see that the correlation between the genres _Comedy_ and _Film Noir_ and between the genres _Film Noir_ and _Horror_ has increased considerably to 0.75 and 0.98, respectively. This means that a movie categorized as _Film Noir_  is most likely to be categorized as a _Comedy_  or _Horror_ movie as well.
# 
# On the other hand, the correlation between the genres _Film Noir_ and _Sci Fi_ has decreased to -0.87, which means that a film categorized as _Film Noir_ is most likely to not be categorized as _Horror_ as well.
# 

# In[37]:


data.plot()
plt.legend(loc=(2.05,0), ncol=1)
plt.xlabel('Movie rating')
plt.title('Movie rating trends by year of movie')
plt.show()


# In[38]:


data_movies.to_csv('data_movies.csv', index = None, header=True)

