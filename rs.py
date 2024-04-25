import gdown

# Replace 'your_file_id' with the actual file ID from your Google Drive link
file_id = '17JgsMrRlRtEZZdBSsKAcqfa3_NFakzQU'

# Define the output file path
output_path = 'tops_fashion.json'

# Download the file
gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)

import pandas as pd

data = pd.read_json('tops_fashion.json')

print('Number of data points : ', data.shape[0])
print()
print('Number of features/variables:', data.shape[1])

data.columns # prints column-names or feature-names.

data = data[['asin', 'brand', 'color', 'medium_image_url', 'product_type_name', 'title', 'formatted_price']]

print('Number of data points : ', data.shape[0])
print()
print('Number of features:', data.shape[1])
print()
data.head() # prints the top rows in the table.

print(data['product_type_name'].describe())

# names of different product types
print(data['product_type_name'].unique())

from collections import Counter

# find the 10 most frequent product_type_names.
product_type_count = Counter(list(data['product_type_name']))
product_type_count.most_common(10)

print(data['brand'].describe())

# find the 10 most frequent brand.
brand_count = Counter(list(data['brand']))
brand_count.most_common(10)

print(data['color'].describe())

# find the 10 most frequent color.
color_count = Counter(list(data['color']))
color_count.most_common(10)

print(data['formatted_price'].describe())

price_count = Counter(list(data['formatted_price']))
price_count.most_common(10)

print(data['title'].describe())

data.isnull().sum()

data.to_pickle('180k_apparel_data')

# consider products which have price information, data['formatted_price'].isnull() => gives the information
# about the dataframe row's which have null values price == None|Null
data = data.loc[~data['formatted_price'].isnull()]
#This will store those values whose 'formatted_price' is not null
print('Number of data points After eliminating price = NULL :', data.shape[0])

# consider products which have color information
# data['color'].isnull() => gives the information about the dataframe row's which have null values price == None|Null
data =data.loc[~data['color'].isnull()]
print('Number of data points After eliminating color = NULL :', data.shape[0])

data.to_pickle('28k_apparel_data')

# find number of products that have duplicate titles.
print(sum(data.duplicated('title')))

# read data from pickle file from previous stage
data = pd.read_pickle('28k_apparel_data')
data.head()

# Remove All products with very few words in title
data_sorted = data[data['title'].apply(lambda x: len(x.split())>4)]
print("After removal of products with short description:", data_sorted.shape[0])

# Sort the whole data based on title (alphabetical order of title)
data_sorted.sort_values('title',inplace=True, ascending=False)
data_sorted.head()

indices = []
for i, row in data_sorted.iterrows():  # Iterate over DataFrame rows as (index, Series) pairs.
    indices.append(i)

import itertools

# We'll remove the duplicates which differ only at the end.
duplicates = []
# 2 variables used at iterators.
i = 0
j = 0
num_data_points = data_sorted.shape[0]   # Number of data points in our dataframe.

while i < num_data_points and j < num_data_points:

    previous_i = i

    # store the list of words of ith string in a, ex: a = ['tokidoki', 'The', 'Queen', 'of', 'Diamonds', 'Women's', 'Shirt', 'X-Large']
    a = data['title'].loc[indices[i]].split()

    # search for the similar products sequentially as they are arranged alphabetically.
    j = i+1
    while j < num_data_points:

        # store the list of words of jth string in b, ex: b = ['tokidoki', 'The', 'Queen', 'of', 'Diamonds', 'Women's', 'Shirt', 'Small']
        b = data['title'].loc[indices[j]].split()

        # store the maximum length of two strings.
        length = max(len(a), len(b))

        # count is used to store the number of words that are matched in both strings.
        count  = 0

        # itertools.zip_longest(a,b): will map the corresponding words in both strings, it will appened None in case of unequal strings
        # example: a =['a', 'b', 'c', 'd']
        # b = ['a', 'b', 'd']
        # itertools.zip_longest(a,b): will give [('a','a'), ('b','b'), ('c','d'), ('d', None)]
        for k in itertools.zip_longest(a, b):
            if (k[0] == k[1]):          # Checking if the pair made is same or not.
                count += 1              # If one pair is same, we'll increase the count by 1.

        # if the number of words in which both strings differ are > 2 , we are considering it as those two apperals are different.
        # if the number of words in which both strings differ are < 2 , we are considering it as those two apperals are same, hence we are ignoring them.
        if (length - count) > 2: # number of words in which both sentences differ.
            # if both strings are differ by more than 2 words we include the 1st string index.
            duplicates.append(data_sorted['asin'].loc[indices[i]])

            # if the comparision between is between num_data_points, num_data_points-1 strings and they differ in more than 2 words we include both.
            if j == num_data_points-1: duplicates.append(data_sorted['asin'].loc[indices[j]])

            # start searching for similar apperals corresponds 2nd string.
            i = j
            break
        else:
            j += 1
    if previous_i == i:
        break

# We'll take only those 'asins' which have not similar titles(After removing titles that differ only in last few words).
data = data.loc[data['asin'].isin(duplicates)]   # Whether each element in the DataFrame is contained in values.

data.to_pickle('17k_apparel_data')

print('Number of data points at final stage:', data.shape[0])

data = pd.read_pickle('17k_apparel_data')

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# we use the list of stop words that are downloaded from nltk lib.
stop_words = set(stopwords.words('english'))
# print('list of stop words:', stop_words)

def nlp_preprocessing(total_text, index, column):
    if type(total_text) is not int:
        string = ""
        for words in total_text.split():
            # remove the special chars in review like '"#$@!%^&*()_+-~?>< etc.
            word = ("".join(e for e in words if e.isalnum())) # Returns only words with (A-z) and (0-9)
            # Conver all letters to lower-case
            word = word.lower()
            # stop-word removal
            if not word in stop_words:
                string += word + " "
        data[column][index] = string

# we take each title and we text-preprocess it.
for index, row in data.iterrows():
    nlp_preprocessing(row['title'], index, 'title')

data.head()

from nltk.stem.porter import *
stemmer = PorterStemmer()

from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import seaborn as sns

def display_img(url, ax, fig):
    # we get the url of the apparel and download it
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    # we will display it in notebook
    plt.imshow(img)

def plot_heatmap(keys, values, labels, url, text):
        # keys: list of words of recommended title
        # values: len(values) ==  len(keys), values(i) represents the occurence of the word keys(i)
        # labels: len(labels) == len(keys), the values of labels depends on the model we are using
                # if model == 'bag of words': labels(i) = values(i)
                # if model == 'tfidf weighted bag of words':labels(i) = tfidf(keys(i))
                # if model == 'idf weighted bag of words':labels(i) = idf(keys(i))
        # url : apparel's url

        # we will devide the whole figure into two parts
        gs = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[4,1])
        fig = plt.figure(figsize=(25,3))

        # 1st, ploting heat map that represents the count of commonly ocurred words in title2
        ax = plt.subplot(gs[0])
        # it displays a cell in white color if the word is intersection(lis of words of title1 and list of words of title2), in black if not
        ax = sns.heatmap(np.array([values]), annot=np.array([labels]))
        ax.set_xticklabels(keys) # set that axis labels as the words of title
        ax.set_title(text) # apparel title

        # 2nd, plotting image of the the apparel
        ax = plt.subplot(gs[1])
        # we don't want any grid lines for image and no labels on x-axis and y-axis
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # we call dispaly_img based with paramete url
        display_img(url, ax, fig)

        # displays combine figure ( heat map and image together)
        plt.show()

def plot_heatmap_image(doc_id, vec1, vec2, url, text, model):

    # doc_id : index of the title1
    # vec1 : input apparels's vector, it is of a dict type {word:count}
    # vec2 : recommended apparels's vector, it is of a dict type {word:count}
    # url : apparels image url
    # text: title of recomonded apparel (used to keep title of image)
    # model, it can be any of the models,
        # 1. bag_of_words
        # 2. tfidf
        # 3. idf

    # we find the common words in both titles, because these only words contribute to the distance between two title vec's
    intersection = set(vec1.keys()) & set(vec2.keys())

    # we set the values of non intersecting words to zero, this is just to show the difference in heatmap
    for i in vec2:
        if i not in intersection:
            vec2[i]=0

    # for labeling heatmap, keys contains list of all words in title2
    keys = list(vec2.keys())
    # if ith word in intersection(lis of words of title1 and list of words of title2): values(i)=count of that word in title2 else values(i)=0
    values = [vec2[x] for x in vec2.keys()]

    # labels: len(labels) == len(keys), the values of labels depends on the model we are using
        # if model == 'bag of words': labels(i) = values(i)
        # if model == 'tfidf weighted bag of words':labels(i) = tfidf(keys(i))
        # if model == 'idf weighted bag of words':labels(i) = idf(keys(i))

    if model == 'bag_of_words':
        labels = values
    elif model == 'tfidf':
        labels = []
        for x in vec2.keys():
            # tfidf_title_vectorizer.vocabulary_ it contains all the words in the corpus
            # tfidf_title_features[doc_id, index_of_word_in_corpus] will give the tfidf value of word in given document (doc_id)
            if x in  tfidf_title_vectorizer.vocabulary_:
                labels.append(tfidf_title_features[doc_id, tfidf_title_vectorizer.vocabulary_[x]])
            else:
                labels.append(0)

    plot_heatmap(keys, values, labels, url, text)

# this function gets a list of words along with the frequency of each
# word given "text"
def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    # words stores list of all words in given string, you can try 'words = text.split()' this will also gives same result
    return Counter(words) # Counter counts the occurence of each word in list, it returns dict type object {word1:count}

def get_result(doc_id, content_a, content_b, url, model):
    text1 = content_a
    text2 = content_b

    # vector1 = dict{word11:#count, word12:#count, etc.}
    vector1 = text_to_vector(text1)

    # vector1 = dict{word21:#count, word22:#count, etc.}
    vector2 = text_to_vector(text2)

    plot_heatmap_image(doc_id, vector1, vector2, url, text2, model)

def save_recommendations_to_js(recommendations_list, filename='rs.js'):
    with open(filename, 'w') as js_file:
        js_file.write('import React from "react";\n')
        js_file.write('const recom = [\n')
        for index, recommendation in enumerate(recommendations_list, start=1):
            js_file.write('  {\n')
            js_file.write(f'    "id": "{index}",\n')
            for key, value in recommendation.items():
                if key != "id":
                    js_file.write(f'    "{key}": "{value}",\n')
            js_file.write('  },\n')
        js_file.write('];\n\n')
        js_file.write('export default recom;')
    print(f'Recommendations saved to {filename}')

# Make sure to call save_recommendations_to_js(recommendations_list) from tfidf_model

from sklearn.feature_extraction.text import CountVectorizer
bow_title_vectorizer = CountVectorizer()
bow_title_features = CountVectorizer().fit_transform(data['title'])
bow_title_features.get_shape()

def bag_of_words_model(doc_id, num_results):
    recommendations_list = []
    # doc_id: apparel's id in given corpus

    # pairwise_dist will store the distance from given input apparel to all remaining apparels
    # the metric we used here is cosine, the cosine distance is mesured as K(X, Y) = <X, Y> / (||X||*||Y||)
    # http://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
    pairwise_dist = pairwise_distances(bow_title_features, bow_title_features[doc_id], metric='cosine', n_jobs=-1)

    # np.argsort will return `num_results` indices of the smallest distances
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    #pdists will store the smallest distances
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
    # np.argsort returns the indices of the smallest distances in ascending order.
    # The sort() method sorts the list ascending by default.

    #data frame indices of the 9 smallest distace's
    df_indices = list(data.index[indices])

    #displaying the results.
    for i in range(0, len(indices)):
        recommendation = {
            "ASIN": data['asin'].loc[df_indices[i]],
            "BRAND": data['brand'].loc[df_indices[i]],
            "Title": data['title'].loc[df_indices[i]],
            "Euclidean_similarity": pdists[i],
            "Image_URL": data['medium_image_url'].loc[df_indices[i]],
            "Model": 'bag_of_words'
        }
        recommendations_list.append(recommendation)
        
        save_recommendations_to_js(recommendations_list)
bag_of_words_model(12566, 20) # change the index if you want to.
# In the output heat map each value represents the count value
# of the label word, the color represents the intersection
# with inputs title.
# 12566 is the index of the "Query title"
#try 931

tfidf_title_vectorizer = TfidfVectorizer(min_df = 0)
tfidf_title_features = tfidf_title_vectorizer.fit_transform(data['title'])
# returns the a sparase matrix of dimensions #data_points * #words_in_corpus
tfidf_title_features.shape
# CountVectorizer().fit_transform(courpus) returns the a sparase matrix of dimensions #data_points * #words_in_corpus
# tfidf_title_features[doc_id, index_of_word_in_corpus] = tfidf values of the word in given doc

def tfidf_model(doc_id, num_results):
    recommendations_list = []
    # doc_id: apparel's id in given corpus
    # Similar to Query in Transformers^

    # pairwise_dist will store the distance from given input apparel to all remaining apparels
    # the metric we used here is cosine, the cosine distance is mesured as K(X, Y) = <X, Y> / (||X||*||Y||)
    # http://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
    pairwise_dist = pairwise_distances(tfidf_title_features, tfidf_title_features[doc_id])

    # np.argsort will return indices of 9 smallest distances
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    #pdists will store the 9 smallest distances
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]

    #data frame indices of the 9 smallest distace's
    df_indices = list(data.index[indices])
    for i in range(0, len(indices)):
        recommendation = {
            "ASIN": data['asin'].loc[df_indices[i]],
            "BRAND": data['brand'].loc[df_indices[i]],
            "Title": data['title'].loc[df_indices[i]],
            "Euclidean_similarity": pdists[i],
            "Image_URL": data['medium_image_url'].loc[df_indices[i]],
            "Model": 'tfidf'
        }
        recommendations_list.append(recommendation)
    save_recommendations_to_js(recommendations_list)

tfidf_model(12569, 20)
# in the output heat map each value represents the tfidf values of the label word, the color represents the intersection with inputs title

tfidf_model(15099, 20)
