import os.path
import gdown

output_path = 'tops_fashion.json'

if not os.path.exists(output_path):
    file_id = '17JgsMrRlRtEZZdBSsKAcqfa3_NFakzQU'
    
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)

import pandas as pd

data = pd.read_json(output_path)

print('Number of data points : ', data.shape[0])
print()
print('Number of features/variables:', data.shape[1])

data.columns 

data = data[['asin', 'brand', 'color', 'medium_image_url', 'product_type_name', 'title', 'formatted_price']]

print('Number of data points : ', data.shape[0])
print()
print('Number of features:', data.shape[1])
print()
data.head() 

print(data['product_type_name'].describe())

print(data['product_type_name'].unique())

from collections import Counter

product_type_count = Counter(list(data['product_type_name']))
product_type_count.most_common(10)

print(data['brand'].describe())

brand_count = Counter(list(data['brand']))
brand_count.most_common(10)

print(data['color'].describe())

color_count = Counter(list(data['color']))
color_count.most_common(10)

print(data['formatted_price'].describe())

price_count = Counter(list(data['formatted_price']))
price_count.most_common(10)

print(data['title'].describe())

data.isnull().sum()

data.to_pickle('180k_apparel_data')


data = data.loc[~data['formatted_price'].isnull()]
print('Number of data points After eliminating price = NULL :', data.shape[0])

data =data.loc[~data['color'].isnull()]
print('Number of data points After eliminating color = NULL :', data.shape[0])

data.to_pickle('28k_apparel_data')

print(sum(data.duplicated('title')))

data = pd.read_pickle('28k_apparel_data')
data.head()

data_sorted = data[data['title'].apply(lambda x: len(x.split())>4)]
print("After removal of products with short description:", data_sorted.shape[0])

data_sorted.sort_values('title',inplace=True, ascending=False)
data_sorted.head()

indices = []
for i, row in data_sorted.iterrows(): 
    indices.append(i)

import itertools

duplicates = []
i = 0
j = 0
num_data_points = data_sorted.shape[0]  

while i < num_data_points and j < num_data_points:

    previous_i = i

    a = data['title'].loc[indices[i]].split()

    j = i+1
    while j < num_data_points:

        b = data['title'].loc[indices[j]].split()

        length = max(len(a), len(b))

        count  = 0

 
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



    if model == 'bag_of_words':
        labels = values
    elif model == 'tfidf':
        labels = []
        for x in vec2.keys():
           
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
    print(recommendations_list)

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

def bag_of_words_model(num_results):
    recommendations_list = []
  
    doc_id_df = pd.read_csv('doc_id.csv')
    doc_id = doc_id_df['doc_id'].iloc[0] 
    pairwise_dist = pairwise_distances(bow_title_features, bow_title_features[doc_id], metric='cosine', n_jobs=-1)

    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
    
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

bag_of_words_model(20)



tfidf_title_vectorizer = TfidfVectorizer(min_df = 0)
tfidf_title_features = tfidf_title_vectorizer.fit_transform(data['title'])
tfidf_title_features.shape


def tfidf_model(doc_id, num_results):
    recommendations_list = []
    
    pairwise_dist = pairwise_distances(tfidf_title_features, tfidf_title_features[doc_id])

    # np.argsort will return indices of 9 smallest distances
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    #pdists will store the 9 smallest distances
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]

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

tfidf_model(15099, 20)
