import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import csv
import pandas

def word_tokenizer(text):
    #tokenizes and stems the text
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
    return tokens


def cluster_sentences(sentences, nb_of_clusters=5):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
	                                    stop_words=stopwords.words('english'),
	                                    max_df=0.99,
	                                    min_df=0.01,
	                                    lowercase=True)
    #builds a tf-idf matrix for the sentences
    tfidf_matrix_1 = tfidf_vectorizer.fit_transform(sentences)
    tfidf_matrix = tfidf_matrix_1.todense()
    kmeans = FeatureAgglomeration(n_clusters=nb_of_clusters)
    kmeans.fit(tfidf_matrix)
    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)
    return dict(clusters)

import csv

with open(r'C:\Sales\SP.csv') as f:
  reader = csv.reader(f)
  Pre_sentence = list(reader)

flatten = lambda l: [item for sublist in l for item in sublist]
sentences = flatten(Pre_sentence)

with open(r'C:\Sales\Cat.csv') as g:
  reader_cat = csv.reader(g)
  Pre_Cat = list(reader_cat)
Cats = flatten(Pre_Cat)

if __name__ == "__main__":
    # sentences = ["Nature is beautiful","I like green apples",
	   #          "We should protect the trees","Fruit trees provide fruits",
	   #          "Green apples are tasty","My name is Dami"]
    nclusters = 19
    clusters = cluster_sentences(sentences, nclusters)
    for cluster in range(nclusters):
            print ("Grouped Engagements  ",cluster,":")
            for i,sentence in enumerate(clusters[cluster]):
                    print ("\tEngagement ", Cats[sentence],": ", sentences[sentence])
