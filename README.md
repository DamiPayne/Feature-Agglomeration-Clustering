# Feature Agglomeration Clustering
Reads CSV text files and uses a Tfidf vectoriser to semantically cluster like sentences, then uses a hierarchical clustering algorithm to assign the words to n clusters  

## Dependencies
Numpy (http://www.numpy.org/)

Sci-kit Learn (http://scikit-learn.org/stable/index.html) (you will need to compile numpy and scikit learn from source on windows)

Pandas (http://pandas.pydata.org/) 

NLTK (http://www.nltk.org/)

Matplotlib (https://github.com/matplotlib/matplotlib)

Virtual Env (https://virtualenv.pypa.io/en/stable/) (creating a virtual environment is my preferred method of installing dependencies)

## How to use it?

1. Install dependencies using pi
2. run python.exe > `import nltk` > `nltk.download()`
3. Download the stopwords corpus
4. run `Cluster.py` choose the CSV file you want to cluster and the number of clusters
5. View results
