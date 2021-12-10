import pickle
import gensim.downloader as api

with open('trainTest.pk', 'rb') as f:
    classes = pickle.load(f)
    train = pickle.load(f)
    test = pickle.load(f)

wv = api.load('word2vec-google-news-300')

print(wv['king'])
print(type(wv['king']))