import re
import string
import ast
import numpy as np
import pandas as pd
import copy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
from gensim.models import Doc2Vec
from nltk.stem import PorterStemmer


class DataCleaner(object):
    def __init__(self, data, **params):
        """ Data should be a pandas Series object"""
        self.data = data
        self.params = params.copy()
        self.stemmer = PorterStemmer()
        
    def _fill_urls(self, tweet_text):
        return re.sub(r"http\S+", 'URL', tweet_text)
    
    def _fill_usernames(self, tweet_text):
        return re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)', 'USERNAME', tweet_text)

    def _truncate_extra_letters(self, tweet_text):
        return re.sub(r'(.)\1+', r'\1\1', tweet_text)
    
    def _remove_punctuation(self, tweet_text):
        return re.sub('['+string.punctuation.replace('#', '')+']', '', tweet_text)
    
    def _fill_hashtags(self, tweet_text):
        return re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9]+)', 'HASHTAG', tweet_text)
    
    def _tokenize(self, tweet_text):
        return tweet_text.split()
    
    def _stem_word_list(self, word_list):
        return [self.stemmer.stem(word) for word in word_list]

    def clean_data(self):
        data = self.data.copy()
        data = data.apply(self._fill_urls)
        data = data.apply(self._fill_usernames)
        data = data.apply(self._truncate_extra_letters)
        data = data.apply(self._remove_punctuation)
        if not self.params['keep_hashtags']:
            data = data.apply(self._fill_hashtags)
        data = data.apply(self._tokenize)
        data = data.apply(self._stem_word_list)
        return data

class SentimentData:
    def __init__(self,
                 data_dir,
                 filename="training.1600000.processed.noemoticon.csv",
                 test_size=0.3):
        
        self.data_dir = data_dir
        self.filename = filename
        self.test_size = test_size
        
    def load_data(self):
        try:
            self.data, self.cleaned_data = self.load_data_from_pickle()
            #cleaned_data = pd.read_csv(
            #    self.data_dir+'cleaned_data.csv',
            #    index_col=0,
            #)
            #cleaned_data.cleaned_tweets = cleaned_data.cleaned_tweets.apply(ast.literal_eval)
            #self.cleaned_data = cleaned_data
            self.cleaned_tweets = self.cleaned_data.cleaned_tweets
            self.sentiment = self.cleaned_data.sentiment
            
            #self.data = pd.read_csv(
            #    self.data_dir+self.filename,
            #    index_col=2,
            #    names=['polarity', 'tweet_id', 'query', 'user', 'tweet_text'],
            #    encoding='latin-1'
            #)
            self.complete_tweets = self.data.tweet_text
            
        except FileNotFoundError:
            self.data = pd.read_csv(
                self.data_dir+self.filename,
                index_col=2,
                names=['polarity', 'tweet_id', 'query', 'user', 'tweet_text'],
                encoding='latin-1'
            )
            self.complete_tweets = self.data.tweet_text
            
            data_cleaner = DataCleaner(self.complete_tweets, keep_hashtags=False)

            self.cleaned_tweets = data_cleaner.clean_data()
            self.cleaned_tweets.name = 'cleaned_tweets'
            self.sentiment = self.data.polarity[self.data.polarity != 2]
            self.sentiment.name = 'sentiment'
            self.cleaned_tweets = self.cleaned_tweets[self.sentiment != 2]

            self.cleaned_data = pd.concat([self.cleaned_tweets, self.sentiment], axis=1)
            self.cleaned_data.to_csv(self.data_dir+'cleaned_data.csv')
            self.save_data_to_pickle()
            
    def save_data_to_pickle(self):
        self.cleaned_data.to_pickle(self.data_dir+'cleaned_data_pkl')
        self.data.to_pickle(self.data_dir+'data_pkl')
        
    def load_data_from_pickle(self):
        data = pd.read_pickle(self.data_dir+'data_pkl')
        cleaned_data = pd.read_pickle(self.data_dir+'cleaned_data_pkl')
        return data, cleaned_data
         
    def concatenate_phrases(self, concat=False):
        if concat:
            self.cleaned_data.cleaned_tweets = self.cleaned_data.cleaned_tweets.apply(' '.join)
            self.cleaned_tweets = self.cleaned_data.cleaned_tweets
        
    def labelize_data(self, data, label_type):
        labelized = []
        for i,v in enumerate(data):
            label = '%s_%s'%(label_type,i)
            labelized.append(gensim.models.doc2vec.LabeledSentence(v, [label]))
        return labelized
    
    def split_and_label_data(self, label=True):
        self.feature_train, self.feature_test, self.label_train, self.label_test = train_test_split(
            self.cleaned_tweets.reset_index().cleaned_tweets,
            self.sentiment.reset_index().sentiment,
            test_size=self.test_size
        )
        
        self.complete_feature_train = self.data.tweet_text.iloc[self.feature_train.index]
        self.complete_feature_test = self.data.tweet_text.iloc[self.feature_test.index]
        
        if label:
            self.labeled_training_features = self.labelize_data(self.feature_train, 'TRAIN')
            self.labeled_testing_features = self.labelize_data(self.feature_test, 'TEST')
            
    def label_all_data(self):
        self.all_data_labeled = self.labelize_data(self.cleaned_tweets, 'CORPUS_DOC')
        

class BagOfWords:
    def __init__(self,
                 data,
                 max_features,
                 ngram_range=(1, 2),
                 analyzer='word'):
        
        self.data = copy.deepcopy(data)
        self.data.concatenate_phrases(concat=True)
        
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            stop_words='english',
            max_features=max_features
        )
        
    def _fit_transform(self, data):
        return self.vectorizer.fit_transform(data)
    
    def _transform(self, data):
        return self.vectorizer.transform(data)
        
    def transform_data(self):
        fit_transform = self._fit_transform
        transform = self._transform
        
        self.data.split_and_label_data(label=False)
        
        self.transformed_feature_train = fit_transform(self.data.feature_train)
        self.transformed_feature_test = transform(self.data.feature_test)
        
        self.complete_feature_train = self.data.complete_feature_train
        self.complete_feature_test = self.data.complete_feature_test
        
    def _fit_transform_all(self):
        self.all_data_transformed = self.vectorizer.fit_transform(self.data.cleaned_tweets)
        
    def fit_fully(self):
        self._fit_tranform_all()
        return self.all_data_transformed, self.data.sentiment
        
class Doc2VecBuilder:
    def __init__(self,
                 data,
                 embedding_dim,
                 min_count):
        self.data = copy.deepcopy(data)
        self.embedding_dim = embedding_dim
        self.min_count=min_count
        
    def train_doc2vec(self):
        self.data.split_and_label_data(label=True)
        d2v = Doc2Vec(size=self.embedding_dim, min_count=self.min_count)
        d2v.build_vocab(self.data.labeled_training_features)
        d2v.train(
            self.data.labeled_training_features,
            total_examples=len(self.data.labeled_training_features),
            epochs=d2v.iter
        )
        self.d2v = d2v
        return d2v
    
    def train_full_doc2vec(self):
        self.data.label_all_data()
        d2v = Doc2Vec(size=self.embedding_dim, min_count=self.min_count)
        d2v.build_vocab(self.data.all_data_labeled)
        d2v.train(
            self.data.all_data_labeled,
            total_examples=len(self.data.all_data_labeled),
            epochs=d2v.iter
        )
        self.full_d2v = d2v
        return d2v
    
    def save_d2v(self, filename='d2v_model'):
        if self.d2v:
            self.d2v.save(self.data.data_dir+filename)
        else:
            raise ValueError("Need to build your Doc2Vec model first!")
    def load_d2v(self, filename='d2v_model'):
        try:
            self.d2v = Doc2Vec.load(self.data.data_dir+filename)
        except FileNotFoundError('You do not have a model saved!'):
            return
        
    def vector_representations(self):
        self.train_array = np.zeros((len(self.data.labeled_training_features), self.embedding_dim))
        for i, document in enumerate(self.data.labeled_training_features):
            self.train_array[i,:] = self.d2v.docvecs[document.tags[0]]
        self.train_array_labels = self.data.label_train.values
        
        self.test_array = np.zeros((len(self.data.labeled_testing_features), self.embedding_dim))
        for i, document in enumerate(self.data.labeled_testing_features):
            self.test_array[i,:] = self.d2v.infer_vector(document.words)
        self.test_array_labels = self.data.label_test.values
        return self.train_array, self.train_array_labels, self.test_array, self.test_array_labels
        
        
        
        