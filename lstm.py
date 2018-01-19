from itertools import chain
import pathlib

import keras
import numpy as np
from sklearn.model_selection import train_test_split

from model import SentimentModel, Vocabulary
#from .utils import


class _InnerLSTM:
    """An LSTM model for document sentiment modeling which trains on texts with (x, time) dimensions
    
    Parameters
    ----------
    vocabulary : Vocabulary, optional
        The vocabulary used to pretrain the embedding layer. Include if you want
        to pretrain for better results.
    sequence_length : int
        The maximum length of a given sequence of words in a document. Data is
        padded to fit this shape.
    max_features : int
        The n most significant word vectors in the space.
    embedding_dim : int
        The size of the word embeddings for each word in the corpus.
    layer_sizes : tuple[int, int]
        The sizes of all layers.
    dropout : float
        The dropout ratio.
        with word vectors for increased accuracy from the beginning.
    inner_activation : str
        The activation function of the hidden layers. Must be one of:
        'tanh', 'softplus, 'softsign', 'relu', 'sigmoid', 'hard_sigmoid', or
        'linear'.
    outer_activation : str
        The activation function of the output layer. Must be one of:
        'tanh', 'softplus, 'softsign', 'relu', 'sigmoid', 'hard_sigmoid', or
        'linear'.
    loss : {'binary_crossentropy', 'categorical_crossentropy'}
        The loss function.
    metrics : {}
        Metrics to calculate.
    optimizer : str
        The optimizer to use.
        
    Notes
    -----
    """
    def __init__(self,
                 *,
                 vocabulary=None,
                 max_features=50000,
                 embedding_dim=100,
                 sequence_length=70,
                 batch_size=32,
                 layer_sizes=(64, 128, 64),
                 dropout=0.1,
                 inner_activation='tanh',
                 outer_activation='sigmoid',
                 loss='binary_crossentropy',
                 metrics=['accuracy'],
                 optimizer='rmsprop'):
        
        if embedding_dim < 1:
            raise ValueError('word embeddings must be at least 1-dimensional')
        
        self._max_features = max_features
        self._embedding_dim = embedding_dim
        
        self._batch_size = batch_size
       
        if len(layer_sizes) < 3:
            raise ValueError('there must be at least one hidden layer')

        input_ = keras.layers.Input(shape=(sequence_length,))
        
        if vocabulary:
            if not vocabulary.size == embedding_dim:
                raise ValueError('embedding dimension not the same as vocabulary embedding dimension')
            embedding_layer = keras.layers.Embedding(
                max_features+1,
                embedding_dim,
                weights=[vocabulary.embedding_matrix],
                input_length=sequence_length,
                trainable=False
            )(input_)
        else:
            embedding_layer = keras.layers.Embedding(
                max_features+1,
                embedding_dim,
                input_length=sequence_length
            )(input_)

        lstm = keras.layers.LSTM(
            layer_sizes[0],
            dropout=dropout,
            activation=inner_activation,
            return_sequences=True
        )(embedding_layer)
        
        for size in layer_sizes[1:-1]:
            lstm = keras.layers.LSTM(
                size,
                dropout=dropout,
                activation=inner_activation,
                return_sequences=True
            )(lstm)
        
        lstm = keras.layers.LSTM(
            layer_sizes[-1],
            dropout=dropout,
        )(lstm)
        
        sentiment = keras.layers.Dense(
            1,
            activation=outer_activation,
            name='sentiment'
        )(lstm)
        
        self._model = model = keras.models.Model(
            inputs=input_,
            outputs=sentiment,
        )
        model.compile(
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
        )
        
    def fit(self, corpus, sentiment, *, validation_split=0.20, epochs=10, verbose=False):
        x_train, x_test, y_train, y_test = train_test_split(corpus,
                                                            sentiment,
                                                            test_size=validation_split)

        self._model.fit(x_train,
                        y_train,
                        validation_set=(x_test, y_test),
                        nb_epoch=epochs,
                        verbose=verbose)
        
    def predict_one(self, phrase):
        pass
        
    def save_path(self, path):
        self._model.save(path/'lstm')
        
    @classmethod
    def load_path(cls, path):
        self = cls()
        self._model = keras.models.load_model(path)
        # load other things
            
        return self

class SentimentLSTM(SentimentModel):
    """An LSTM model for document sentiment modeling which trains on texts with (x, time) dimensions
    
    Parameters
    ----------
    sequence_length : int
        The maximum length of a given sequence of words in a document. Data is
        padded to fit this shape.
    max_features : int
        The n most significant word vectors in the space.
    embedding_dim : int
        The size of the word embeddings for each word in the corpus.
    layer_sizes : tuple[int, int]
        The sizes of all layers.
    dropout : float
        The dropout ratio.
    inner_activation : str
        The activation function of the hidden layers. Must be one of:
        'tanh', 'softplus, 'softsign', 'relu', 'sigmoid', 'hard_sigmoid', or
        'linear'.
    outer_activation : str
        The activation function of the output layer. Must be one of:
        'tanh', 'softplus, 'softsign', 'relu', 'sigmoid', 'hard_sigmoid', or
        'linear'.
    loss : {'binary_crossentropy', 'categorical_crossentropy'}
        The loss function.
    metrics : {}
        Metrics to calculate.
    optimizer : str
        The optimizer to use.
        
    Notes
    -----
    """
    version = 0
    
    def __init__(self, w2v=None, *args, **kwargs):
        if w2v:
            self._vocabulary = SentimentVocab(w2v)
        else:
            self._vocabulary = None
        self._model = _InnerLSTM(*args, **kwargs)
        
    def save_path(self, path):
        path = pathlib.Path(path)
        path.mkdir(exist_ok=True)
        (path/'version').write_text(str(self.version))
        self._model.save_path(path)
        if self._vocabulary:
            self._vocabulary.save_path(path/'vocabulary'),
    
    @classmethod
    def load_path(cls, path):
        path = pathlib.Path(path)
        
        self= cls()
        self._model = _InnerLSTM.load_path(path+'lstm')
        try:
            self._vocabulary = Vocabulary.load_path(path+'vocabulary')
        except FileNotFoundError:
            self._vocabulary = None
        return self
    
    def fit(self, corpus, sentiment, *, validation_split=0.20, epochs=10, verbose=False):
        self._model.fit(corpus,
                        sentiment,
                        validation_split=validation_split,
                        epochs=epochs,
                        verbose=False)
    
    def predict(self, phrases):
        model = self._model
        predictions = []
        # check dims

        # call model.predict_one for each phrase

        return predictions
    
    def predict_proba(self, documents_and_chars):
        pass

class SentimentVocab(Vocabulary):
    """Vocabulary object for holding a Word2Vec on Twitter data

    Parameters
    ----------
    w2v : Word2Vec
        The vector representation of the twitter corpus

    Notes
    -----
    """
    def __init__(self, w2v):
        self._w2v = w2v
        # create tokenizer
        # create embedding matrix

    def save_path(self, path):
        path = pathlib.Path(path)
        self._w2v.save(path/'w2v')
        # save embedding matrix
        # save tokenizer

    @classmethod
    def load_path(cls, path):
        path = pathlib.Path(path)

        self = cls()
        self._w2v = Word2Vec.load(path+'w2v')
        # load embedding matrix
        # load tokenizer

