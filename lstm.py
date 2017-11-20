from itertools import chain
import pathlib

import keras
import numpy as np

from .model import SentimentModel
# from .utils import


class _InnerLSTM:
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
    embedding_matrix : np.array[float, float], optional 
        A pre-trained matrix for word embedding weights. Use this if you want to pretrain the model
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
                 max_features=50000
                 embedding_dim=100,
                 batch_size=32,
                 layer_sizes=(64, 128, 64),
                 dropout=0.1,
                 embedding_matrix=None,
                 inner_activation='tanh',
                 outer_activation='sigmoid',
                 loss='binary_crossentropy',
                 metrics=['accuracy'],
                 optimizer='rmsprop'):
        
        if len(embedding_dim) < 1:
            raise ValueError('word embeddings must be at least 1-dimensional')
        
        self._max_features = max_features
        self._embedding_dim = embedding_dim
        
        self._batch_size = batch_size
        
        if len(layer_sizes) < 3:
            raise ValueError('there must be at least one hidden layer')

        if embedding_matrix and not (embedding_matrix.shape == (max_features+1, embedding_dim))
            raise ValueError(
                'embedding matrix is not the correct shape.expected {0} and got {1}'.format(
                    (max_features+1, embedding_dim), (embedding_matrix.shape)
                )
        
        input_ = keras.layers.Input(shape=(sequence_length,))
        
        if embedding_matrix:
            embedding_layer = keras.layers.Embedding(
                max_features+1,
                embedding_dim,
                weights=[embedding_matrix],
                input_length=sequence_length,
                trainable=False)
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
        
    def fit(self, corpus, *, validation_split=0.20, epochs=10, verbose=False):
        pass
        
    def predict_one(self, document):
        pass
        
    def save_path(self, path):
        self._model.save(path)
        # save other things
        
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
    name : str
        The name of the model.
    sequence_length : int
        The maximum length of a given sequence of words in a document. Data is
        padded to fit this shape.
    max_features : int
        The n most significant word vectors in the space.
    embedding_dim : int
        The size of the word embeddings for each word in the corpus.
    embedding_matrix : np.array[float, float], optional
        A pre-trained matrix for word embedding weights.
    layer_sizes : tuple[int, int]
        The sizes of all layers.
    dropout : float
        The dropout ratio.
    embedding_matrix : np.array[float, float], optional 
        A pre-trained matrix for word embedding weights. Use this if you want to pretrain the model
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
    
    def __init__(self, name, *args, **kwargs):
        self._name = name
        self._model = _InnerLSTM(*args, **kwargs)
        
    def save_path(self, path):
        path = pathlib.Path(path)
        path.mkdir(exist_ok=True)
        
        self._model.save_path(path / 'lstm' / self._name)
    
    @classmethod
    def load_path(cls, path):
        path = pathlib.Path(path)
        
        self= cls()
        self._name = path.name()
        self._model = _InnerLSTM.load_path(path)
        return self
    
    def fit(self, corpus, *, validation_split=0.20, epochs=10, verbose=False):
        # Make a w2v model and fit it here
        pass
    
    def predict(self, documents_and_chars):
        pass
    
    def predict_proba(self, documents_and_chars):
        pass
        
