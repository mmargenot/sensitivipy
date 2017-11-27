import gensim
import pandas as pd
import numpy as np
import pathlib


class Vocabulary:

    def __init__(self, data, embedding_dim=100, min_count=10, workers=2):
        self.trained = False 
        self.embedding_dim = embedding_dim
        # check the data first
        
        # once the data has been checked
        w2v = gensim.models.Word2Vec(size=embedding_dim, min_count=min_count, workers=workers) 
        w2v.build_vocab(data)
        self._model = w2v

    def build_model(self, data=None):
        if not data:
            data = self.data
        model = self._model  
        model.train(
            data,
            total_examples=len(data),
            epochs = model.iter
        )
        self.trained = True

    def save_path(self, path):
        w2v = self._model
        embedding_matrix = self._embedding_matrix
        path = pathlib.Path(path)
        w2v.save(str(path.resolve())+'w2v') # might not be necessary
        
        with open(path.with_suffix('.embedding_matrix')) as f:
            np.savez(
                f,
                embedding_matrix=embedding_matrix
            )

    @class_mathod
    def load_path(cls, path):
        self = cls()
        # load the gensim model
        # load whatever numpy arrays are included

        self.trained = True

        return self

class Corpus:
    """The container for any document to pass into a SentimentModel

    Parameters
    ----------
    data : gensim.corpora, pd.DataFrame, or path-like    
        The data to store within the Corpus object. May include labels
        as well
    labels : np.array, pd.Series, optional
        The labels for each observations within data. Must be included
        if labels are not present within data.
    pretrain_w2v : bool, optional
        Do you expect to build your model with pretrained word vectors?

    Notes
    -----
    """

    def __init__(self, data, labels=None, pretrain_w2v=False):
        # what do i want to take as input? gensim corpus, pandas dataframe, csv
        
        if not labels:
            # do a thing that gets the labels

        self._data = data
        self._labels = labels
        
        w2v = gensim.models.Word2Vec(self._data, 
        if pretrain_w2v:
            # feed my data into a w2v model
            # self._embedding_matrix = thing


        pass

    def save_path(self, path):
        pass

    @class_method
    def load_path(cls, path):
        self = cls()
        self_model = # load the path
        return self

    def _build_w2v(self):
        pass

    def __iter__(self):
        # yield some stuff; generators nice
        pass

