import gensim
import pandas as pd
import numpy as np

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
        """
        """
        # what do i want to take as input? gensim corpus, pandas dataframe, csv
        
        if not labels:
            # do a thing that gets the labels

        self._data = data
        self._labels = labels

        if pretrain_w2v:
            # feed my data into a w2v model
            # self._embedding_matrix = thing


        pass
    def load_path(self, path):
        pass

    @class_method(cls, path):
        self = cls()
        self_model = # load the path
        return self


    def _build_w2v(self):
        pass

    def __iter__(self):
        # yield some stuff; generators nice
        pass
