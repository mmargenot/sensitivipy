from abc import ABCMeta, abstractmethod

class SentimentModel(metaclass=ABCMeta):
    """Abstract sentiment model.
    """
    @abstractmethod
    def save_path(self, path):
        """Save the model to a file path.
        
        Parameters
        ----------
        path : path-like
            The path to save the model to.
        """
        raise NotImplementedError('save_path')
    
    @classmethod
    @abstractmethod
    def load_path(cls, path):
        """Load a model from a file path
        
        Parameters
        ----------
        path : path-like
            The path to the saved model.
        
        Returns
        -------
        model : SentimentModel
            The deserialized sentiment model.
        """
    
    @abstractmethod
    def fit(self, corpus, *, validation_split=0.20, epochs=None, verbose=False):
        """Fit the model to some corpus
        
        Parameters
        ----------
        corpus : 
            The documents to fit the model to
        validation_split: float
            What percentage of the fitting data to hold apart as a validation set
        epochs : int, optional
            The number of passes the model runs over documents
        verbose : bool, optional
            Print verbose information to ``sys.stdout``
        **kwargs
            Model-specific arguments.
        """
        raise NotImplementedError('fit')
    
    @abstractmethod
    def predict(self, documents_and_chars):
        """Predict the sentiment for a set of documents
        
        Parameters
        ----------
        documents_and_chars : iterable[(Document, dict)]
            The documents and characteristics to predict.
            
        Returns
        -------
        sentiment_class : np.ndarray[int]
            The categorical class the document falls into for
            each (Document, characteristics) pair.
        """
        raise NotImplementedError('predict')
    def predict_proba(self, documents_and_chars):
        """Predict the probability of a given sentiment class for a set of documents
        
        Parameters
        ----------
        documents_and_chars : iterable[(Document, dict)]
            The documents and characteristics to predict.
        
        Returns
        -------
        sentiment_proba : np.ndarray[float]
            The probability of each class for each
            (Document, characteristics) pair.
        """
        raise NotImplementedError('predict_proba')
            
#    @abstractmethod
#    def predict_doc(self, documents_and_chars):
#        """Predict the sentiment for a set of documents
#        
#        Parameters
#        ----------
#        documents_and_chars : iterable[(Document, dict)]
#            The documents and characteristics to predict.
#            
#        Returns
#        -------
#        sentiment_class : np.ndarray[int]
#            The categorical class the document falls into for
#            each (Document, characteristics) pair.
#        """
#        raise NotImplementedError('predict')
#    def predict_doc_proba(self, documents_and_chars):
#        """Predict the probability of a given sentiment class for a set of documents
#        
#        Parameters
#        ----------
#        documents_and_chars : iterable[(Document, dict)]
#            The documents and characteristics to predict.
#        
#        Returns
#        -------
#        sentiment_proba : np.ndarray[float]
#            The probability of each class for each
#            (Document, characteristics) pair.
#        """
#        raise NotImplementedError('predict_proba')
