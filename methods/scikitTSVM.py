from sklearn.base import BaseEstimator
import sklearn.metrics
import random as rnd
import numpy
from sklearn.linear_model import LogisticRegression as LR
from qns3vm import QN_S3VM

class SKTSVM(BaseEstimator):
    """
    Scikit-learn wrapper for transductive SVM (SKTSVM)
    
    Wraps QN-S3VM by Fabian Gieseke, Antti Airola, Tapio Pahikkala, Oliver Kramer (see http://www.fabiangieseke.de/index.php/code/qns3vm) 
    as a scikit-learn BaseEstimator, and provides probability estimates using Platt scaling

    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    kernel : string, optional (default='rbf')
         Specifies the kernel type to be used in the algorithm.
         It must be 'linear' or 'rbf'

    gamma : float, optional (default=0.0)
        Kernel coefficient for 'rbf'

    lamU: float, optional (default=1.0) 
        cost parameter that determines influence of unlabeled patterns
        must be float >0

    probability: boolean, optional (default=False)
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, and will slow down that method.
    """
    
    # lamU -- cost parameter that determines influence of unlabeled patterns (default 1, must be float > 0)
    def __init__(self, kernel = 'RBF', C = 1e-4, gamma = 0.5, lamU = 1.0, probability=True):
        self.random_generator = rnd.Random()
        self.kernel = kernel
        self.C = C
        self.gamma = gamma 
        self.lamU = lamU
        self.probability = probability
        
    def fit(self, X, y): # -1 for unlabeled
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target vector relative to X
            Must be 0 or 1 for labeled and -1 for unlabeled instances 

        Returns
        -------
        self : object
            Returns self.
        """
        
        # http://www.fabiangieseke.de/index.php/code/qns3vm
        
        unlabeledX = X[y==-1, :].tolist()
        labeledX = X[y!=-1, :].tolist()
        labeledy = y[y!=-1]
        
        # convert class 0 to -1 for tsvm
        labeledy[labeledy==0] = -1
        labeledy = labeledy.tolist()
        
        if 'rbf' in self.kernel.lower():
            self.model = QN_S3VM(labeledX, labeledy, unlabeledX, self.random_generator, lam=self.C, lamU=self.lamU, kernel_type="RBF", sigma=self.gamma)
        else:
            self.model = QN_S3VM(labeledX, labeledy, unlabeledX, self.random_generator, lam=self.C, lamU=self.lamU)
            
        self.model.train()
        
        # probabilities by Platt scaling
        if self.probability:
            self.plattlr = LR()
            preds = self.model.mygetPreds(labeledX)
            self.plattlr.fit( preds.reshape( -1, 1 ), labeledy )
        
    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        
        if self.probability:
            preds = self.model.mygetPreds(X.tolist())
            return self.plattlr.predict_proba(preds.reshape( -1, 1 ))
        else:
            raise RuntimeError("Probabilities were not calculated for this model - make sure you pass probability=True to the constructor")
        
    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Class labels for samples in X.
        """
        
        y = numpy.array(self.model.getPredictions(X.tolist()))
        y[y == -1] = 0
        return y
    
    def score(self, X, y, sample_weight=None):
        return sklearn.metrics.accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    