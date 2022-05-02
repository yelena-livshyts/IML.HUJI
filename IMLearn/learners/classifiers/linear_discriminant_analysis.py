import math
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None


    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        #initialize self.classes
        k =0
        classes= []
        for i in range(len(y)):
            if y[i] not in classes:
                classes.append(y[i])
                k+=1

        self.classes_ = np.zeros(k,)
        for i in range(k):
            self.classes_[i] = classes[i]
        self.classes_.sort()

        #initialize self.pi and self.mu

        self.pi_ = np.zeros(k,)
        self.mu_ = np.zeros((k,len(X[0])))
        for i in range(len(y)):
            index = np.where(self.classes_==y[i])[0][0]
            self.pi_[index]+=1
        self.pi_ = self.pi_* (1/len(y))
        for i in range(k):
            mk = 0
            for j in range(len(X)):
                if y[j] == self.classes_[i]:
                    self.mu_[i] = self.mu_[i]+X[j]
                    mk+=1
            if(mk !=0):
                self.mu_[i] = self.mu_[i]*(1/mk)
        cov_temp = np.zeros((len(X[0]), len(X[0])))
        for i in range(len(X)):
            classof_yi = np.where(self.classes_ == y[i])
            classof_yi= classof_yi[0][0]
            cov_temp = cov_temp+ np.matmul((X[i] - self.mu_[classof_yi]).reshape((-1,1)), (X[i] - self.mu_[classof_yi]).reshape((1,-1)))
        self.cov_ = cov_temp*(1/(len(X)-k))
        self._cov_inv= inv(self.cov_)




    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        responses = np.zeros(len(X),)
        likelihood = self.likelihood(X)
        for i in range(len(X)):
            responses[i] = self.classes_[np.where(likelihood[i]== np.amax(likelihood[i]))[0][0]]
        return responses



    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.
        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        d = len(X)
        detSigma = det(self.cov_)
        likelihood = np.zeros((len(X), len(self.classes_)))
        dim = len(X[0])

        for i in range(d):
            for j in range(len(self.classes_)):
                probYisJ =self.pi_[j]
                a1 = 1/(math.sqrt(detSigma*((2*math.pi)**dim)))
                xi_minus_muj = X[i]- self.mu_[j]
                a2 = (-1/2)*(np.matmul(np.matmul((xi_minus_muj).reshape((1,-1)) , self._cov_inv), xi_minus_muj.reshape((-1,1))))
                probXiifYisJ = a1*math.exp(a2)
                likelihood[i][j]= probXiifYisJ*probYisJ
        return likelihood




    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
<<<<<<< HEAD
        y_pred = self._predict(X)
        loss = misclassification_error(y, y_pred)
        return loss
=======
        raise NotImplementedError()
>>>>>>> c87be5d7872d40b4409d315bf2d2360bc8a3d675
