import math
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        features = len(X[0])
        k = 0
        classes = []
        for i in range(len(y)):
            if y[i] not in classes:
                classes.append(y[i])
                k += 1

        self.classes_ = np.zeros(k, )
        for i in range(k):
            self.classes_[i] = classes[i]
        self.classes_.sort()

        # initialize self.pi

        self.pi_ = np.zeros(k, )
        for i in range(len(y)):
            self.pi_[np.where(self.classes_ == y[i])[0][0]] += 1
        self.pi_ = self.pi_ * (1 / k)


        #initialize self.mu


        self.mu_ = np.zeros((k, features))
        for i in range(k):
            self.mu_[i] = (X[y==self.classes_[i]]).mean(axis=0)
        #initialize vars

        self.vars_ = np.zeros((k, features))
        for i in range(k):
            self.vars_[i] = (X[y==self.classes_[i]]).var(axis=0)


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
        samples = len(X)
        y_pred = np.zeros(samples,)
        likelihood = self.likelihood(X)
        for i in range(samples):
            y_pred[i] = self.classes_[np.where(likelihood[i]== np.amax(likelihood[i]))[0][0]]
        return y_pred

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
        likelihood = np.zeros((len(X), len(self.classes_)))
        k = len(self.classes_)
        for s in range(len(X)):
            for c in range(k):
                probMult = 1
                a1 = self.pi_[c]
                for f in range(len(X[0])):
                    z = (-1/(2*self.vars_[c][f]))*((X[s][f] - self.mu_[c][f])**2)
                    prFeaturei = 1/math.sqrt(2*math.pi*self.vars_[c][f])*math.exp(z)
                    probMult=probMult*prFeaturei
                likelihood[s][c] = a1*probMult

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
<<<<<<< HEAD

        from ...metrics import misclassification_error
        y_pred = self._predict(X)
        loss = misclassification_error(y, y_pred)
        return loss

=======
        from ...metrics import misclassification_error
        raise NotImplementedError()
>>>>>>> c87be5d7872d40b4409d315bf2d2360bc8a3d675
