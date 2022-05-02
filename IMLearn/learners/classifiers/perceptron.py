from __future__ import annotations
from typing import Callable
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import loss_functions


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    training_loss_: array of floats
        holds the loss value of the algorithm during training.
        training_loss_[i] is the loss value of the i'th training iteration.
        to be filled in `Perceptron.fit` function.

    """
    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        max_iter): int, default = 1000
            Maximum number of passes over training data

        callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by Perceptron. To be set in `Perceptron.fit` function.
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to to given samples. Iterate over given data as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """
        dim_w=len(X[0])
        old_X = X
        if self.include_intercept_==True:
            new_coor = [1]
            new_X = np.zeros((len(X), len(X[0])+1))
            for i in range(len(X)):
                new_xi = np.r_[new_coor, X[i]]
                new_X[i] = new_xi
            dim_w +=1
            X = new_X
        w = np.zeros(dim_w,)
        t=1
        self.coefs_ = w
        while t < self.max_iter_ :
            transformed = False
            for i in range(len(y)):
                if (np.dot(w.transpose(), X[i]))*y[i]<=0:
                    w = w+y[i]*X[i]
                    self.fitted_ = True
                    self.callback_(self,old_X[i], y[i])
                    self.coefs_=w
                    transformed = True
                    break
            if transformed == False:
                self.callback_(self, old_X[i], y[i])
                self.coefs_=w
                return
            t=t+1
        self.coefs_ = w
        return




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
        prediction = np.zeros(len(X),)
        if self.include_intercept_ == True:
            new_coefs = np.delete(self.coefs_, 0)
            if X.ndim == 1:
                a = np.matmul(X, new_coefs)+ self.coefs_[0]
                a2 = np.zeros(1,)
                a2[0]=a
                prediction= a2
            else:
                prediction= np.matmul(X, new_coefs)+ self.coefs_[0]
        else:
            prediction= np.matmul(X,self.coefs_)
        for i in range(len(prediction)):
            if prediction[i]<0:
                prediction[i]= -1
            if prediction[i]>0:
                prediction[i] = 1
        return prediction

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
        y_pred = self._predict(X)
        error = loss_functions.misclassification_error(y, y_pred)
        return error
