from __future__ import annotations
import numpy as np
import copy
import math
from math import e
from numpy.linalg import inv, det, slogdet






class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:

        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        length = len(X)
        sum = np.sum(X)
        X2 = copy.deepcopy(X)
        mu = sum / length
        X2 = X2-mu
        X2 = np.multiply(X2,X2)
        sum2 = np.sum(X2)
        if self.biased_==True:
            var = sum2 / (length)
        else:
            var=sum2/(length-1)
        self.mu_=mu
        self.var_=var
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        mu = self.mu_
        var = self.var_
        X2 = copy.deepcopy(X)
        X2 = X2 - mu
        X2 = np.multiply(X2, X2)
        X2 = X2/var
        X2 = X2*(-0.5)
        lenght = len(X)
        X3 = np.zeros(lenght,)
        for i in range (lenght):
            X3[i]= (e**X2[i])/(math.sqrt(var*2*math.pi))
        return X3


    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        X2 = copy.deepcopy(X)
        X2 = X2 - mu
        X2 = np.multiply(X2,X2)
        a = (np.sum(X2)) / (sigma*(-2))
        lenght = len(X)
        devisor =(math.sqrt(sigma * 2 * math.pi))**lenght
        log_likelihood= math.log(1/devisor)+a
        return log_likelihood



class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        X2 = copy.deepcopy(X)
        X2= X2.transpose()
        dim = len(X2[0])
        self.mu_=np.zeros(len(X2), )
        for i in range(len(X2)):
            self.mu_[i] = (np.sum(X2[i]))/dim
        self.mu_=self.mu_
        X= X-self.mu_
        sigma = (np.dot(X.transpose(), X))/(len(X)-1)
        self.cov_ = sigma
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        length = len(X)
        X3 = np.zeros(length, )
        det = np.linalg.det(self.cov_)
        for i in range(length):
            X2 = X[i]-self.mu_
            in_power = (np.matmul(np.matmul(X2.transpose(),inv(self.cov_)),X2))/(-2)
            X3[i] = (math.e**in_power)/math.sqrt(((2*math.pi)**len(X2))*det)
        return X3




    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        length = len(X)
        X3 = np.zeros(length, )
        det = np.linalg.det(cov)
        cov_inv =inv(cov)
        for i in range(length):
            X2 = X[i] - mu
            X3[i] = (np.matmul(np.matmul(X2.transpose(), cov_inv),
                                  X2)) / (2)
        dim = length*len(X[0])
        a1 = (((-1)*dim)/2)*math.log(2*math.pi)
        log_likelihood = a1 - (length/2)*(math.log(1/det)) - np.sum(X3)
        return log_likelihood

