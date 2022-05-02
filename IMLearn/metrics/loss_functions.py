import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    #length = len(y_true)
    #a = np.zeros(length,)
    #for i in range(length):
    #    a[i] = (y_true[i]-y_pred[i])**2
    #mse = (1/length)*(np.sum(a))

    #return mse
    num_samples = y_true.size
    squared_diff = y_true - y_pred
    squared_diff = squared_diff**2
    sam_samples = sum(squared_diff)
    return (1/num_samples)*sam_samples


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not
    Returns
    -------
    Misclassification of given predictions
    """
    error = 0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            error+=1
    if normalize == True:
        return error/len(y_true)

    else:
        return error




def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------

    Accuracy of given predictions

    """
    classes = []
    numClasses=0
    for i in range(len(y_true)):
        if y_true[i] not in classes:
            classes.append(y_true[i])
            numClasses+=1

    true = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            true+=1

    return true/numClasses






def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()

