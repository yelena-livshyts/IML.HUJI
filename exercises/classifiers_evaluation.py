
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
<<<<<<< HEAD
pio.templates.default = "simple_white"
=======
>>>>>>> c87be5d7872d40b4409d315bf2d2360bc8a3d675
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)
<<<<<<< HEAD


=======
>>>>>>> c87be5d7872d40b4409d315bf2d2360bc8a3d675


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        losses_temp = []
        filename = r"C:\Users\elena\PycharmProjects\IML.HUJI\datasets\%s" % f
        X, y = load_dataset(filename)


        def myCallback(perceptron, X1, y1):
            loss = perceptron.loss(X, y)
            losses_temp.append(loss)


        parceptron = Perceptron(callback = myCallback).fit(X,y)
        # Fit Perceptron and record loss in each fit iteration
        losses = []
<<<<<<< HEAD
        losses = losses_temp
        if n == "Linearly Separable":
            print(losses)
        iterations = np.zeros(len(losses),)
        for i in range(len(losses)):
            iterations[i]=i+1
        # Plot figure
        go.Figure([go.Scatter(x=iterations, y=losses, mode='lines',
                              name=r'$\widehat\mu$')],

                  layout=go.Layout(
                      title=r"$\text{ loss over training set after t iterations on %s}$" %n,
                      xaxis_title="$t\\text{ - iteration number}$",
                      yaxis_title="$\\text{ loss }$",
                      height=300)).show()

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))
=======
        raise NotImplementedError()

        # Plot figure of loss as function of fitting iteration
        raise NotImplementedError()
>>>>>>> c87be5d7872d40b4409d315bf2d2360bc8a3d675

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        filename = r"C:\Users\elena\PycharmProjects\IML.HUJI\datasets\%s" % f
        X, y = load_dataset(filename)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X,y)
        res1 = lda.predict(X)


        gaussianNB = GaussianNaiveBayes()
        gaussianNB.fit(X,y)

        res2=gaussianNB.predict(X)



        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        accLda = accuracy(y, res1)
        accGaus = accuracy(y, res2)

        subplotTitles= ["gaussianNB, accuracy = "+ str(accGaus) , "LDA, accuracy = "+ str(accLda)]
        models = [gaussianNB, lda]
        symbols = np.array(['204', 'diamond-cross', 'star-triangle-up'])

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[rf"$\textbf{{{m}}}$" for m in
                                            subplotTitles],
                            horizontal_spacing=0.05, vertical_spacing=.07)
        for i, m in enumerate(models):
            k=m.predict
            y_pred = k(X).astype(int)
            fig.add_traces(
                            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                       showlegend=False,
                                       marker=dict(color=y_pred, symbol=symbols[y],
                                                   colorscale=["red", "yellow", "blue"],
                                                   line=dict(color="black",
                                                             width=1))),
                           rows=1, cols=i+1)

            if i == 0:
                fig.add_traces(
                    go.Scatter(
                        x=m.mu_.transpose()[0],
                        y=m.mu_.transpose()[1],
                        mode="markers",
                        line=go.scatter.Line(color="gray"),
                        showlegend=False,
                        marker=dict(color="black", symbol="x",
                                    line=dict(color="black",
                                              width=1))),
                    rows=1, cols=i + 1
                )
                for k in range(len(m.mu_)):
                    fig.add_traces(get_ellipse(m.mu_[k],np.diag(m.vars_[k])) , rows=1, cols=i+1)
            if i==1:
                fig.add_traces(
                    go.Scatter(
                        x=m.mu_.transpose()[0],
                        y=m.mu_.transpose()[1],
                        mode="markers",
                        line=go.scatter.Line(color="gray"),
                        showlegend=False,
                        marker=dict(color="black", symbol="x",
                                    line=dict(color="black",
                                              width=1))),
                    rows=1, cols=i + 1
                )
                for k in range(len(m.mu_)):
                    fig.add_traces(get_ellipse(m.mu_[k],m.cov_), rows=1, cols=i+1)


        fig.update_layout(
            title=rf"$\textbf{{predictions of gaussian naive base and lda on %s}}$" %f,
            margin=dict(t=100)) \
            .update_xaxes(visible=True).update_yaxes(visible=True)
        fig.show()


        # Add traces for data-points setting symbols and colors

        # Add `X` dots specifying fitted Gaussians' means

        # Add ellipses depicting the covariances of the fitted Gaussians

        # Add traces for data-points setting symbols and colors
        raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    # compare_gaussian_classifiers()

    X = [[1,1], [1,2], [2,3], [2,4], [3,3], [3,4]]
    y = [0,0,1,1,1,1]
    X2 = np.array(X)
    y2 = np.array(y)
    gas = GaussianNaiveBayes()
    gas.fit(X2, y2)
    print(gas.vars_[1][0])
