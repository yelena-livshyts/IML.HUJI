import math
from os import path

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from pandas import Series

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()
    for c in ["id", "date", "zipcode", "lat", "long"]:
        df = df.drop(c, axis=1)
    #df["zipcode"] = df["zipcode"].astype(int)

    for c in ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built", "sqft_living15", "sqft_lot15"]:
        df = df[df[c] > 0]
    for c in ["bathrooms", "floors", "sqft_basement", "yr_renovated"]:
        df = df[df[c] >= 0]

    df["recently_renovated"] = np.where(
        df["yr_renovated"] >= np.percentile(df.yr_renovated.unique(), 60), 1, 0)
    df = df.drop("yr_renovated", axis=1)
    df["decade_built"] = (df["yr_built"] / 100).astype(int)
    df = df.drop("yr_built", axis=1)


    df = df[df["waterfront"].isin([0, 1]) &
            df["view"].isin(range(5)) &
            df["condition"].isin(range(1, 6)) &
            df["grade"].isin(range(1, 15))]


#return the samples and the vector of their prices
    return df.drop("price", axis=1), df.price



def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved

    """

    for f in X:
        rho = np.cov(X[f], y)[0, 1] / (np.std(X[f]) * np.std(y))

        fig = px.scatter(pd.DataFrame({'x': X[f], 'y': y}), x="x", y="y",
                         trendline="ols",
                         title=f"Correlation Between {f} Values and Response <br>Pearson Correlation {rho}",
                         labels={"x": f"{f} Values", "y": "Response Values"})
        pio.write_image(fig, path.join(output_path,"Pearson Correlation of %s.png" % f))



if __name__ == '__main__':
    df = pd.read_csv(r"C:\Users\elena\PycharmProjects\IML.HUJI\datasets\house_prices.csv")
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(r"C:\Users\elena\PycharmProjects\IML.HUJI\datasets\house_prices.csv")


    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, )

    # Question 3 - Split samples into training- and testing sets.
    splitTuple = split_train_test(X, y, 0.75)


    # Question 4 - Fit model over increasing percentages of the overall training data
    trainingSetX = splitTuple[0]
    trainingSety = splitTuple[1]
    testSetX = splitTuple[2].to_numpy()
    testSety = splitTuple[3].to_numpy()
    average_loss = np.zeros(91, )
    percent = np.zeros(91, )
    variances = np.zeros(91, )

    for i in range(10,101):
        av_mse=0
        losses = np.zeros(10, )
        for j in range(10):
            split_by_i = split_train_test(trainingSetX, trainingSety, i/100)
            pSample = split_by_i[0].to_numpy()
            responces= split_by_i[1].to_numpy()
            lr =LinearRegression()
            lr._fit(pSample, responces)
            mse = lr._loss(testSetX, testSety)
            losses[j] = mse
            av_mse=av_mse+mse
        percent[i-10]=i
        av_mse= av_mse/10
        average_loss[i-10]= av_mse
        variances[i-10] = math.sqrt(np.var(losses))*2

    fig = go.Figure([
        go.Scatter(
            name='average loss',
            x=percent,
            y=average_loss,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=percent,
            y=average_loss + variances,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),

        go.Scatter(
            name='Lower Bound',
            x=percent,
            y=average_loss - variances,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(
        xaxis_title = '%p',
        yaxis_title='average loss',
        title='mean loss as function of %p with confidence interval',
        hovermode="x"
    )
    fig.show()




    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
