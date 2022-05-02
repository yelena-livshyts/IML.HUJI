import math

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
from datetime import datetime, date
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename)


    for c in ["Month", "Day", "Year"]:
        df = df[df[c] > 0]
    temp = df["Date"].to_numpy()
    df = df[df["Day"].isin(range(1, 32)) &
            df["Month"].isin(range(1, 13))]

    df = df[df["Temp"] > 0]

    dayOfYearT = np.zeros(len(temp),)
    for t in range(len(temp)):
        date = datetime.strptime(temp[t],
                                          "%Y-%m-%d")
        dayOfYearT[t] = date.timetuple().tm_yday
    seriesDayOfYear = pd.Series(dayOfYearT)


    df["day_Of_Year"] = seriesDayOfYear
    df["day_Of_Year"] = (df["day_Of_Year"]).astype(int)
    df = df.drop("Date", axis=1)

    # return the samples and the vector of their prices
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(r"C:\Users\elena\PycharmProjects\IML.HUJI\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    dfIsrael = df.loc[(df["Country"] == "Israel")].copy()


    dfIsrael["Year"] = dfIsrael["Year"].astype(str)



    fig = px.scatter(dfIsrael, x="day_Of_Year", y="Temp", color= "Year",
                       title="average daily temperature change as a function of Day of year",
                       width=700, height=400,
                       template="simple_white"
                       )
    fig.update_traces(marker_size=3)
    fig.show()
    #polynom of degree 3

    #
    month_dev = dfIsrael.groupby("Month").agg('std').reset_index()
    month_dev["Month"] = month_dev["Month"].astype(str)
    fig2 = px.histogram(month_dev, x="Month", y="Temp",histnorm ="", nbins=12,
                     title="monthly temperture deviation",
                     width=700, height=400,
                        )

    fig2.show()

    # Question 3 - Exploring differences between countries
    countrys = df.drop("City", axis=1)
    countrys = countrys.groupby(["Month", "Country"]).agg({'Temp':['std', 'mean']}).reset_index()
    countrys.columns = ["Month", "Country", "Temp_std", "Temp_average"]


    fig3 = px.line(countrys, x="Month", y="Temp_average", color="Country", error_y = "Temp_std",
                       title="average monthly temperature of countries",
                       width=700, height=400,
                       template="simple_white"
                       )
    fig3.update_traces(marker_size=3)
    fig3.show()


    # Question 4 - Fitting model for different values of `k`
    df_israel_new = df.loc[(df["Country"] == "Israel")].copy()
    splitTuple = split_train_test(df_israel_new["day_Of_Year"], df_israel_new["Temp"], 0.75)
    trainingSetX = splitTuple[0].to_numpy()
    trainingSety = splitTuple[1].to_numpy()
    testSetX = splitTuple[2].to_numpy()
    testSety = splitTuple[3].to_numpy()
    poly_losses = np.zeros(10, )
    ks = np.zeros(10, )
    for k in range(1,11):
        poly_obj = PolynomialFitting(k)
        ks[k-1]=k
        poly_obj.fit(trainingSetX, trainingSety)
        poly_losses[k-1] = np.round(poly_obj.loss(testSetX, testSety),2)
        print("for k ", k, "the  test error recorded is ", poly_losses[k-1])
    data_cols = {'k_degree': ks, 'loss_err': poly_losses}
    data_f = pd.DataFrame(data=data_cols)
    data_f['k_degree'] = data_f['k_degree'].astype(int)

    fig4 = px.histogram(data_f, x="k_degree", y ="loss_err", histnorm ="",nbins=10,
                      title="test error recorded for values of k",
                      width=700, height=400,
                         )

    fig4.show()

    # Question 5 - Evaluating fitted model on different countries


    poly_ob = PolynomialFitting(5)
    countrys_loss = (df.copy()).drop("City", axis=1)
    countrys_loss = countrys_loss.drop("Day", axis=1)
    countrys_loss = countrys_loss.drop("Month", axis=1)
    countrys_loss = countrys_loss.drop("Year", axis=1)
    countrys_loss = countrys_loss.drop_duplicates()
    dfIsrael = countrys_loss.loc[(countrys_loss["Country"] == "Israel")].copy().drop_duplicates()
    dfJordan = countrys_loss.loc[(countrys_loss["Country"] == "Jordan")].copy().drop_duplicates()
    df_South_Africa = countrys_loss.loc[
        (countrys_loss["Country"] == "South Africa")].copy().drop_duplicates()
    df_The_Netherlands = countrys_loss.loc[
        (countrys_loss["Country"] == "The Netherlands")].copy().drop_duplicates()
    p_dIsrael_t = (dfIsrael["day_Of_Year"]).to_numpy()
    poly_ob.fit(p_dIsrael_t, (dfIsrael["Temp"]).to_numpy())

    country_list = ["Jordan", "South Africa", "The Netherlands"]
    loss_of_countrys = np.zeros(3, )
    loss_of_countrys[0] = poly_ob.loss(dfJordan["day_Of_Year"].to_numpy(),
                                        dfJordan["Temp"].to_numpy())
    loss_of_countrys[1] = poly_ob.loss(df_South_Africa["day_Of_Year"].to_numpy(),
                                        df_South_Africa["Temp"].to_numpy())
    loss_of_countrys[2] = poly_ob.loss(
        df_The_Netherlands["day_Of_Year"].to_numpy(),
        df_The_Netherlands["Temp"].to_numpy())

    data_country_losses = {'country': country_list, 'loss_0f_country': loss_of_countrys}
    data_countrys = pd.DataFrame(data=data_country_losses)
    fig5 = px.histogram(data_countrys, x="country", y ="loss_0f_country", histnorm ="",nbins=4,
                     title="loss of countrys with k=5 fitted on israel",
                     width=700, height=400,
                        )

    fig5.show()







