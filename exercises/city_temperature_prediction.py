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
    dayOfYearT = np.zeros(len(temp),)
    for t in range(len(temp)):
        date = datetime.strptime(temp[t],
                                          "%Y-%m-%d")
        dayOfYearT[t] = date.timetuple().tm_yday
    seriesDayOfYear = pd.Series(dayOfYearT)

    df["day_Of_Year"] = seriesDayOfYear
    df["day_Of_Year"] = (df["day_Of_Year"]).astype(int)
    df = df.drop("Date", 1)



    df = df[df["Day"].isin(range(1,32)) &
             df["Month"].isin(range(1,13))]


    # return the samples and the vector of their prices
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(r"C:\Users\elena\PycharmProjects\IML.HUJI\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    dfIsrael = df.loc[(df["Country"] == "Israel") & (df["Temp"] >= -20)]


    dfIsrael["Year"] = dfIsrael["Year"].astype(str)



    # fig = px.scatter(dfIsrael, x="day_Of_Year", y="Temp", color= "Year",
    #                    title="average daily temperature change as a function of Day of year",
    #                    width=700, height=400,
    #                    template="simple_white"
    #                    )
    # fig.update_traces(marker_size=3)
    # fig.show()
    #polynom of degree 3

    #
    # month_dev = dfIsrael.groupby("Month").agg('std').reset_index()
    # month_dev["Month"] = month_dev["Month"].astype(str)
    # fig2 = px.histogram(month_dev, x="Month", y="Temp",
    #                  title="monthly temperture deviation",
    #                  width=700, height=400,
    #                     )
    #
    # fig2.show()

    # Question 3 - Exploring differences between countries
    countrys = df.drop("City", 1)
    countrys = countrys.groupby(["Month", "Country"]).agg({'Temp':['std', 'mad']}).reset_index()
    countrys.columns = ["Month", "Country", "Temp_std", "Temp_mad"]


    # fig3 = px.line(countrys, x="Month", y="Temp_mad", color="Country", error_y = sub("Temp_std","Temp_mad"),
    #                    title="average monthly temperature of countries",
    #                    width=700, height=400,
    #                    template="simple_white"
    #                    )
    # fig3.update_traces(marker_size=3)
    # fig3.show()


    # Question 4 - Fitting model for different values of `k`
    dfIsrael["Year"] = dfIsrael["Year"].astype(int)
    splitTuple = split_train_test(dfIsrael, dfIsrael["Temp"], 0.75)
    trainingSetX = splitTuple[0].to_numpy()
    trainingSety = splitTuple[1].to_numpy()
    testSetX = splitTuple[2]
    testSety = splitTuple[3]
    poly_fitting = np.array(11, )
    for k in range(1,11):
        poly_fitting[k] = PolynomialFitting(k).fit(trainingSetX, trainingSety)




    # Question 5 - Evaluating fitted model on different countries
