#%%
import pandas as pd
from statsmodels import graphics
import statsmodels.api as sm
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import warnings
from tqdm import tqdm_notebook as tqdm
from scipy.stats import shapiro
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model as ARCH

warnings.filterwarnings("ignore")
plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = 15, 5
plt.rcParams["lines.linewidth"] = 1
# %%

data = pd.read_csv(r"USDJPY=X.csv")
data["Date"] = data["Date"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))
data.set_index(keys="Date", inplace=True)
data.ffill(inplace=True)
data.head()
# %%


def graphics(data):

    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.suptitle("Graphical Analysis")

    axes[0].hist(data, bins=20, alpha=0.8)
    axes[0].set_title("Histogram")

    sns.boxplot(y=data["Close"], ax=axes[1], orient="vertical")
    axes[1].set_title("Boxplot")

    sm.qqplot(data["Close"], ax=axes[2], line="q")
    axes[2].set_title("Q-Q Plot against a normal distribution")

    plt.show()


# %%
# For this project, we will work with only the Close values of the USD/JPY Exchange Rate.
data = data[["Close"]]
data.plot()
plt.show()

graphics(data)
# We can see a clear downtrend in the data and maybe even a seasonal pattern. Let us look into this further
# We shall now create a histogram to find the out how the data is distributed.
# The distribution of the data seems to be skewed towards the right. Let us try to take the difference of each value

#%%
newData = data.diff()
newData = newData[1:]  # First value is NaN due to differencing
graphics(newData)
# It seems as though the new distribution has fat-tails and a lot of outliers
# Let us see what a boxplot tells us
# The Q-Q Plot shows us that the distribution of the differenced data has fat tails
# Let us perform a Shapiro test just to verify our assumption


#%%
def shapiro_normality(data, alpha=0.05):

    print("\n------Testing for Normality------ \n")
    _, pvalue = shapiro(data)

    if pvalue >= alpha:
        print("We do not Reject the Null Hypothesis")
        print("The data can be assumed to be Normally distributed")
        return True
    else:
        print("We reject the Null Hypothesis")
        print("The data can not be assumed to be Normally distributed")
        return False


def check_stationarity(data, alpha=0.05):

    print("\n------Checking for Stationarity------ \n")
    testResults = adfuller(data)
    pvalue = testResults[1]

    if pvalue >= alpha:
        print("We do not Reject the Null Hypothesis")
        print("The data can not be assumed to be Stationary")

        return True

    else:
        print("We reject the Null Hypothesis")
        print("The data can be assumed to be Stationary")

        return False


# %%
shapiro_normality(data)
check_stationarity(data)

# Even thoguh the data is not Normally distributed, it is stationary. Since Stationarity is the only requirement of ARIMA, we shall move on with the project

shapiro_normality(newData)
check_stationarity(newData)

#%%

# We want to find out if autocorrelation is present in our time series data
def check_autocorrelation(data):

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle("Autocorrelation Plots", size=20)
    plot_acf(data, ax=axes[0])
    plot_pacf(data, ax=axes[1])
    plt.show()


#%%
check_autocorrelation(data)
# As visible from the plots, there seems to be a high autocorrelation with lag order 1. Hence let us check
# for autocorrelation in our differenced data
check_autocorrelation(newData)
# Our differenced data seems to have slight autocorrelation with lag order 5
#%%

# ARIMA requires that we find out p, d, q for the model first.
def find_best_arima(data):

    """We want to find the order that fits our ARIMA model the best. We use
    Grid Search and go through each and every combination which is O(n2)
    """

    pvalues = [value for value in range(10)]
    qvalues = [value for value in range(10)]
    d = 1
    minAIC, bestOrder, bestTrend = float("inf"), None, None

    for p in tqdm(pvalues):
        for q in qvalues:
            for trend in ["c", "ct", "ctt"]:

                order = (p, d, q)
                try:
                    model = ARIMA(endog=data, order=order, trend=trend)
                    result = model.fit()
                    AIC = result.aic

                    if AIC < minAIC:
                        bestTrend = trend
                        minAIC = AIC
                        bestOrder = order

                    else:
                        continue
                except:
                    continue

    bestResult = ARIMA(endog=data, order=bestOrder, trend=bestTrend).fit()

    return bestResult, bestOrder


# %%
# We want to now split our data into Train and Testing Data
# We shall choose 80% of the data to be training data and the rest shall serve as test data

trainSize = int(len(newData) * 0.8)
trainSet = newData[:trainSize]
testSet = newData[trainSize:]

#%%

ARIMA_model, bestOrder = find_best_arima(trainSet)
ARIMA_model.summary()
# We have now found the best fit ARIMA model. We now want to use it to forecast future values
#%%

ARIMA_model = ARIMAResults.load("ARIMA_model.pkl")
# %%

forecastLength = len(testSet)
ARIMA_forecast = ARIMA_model.get_forecast(steps=forecastLength)

# The get_forecast() method returns an array of predictions, std_erros, and confidence internvals
# We want to plot the testSet and the predicted values side by side to see how well our model can forecast
# %%
predictedArray = pd.DataFrame(
    data={
        "Prediction": ARIMA_forecast.predicted_mean,
        "Confidence_Interval_1": ARIMA_forecast.conf_int(0.05)["lower Close"],
        "Confidence_Interval_2": ARIMA_forecast.conf_int(0.05)["upper Close"],
    }
)

#%%

# We inverse difference our time series so that we are able to compare it to a naive forecast
testSet = testSet + data[trainSize:].shift(1)[1:]


for idx, i in enumerate(predictedArray.index):

    predictedArray.Prediction[i] = (
        predictedArray.Prediction[i] + data[trainSize:].shift(1)[1:].iloc[idx]
    )
    predictedArray.Confidence_Interval_1[i] += data[trainSize:].shift(1).iloc[idx]
    predictedArray.Confidence_Interval_2[i] += data[trainSize:].shift(1)[1:].iloc[idx]
#%%


def plot_forecast(predictedArray, testSet, algo):

    fig, axes = plt.subplots()
    axes.fill_between(
        x=predictedArray.index,
        y1=predictedArray.Confidence_Interval_1,
        y2=predictedArray.Confidence_Interval_2,
        alpha=0.1,
        color="orange",
    )
    axes.plot(predictedArray.Prediction, color="red")
    axes.plot(testSet, color="blue")
    axes.set_title("Forecast of USD/JPY by {}".format(algo))
    axes.legend((predictedArray, testSet), ("Forecast", "True values"))

    plt.show()


# %%

# By plotting the forecast, we are able to clearly see that our model predicts the time series fairly well.
# Let us try to see how it compares against a naive forecast by its rMAPE score.

plot_forecast(predictedArray, testSet, algo="ARIMA")

MAPE = np.mean(abs(predictedArray.Prediction.values - testSet.values) / testSet.values)
naiveForecast = testSet.shift(1).bfill()
naiveMAPE = np.mean((abs(naiveForecast.values - testSet.values) / testSet.values)[1:])
rMAPE = MAPE / naiveMAPE

print("\n------------- CONGRATULATIONS !! -------------\n")
print("The ARIMA model has achieved an rMAPE of: {}".format(round(rMAPE, 4)))

#%%

# Now let us move on to trying to forecast the ARIMA residuals using a GARCH model
# We shall use the order (1,1) for the GARCH model

ARIMA_residuals = ARIMA_model.resid
GARCH_model = ARCH(ARIMA_residuals, vol="GARCH", p=1, q=1).fit(disp="off")
GARCH_model.summary()

#%%

ARIMA_result = ARIMA_model.get_forecast(steps=forecastLength)
predicted_mu = ARIMA_result.predicted_mean
# Use GARCH to predict the residual
GARCH_result = GARCH_model.forecast(horizon=forecastLength)
predicted_et = GARCH_result.mean.iloc[-1].values
# Combine both models' output: yt = mu + et

prediction = predicted_mu + predicted_et

#%%

predictedArray_ARIMA_GARCH = pd.DataFrame(
    data={
        "Prediction": prediction,
        "Confidence_Interval_1": ARIMA_result.conf_int(0.05)["lower Close"],
        "Confidence_Interval_2": ARIMA_result.conf_int(0.05)["upper Close"],
    }
)
# %%

# We inverse difference our time series so that we are able to compare it to a naive forecast
# testSet = testSet + data[trainSize:].shift(1)[1:]

for idx, i in enumerate(predictedArray_ARIMA_GARCH.index):

    predictedArray_ARIMA_GARCH.Prediction[i] = (
        predictedArray_ARIMA_GARCH.Prediction[i]
        + data[trainSize:].shift(1)[1:].iloc[idx]
    )
    predictedArray_ARIMA_GARCH.Confidence_Interval_1[i] += (
        data[trainSize:].shift(1).iloc[idx]
    )
    predictedArray_ARIMA_GARCH.Confidence_Interval_2[i] += (
        data[trainSize:].shift(1)[1:].iloc[idx]
    )

#%%

plot_forecast(predictedArray_ARIMA_GARCH, testSet, algo="ARIMA-GARCH")

MAPE = np.mean(
    abs(predictedArray_ARIMA_GARCH.Prediction.values - testSet.values) / testSet.values
)
naiveForecast = testSet.shift(1).bfill()
naiveMAPE = np.mean((abs(naiveForecast.values - testSet.values) / testSet.values)[1:])
rMAPE = MAPE / naiveMAPE

print("\n------------- CONGRATULATIONS !! -------------\n")
print("The ARIMA-GARCH model has achieved an rMAPE of: {}".format(round(rMAPE, 3)))

# There is a slight improvement in the rMAPE score of the model over the ARIMA model but unfortunately both models are unable to predict better than a naive forecast

