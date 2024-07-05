'''Chinaza Aham-Neze
Stock Market Predictor with Python'''

import yfinance as yf
import matplotlib.pyplot as plt
sp500 = yf.Ticker("^GSPC")

#queries all data from when the index was created 
sp500 = sp500.history(period="max")

'''When I run the sp500 in the terminal, pandas dataframe. I see the prices from a
a specifc trading day. I am using the opening price, high price, low price, and
closing price along with the volume to make a descision'''
                                             
# Data Cleaning and Visualization
sp500.plot.line(y= "Close", use_index=True)

'''
plt.title('S&P 500 Closing Prices')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.show()
'''

'''Remove the uneeded dividends column'''
del sp500["Dividends"]
del sp500["Stock Splits"]

# Setting up a Target to figure out if the price will go up or down
'''Shift the close column and shift the prices back one day'''
sp500["Tomorrow"] = sp500["Close"].shift(-1)

sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
'''The result of sp500 should show a '1' if the price went up and a '0' if price
went down'''

sp500 = sp500.loc["1990-01-01":].copy()

#Training initial learning model
'''Since RandomForestClassifier is multiple decison trees, there is a lower chance
of overfit. It can also pick up non linear relationships'''
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100, min_samples_split=100, random_state =1)

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

# Measuring model accuracy
from sklearn.metrics import precision_score

preds = model.predict(test[predictors])

import  pandas as pd
preds = pd.Series(preds, index= test.index)

precision_score(test["Target"], preds)

'''Need to make a more accurate prediction. Combine actual values and predicted values'''
combined= pd.concat([test["Target"], preds], axis =1)
combined.plot()
plt.show()


# Building a BackTesting System
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index= test.index, name = "Predictions")
    combined= pd.concat([test["Target"], preds], axis =1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    '''
We are taking the data, model, predictors, a start, and step argument. The start represented
by 2500 is the 2500 training days so 10 years of data. The step is the 250 days in a training year


    Backtests a model on historical data.

    Parameters:
    - data: DataFrame containing historical data
    - model: Machine learning model (already trained)
    - predictors: List of predictor columns
    - start: Starting index for training data (default: 2500)
    - step: Size of each training step (default: 250)

    Returns:
    - DataFrame containing predictions for each test period
    '''

    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)  # Assuming predict function is defined elsewhere
        all_predictions.append(predictions)

    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)
predictions["Predictions"].value_counts()
 
precision_score(predictions["Target"], predictions["Predictions"])

'''Need to check if the prediction score up 50% of the time is good or not'''
predictions["Target"].value_counts() / predictions.shape[0]


# Adding additional predictors to our model
'''These horizons represent 2 days, 1 Trading week(5 days), 3 months(60 days), 1 year(250 days),
and 4 years( 1000 days)
'''
horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    ''' This loop will look at the past 2 days and see the sum of the Target in our sp500 data'''
    rolling_avg = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_avg["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ratio_column, trend_column]
    
sp500 = sp500.dropna()


# Improving Our Model
model = RandomForestClassifier(n_estimators =200, min_samples_split = 50, random_state =1)

def predict_new(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index= test.index, name = "Predictions")
    combined= pd.concat([test["Target"], preds], axis =1)
    return combined

predictions = backtest(sp500, model, new_predictors)

predictions["Predictions"]. value_counts()

precision_score(predictions["Target"], predictions["Predictions"])
