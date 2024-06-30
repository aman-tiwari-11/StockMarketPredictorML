import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error


st.title('Stock Trend Prediction')

stock_symbol=st.text_input('Enter the Stock Ticker','AAPL')
df = yf.download(tickers=stock_symbol,period='5y',interval='1d')

st.subheader('Data from last 5 years')
st.write(df.describe())

datas=df
datas=datas.reset_index()

hgh = df[['High']]

def plot_predictions(test,predicted):
    st.subheader(f'{stock_symbol} Stock Price Prediction')
    fig=plt.figure(figsize=(12,6))
    plt.plot(test, color='red',label=f'Real {stock_symbol} Stock Price')
    plt.plot(predicted, color='blue',label=f'Predicted {stock_symbol} Stock Price')
    plt.xlabel('Time')
    plt.ylabel(f'{stock_symbol} Stock Price')
    plt.legend()
    st.pyplot(fig)

def plot_predictions1(test,predicted):
    st.subheader(f'{stock_symbol} Stock Price Movement Prediction 30 days from now')
    fig=plt.figure(figsize=(12,6))
    plt.plot(test, color='red',label=f'Real {stock_symbol} Stock Price')
    plt.plot(predicted, color='blue',label=f'Predicted {stock_symbol} Stock Price')
    plt.xlabel('Time')
    plt.ylabel(f'{stock_symbol} Stock Price')
    plt.legend()
    st.pyplot(fig)

def return_rmse(test,predicted):
    st.subheader(f'{stock_symbol} Rmse Error :')
    rmse = math.sqrt(mean_squared_error(test, predicted))
    st.write("The root mean squared error is {}.".format(rmse))

from datetime import datetime
present_year = datetime.now()
# Extracting the year from the Timestamp object
year = present_year.year
print(year)


dataset=df
training_set = dataset[:year-1].iloc[:,1:2].values
test_set = dataset[year:].iloc[:,1:2].values

from dateutil.relativedelta import relativedelta

current_date = datetime.now()
six_months_ago = current_date - relativedelta(months=6)



st.subheader(f'{stock_symbol} stock price')
fig=plt.figure(figsize=(12,6))
dataset["High"][:six_months_ago].plot(figsize=(16,4),legend=True)
dataset["High"][six_months_ago:].plot(figsize=(16,4),legend=True)
plt.legend(['Training set (Before 6 months)','Test set (Before 6 months and beyond)'])
st.pyplot(fig)

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(df.iloc[:,1:2].values)
p=training_set_scaled.shape

X_train = []
y_train = []
for i in range(150,p[0]):
    X_train.append(training_set_scaled[i-150:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

regressor=load_model('keras_model.h5')

dataset_total=hgh
inputs = dataset_total[len(dataset_total)- 150:].values
inputs = inputs.reshape(-1,1)
inputs  = sc.transform(inputs)

X_test = np.array(inputs[0:150, 0]).reshape(1, 150, 1)
predicted_values = []

for _ in range(30):
    predicted_stock_price = regressor.predict(X_test)
   #predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    predicted_values.append(predicted_stock_price[0, 0])
    
    # Update the input sequence by shifting and replacing the oldest value
    X_test = np.roll(X_test, -1, axis=1)
    X_test[:, -1, :] = predicted_stock_price
    #print(X_test)

predicted_values = np.array(predicted_values)


predicted_values=predicted_values.reshape(-1, 1)
predicted_values=sc.inverse_transform(predicted_values)

training_set = dataset[:six_months_ago].iloc[:,1:2].values
test_set = dataset[six_months_ago:].iloc[:,1:2].values


st.subheader('Closing vs Time chart with 100MA')
fig=plt.figure(figsize=(12,6))
plt.plot(df.High)
ma100=df.High.rolling(100).mean()
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Closing vs Time chart with 100MA')
fig=plt.figure(figsize=(12,6))
plt.plot(df.High)
ma100=df.High.rolling(100).mean()
ma200=df.High.rolling(200).mean()
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

p=training_set_scaled.shape

X_train = []
y_train = []
for i in range(150,p[0]):
    X_train.append(training_set_scaled[i-150:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

dataset_total = hgh
inputs = dataset_total[len(dataset_total)-len(test_set) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = sc.transform(inputs)

q=inputs.shape

X_test = []
for i in range(60,q[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)



plot_predictions(test_set,predicted_stock_price)
return_rmse(test_set,predicted_stock_price)

concatenated_array = np.concatenate((predicted_stock_price, predicted_values), axis=0)
plot_predictions1(test_set,concatenated_array)
