import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from keras.models import load_model
import streamlit as st
import pickle

start = '2009-12-31'
end = '2020-01-01'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Tickets','AAPL')
stock_data = yf.download(user_input,start,end)


#Describing Data
st.subheader('Data from 2010 - 2019')
st.write(stock_data.describe())

#Visualization
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(stock_data.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA')
ma100 = stock_data.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(stock_data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA $ 200MA')
ma100 = stock_data.Close.rolling(100).mean()
ma200 = stock_data.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(stock_data.Close, 'b')
st.pyplot(fig)


#Splitting Data into Training and Testing

data_training = pd.DataFrame(stock_data["Close"][0:int(len(stock_data)*0.70)])
data_testing = pd.DataFrame(stock_data["Close"][int(len(stock_data)*0.70):int(len(stock_data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)






#Load my model
model = load_model('keras_model.h5')

# Testing Part

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days,data_testing], ignore_index =True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor



#Final Graph

st.subheader('Predicted vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, "b", label="Original Price")
plt.plot(y_predicted, "r", label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
st.pyplot(fig2)







