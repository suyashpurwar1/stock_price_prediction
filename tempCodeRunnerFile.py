import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import datetime
from sklearn.preprocessing import MinMaxScaler
st.title('Stock Trend Prediction')

user_input = st.sidebar.text_input('Enter Stock Ticker', 'PNB.NS')

date_end = st.sidebar.date_input("End Date", value=datetime.date(2023, 11, 17))
date_start = st.sidebar.date_input("Start Date", value=datetime.date(2010, 1, 1))
ticker = yf.Ticker(user_input)

# Allow the user to select the column for prediction
selected_column = st.sidebar.selectbox('Select a column for prediction', ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume', '% Change'])

df = yf.download(user_input, start=date_start, end=date_end)

st.header(f' Price Movements from "{date_start}" & "{date_end}"')
df['% Change'] = (df['Adj Close'] / df['Adj Close'].shift(1) - 1) * 100
df.dropna(inplace=True)
st.write(df)

# Visualization
st.subheader(f'{selected_column} vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df[selected_column])
st.pyplot(fig)

st.subheader(f'{selected_column} vs Time Chart Moving Averages MA50 and MA100')
ma50 = df[selected_column].rolling(50).mean()
ma100 = df[selected_column].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma50, 'y', label='MA50', linestyle='dashed')
plt.plot(ma100, 'r', label='MA100', linestyle='dashed')
plt.plot(df[selected_column])
st.pyplot(fig)

st.subheader(f'{selected_column} vs Time Chart with Moving Averages MA100 and MA200')
ma100 = df[selected_column].rolling(100).mean()
ma200 = df[selected_column].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r', label='MA100', linestyle='dashed')
plt.plot(ma200, 'g', label='MA200', linestyle='dashed')
plt.plot(df[selected_column])
plt.legend()
st.pyplot(fig)

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7):])

from sklearn.preprocessing import MinMaxScaler

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(np.array(data_training).reshape(-1, 1))

step_size = 100

x_train, y_train = [], []

for i in range(step_size, data_training_array.shape[0]):
    x_train.append(data_training_array[i - step_size:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape input data for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# Load your LSTM model
model_path = 'LSTM_BTP.h5'
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading the model from {model_path}: {e}")
    st.stop()
# Prepare testing data
data = df['Close']
inputs = data[len(data) - len(data_testing) - step_size:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
x_test, y_test = [], []

for i in range(step_size, len(inputs)):
    x_test.append(inputs[i - step_size:i])
    y_test.append(inputs[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Reshape input data for LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
y_predicted = model.predict(x_test)
y_predicted_rescaled = scaler.inverse_transform(y_predicted.reshape(-1, 1))
y_test_rescaled = data_testing.values.reshape(-1, 1)
st.write("Predictions from  Streamlit app:")
st.write(y_predicted_rescaled)
# Final graph
st.subheader('Prediction vs Original Using LSTM')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Actual Stock Price')
plt.plot(y_predicted_rescaled, label='Predicted Stock Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Evaluate the model
r2 = r2_score(y_test_rescaled, y_predicted_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_predicted_rescaled)
mse = mean_squared_error(y_test_rescaled, y_predicted_rescaled)

st.subheader('Model Evaluation Metrics:')
st.write(f'R^2 Score: {r2}')
st.write(f'Mean Absolute Error (MAE): {mae}')
st.write(f'Mean Squared Error (MSE): {mse}')



