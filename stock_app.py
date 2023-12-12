import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from keras.models import load_model
import datetime
import yfinance as yf

st.title('Stock Price Prediction')

# User input for stock data
user_input = st.sidebar.text_input('Enter Stock Ticker', 'PNB.NS')
start_date = st.sidebar.date_input("Select Start Date", datetime.date(2013, 1, 1))
end_date = st.sidebar.date_input("Select End Date", datetime.date(2023, 11, 17))


# Downloading stock data
df = yf.download(user_input, start=start_date, end=end_date)

selected_column = st.sidebar.selectbox('Select a column for prediction', ['Close', 'Open', 'High', 'Low', 'Adj Close'])

st.subheader(f' Price Movements of {user_input} from " {start_date} "   &  " {end_date} "')
df['% Change']=(df['Adj Close']/df['Adj Close'].shift(1)-1)*100
df.dropna(inplace=True)
st.write(df)

# Plotting the stock data with Moving Averages
ma50 = df[selected_column].rolling(50).mean()
ma100 = df[selected_column].rolling(100).mean()
ma200 = df[selected_column].rolling(200).mean()

st.subheader(f'{selected_column} vs Time Chart with 50MA & 100MA')
fig_ma = plt.figure(figsize=(12, 6))
plt.plot(df[selected_column], label=f'{selected_column} Price')
plt.plot(ma50, label='MA50', linestyle='dashed')
plt.plot(ma100, label='MA100', linestyle='dashed')
plt.legend()
st.pyplot(fig_ma)
st.subheader(f'{selected_column} vs Time Chart with 100MA & 200MA')
fig_ma2 = plt.figure(figsize=(12, 6))
plt.plot(df[selected_column], label=f'{selected_column} Price')
plt.plot(ma100, label='MA100', linestyle='dashed')
plt.plot(ma200, label='MA200', linestyle='dashed')
plt.legend()
st.pyplot(fig_ma2)

# Splitting data into training and testing
data_training = pd.DataFrame(df[selected_column][0:int(len(df) * 0.7)])
data_testing = pd.DataFrame(df[selected_column][int(len(df) * 0.7):])

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

# Load the pre-trained LSTM model
model = load_model('LSTM_unit_64.h5')

# Prepare testing data
data = df[selected_column]
inputs = data[len(data) - len(data_testing) - step_size:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.fit_transform(inputs)

x_test, y_test = [], []

for i in range(step_size, len(inputs)):
    x_test.append(inputs[i - step_size:i])
    y_test.append(inputs[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
# Reshape input data for LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# Training Data vs Time Chart
st.subheader(f'{selected_column} vs Time Chart (Training Data & Testing Data)')
fig5 = plt.figure(figsize=(12, 6))
plt.plot(data_training.index, data_training[selected_column], 'b', label='Training Data')
plt.plot(data_testing.index, data_testing[selected_column], 'r', label='Testing Data')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig5)

# Make predictions
y_predicted = model.predict(x_test)
y_predicted_rescaled = scaler.inverse_transform(y_predicted.reshape(-1, 1))
y_test_rescaled = data_testing.values.reshape(-1, 1)

# Plotting results
st.subheader('Actual vs Predicted Stock Prices (LSTM Model)')
fig_results = plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):],y_test_rescaled, label='Actual Stock Price')
plt.plot(df.index[-len(y_predicted):],y_predicted_rescaled, label='Predicted Stock Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig_results)

# Evaluate the model
r2 = r2_score(y_test_rescaled, y_predicted_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_predicted_rescaled)
mse = mean_squared_error(y_test_rescaled, y_predicted_rescaled)

st.subheader('Model Evaluation Metrics(LSTM Model):')
st.write(f'R^2 Score: {r2}')
st.write(f'Mean Absolute Error (MAE): {mae}')
st.write(f'Mean Squared Error (MSE): {mse}')

# Input for the number of days to predict
num_days_to_predict = st.number_input('Enter the number of days to predict(LSTM Model):', min_value=5, value=10, step=1)

# Make predictions for the next 'num_days_to_predict' days
predicted_stock_prices = []

for _ in range(num_days_to_predict):
    # Make prediction
    predicted_scaled = model.predict(x_test[-1].reshape(1, step_size, 1))
    
    # Inverse transform to get the predicted stock price
    predicted_stock_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))
    
    # Append the predicted stock price to the list
    predicted_stock_prices.append(predicted_stock_price[0, 0])
    
    # Update x_test for the next prediction
    x_test = np.roll(x_test, -1, axis=0)  # Shift the existing data to make room for the new prediction
    x_test[-1, :, 0] = predicted_scaled[0, 0]  # Add the new prediction to x_test

# Create an index for counting
predicted_index = np.arange(1, 1+ num_days_to_predict)

# Plotting results
st.subheader('Predicted Stock Prices for the Next Days(LSTM Model)')
fig_predictions = plt.figure(figsize=(12, 6))
plt.plot(predicted_index, predicted_stock_prices, label='Predicted Stock Prices', marker='o')
plt.title('Predicted Stock Prices for the Next Days')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig_predictions)

# Load the pre-trained GRU model
model_gru = load_model('GRU_unit_64.h5')

# Prepare testing data for GRU model
data_gru = df[selected_column]
inputs_gru = data_gru[len(data_gru) - len(data_testing) - step_size:].values
inputs_gru = inputs_gru.reshape(-1, 1)
inputs_gru = scaler.fit_transform(inputs_gru)

x_test_gru, y_test_gru = [], []

for i in range(step_size, len(inputs_gru)):
    x_test_gru.append(inputs_gru[i - step_size:i])
    y_test_gru.append(inputs_gru[i, 0])

x_test_gru, y_test_gru = np.array(x_test_gru), np.array(y_test_gru)

# Reshape input data for GRU
x_test_gru = np.reshape(x_test_gru, (x_test_gru.shape[0], x_test_gru.shape[1], 1))

# Make predictions using the GRU model
y_predicted_gru = model_gru.predict(x_test_gru)
y_predicted_rescaled_gru = scaler.inverse_transform(y_predicted_gru.reshape(-1, 1))

# Plotting results for the GRU model
st.subheader('Actual vs Predicted Stock Prices (GRU Model)')
fig_results_gru = plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):],y_test_rescaled, label='Actual Stock Price')
plt.plot(df.index[-len(y_test):],y_predicted_rescaled_gru, label='Predicted Stock Price (GRU Model)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig_results_gru)

# Evaluate the GRU model
r2_gru = r2_score(y_test_rescaled, y_predicted_rescaled_gru)
mae_gru = mean_absolute_error(y_test_rescaled, y_predicted_rescaled_gru)
mse_gru = mean_squared_error(y_test_rescaled, y_predicted_rescaled_gru)

st.subheader('Model Evaluation Metrics (GRU Model):')
st.write(f'R^2 Score: {r2_gru}')
st.write(f'Mean Absolute Error (MAE): {mae_gru}')
st.write(f'Mean Squared Error (MSE): {mse_gru}')

# Input for the number of days to predict using the GRU model
num_days_to_predict_gru = st.number_input('Enter the number of days to predict (GRU Model):', min_value=1, value=10, step=1)

# Make predictions for the next 'num_days_to_predict_gru' days using the GRU model
predicted_stock_prices_gru = []

for _ in range(num_days_to_predict_gru):
    # Make prediction using the GRU model
    predicted_scaled_gru = model_gru.predict(x_test_gru[-1].reshape(1, step_size, 1))
    
    # Inverse transform to get the predicted stock price
    predicted_stock_price_gru = scaler.inverse_transform(predicted_scaled_gru.reshape(-1, 1))
    
    # Append the predicted stock price to the list
    predicted_stock_prices_gru.append(predicted_stock_price_gru[0, 0])
    
    # Update x_test_gru for the next prediction
    x_test_gru = np.roll(x_test_gru, -1, axis=0)  # Shift the existing data to make room for the new prediction
    x_test_gru[-1, :, 0] = predicted_scaled_gru[0, 0]  # Add the new prediction to x_test_gru

# Create an index for counting
predicted_index_gru = np.arange(1, 1+ num_days_to_predict_gru)

# Plotting results for the GRU model
st.subheader('Predicted Stock Prices for the Next Days (GRU Model)')
fig_predictions_gru = plt.figure(figsize=(12, 6))
plt.plot(predicted_index_gru, predicted_stock_prices_gru, label='Predicted Stock Prices (GRU Model)', marker='o')
plt.title('Predicted Stock Prices for the Next Days (GRU Model)')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig_predictions_gru)

# Display predicted stock prices for up to 5 days in a table
st.subheader('Predicted Stock Prices for Up to 5 Days')
predicted_table_data = {'Day': np.arange(1, min(6, num_days_to_predict + 1)),
                        'LSTM Model Prediction': predicted_stock_prices[:5],
                        'GRU Model Prediction': predicted_stock_prices_gru[:5]}
predicted_table_df = pd.DataFrame(predicted_table_data)
st.table(predicted_table_df)

# Load the pre-trained bidirectional LSTM model
model_bi_lstm = load_model('Bi_Directional_LSTM_unit_64.h5')

# Prepare testing data for bidirectional LSTM model
data_bi_lstm = df[selected_column]
inputs_bi_lstm = data_bi_lstm[len(data_bi_lstm) - len(data_testing) - step_size:].values
inputs_bi_lstm = inputs_bi_lstm.reshape(-1, 1)
inputs_bi_lstm = scaler.fit_transform(inputs_bi_lstm)

x_test_bi_lstm, y_test_bi_lstm = [], []

for i in range(step_size, len(inputs_bi_lstm)):
    x_test_bi_lstm.append(inputs_bi_lstm[i - step_size:i])
    y_test_bi_lstm.append(inputs_bi_lstm[i, 0])

x_test_bi_lstm, y_test_bi_lstm = np.array(x_test_bi_lstm), np.array(y_test_bi_lstm)

# Reshape input data for bidirectional LSTM
x_test_bi_lstm = np.reshape(x_test_bi_lstm, (x_test_bi_lstm.shape[0], x_test_bi_lstm.shape[1], 1))

# Make predictions using the bidirectional LSTM model
y_predicted_bi_lstm = model_bi_lstm.predict(x_test_bi_lstm)
y_predicted_rescaled_bi_lstm = scaler.inverse_transform(y_predicted_bi_lstm.reshape(-1, 1))

# Plotting results for the bidirectional LSTM model
st.subheader('Actual vs Predicted Stock Prices (Bidirectional LSTM Model)')
fig_results_bi_lstm = plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):], y_test_rescaled, label='Actual Stock Price')
plt.plot(df.index[-len(y_test):], y_predicted_rescaled_bi_lstm, label='Predicted Stock Price (Bidirectional LSTM Model)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig_results_bi_lstm)

# Evaluate the bidirectional LSTM model
r2_bi_lstm = r2_score(y_test_rescaled, y_predicted_rescaled_bi_lstm)
mae_bi_lstm = mean_absolute_error(y_test_rescaled, y_predicted_rescaled_bi_lstm)
mse_bi_lstm = mean_squared_error(y_test_rescaled, y_predicted_rescaled_bi_lstm)

st.subheader('Model Evaluation Metrics (Bidirectional LSTM Model):')
st.write(f'R^2 Score: {r2_bi_lstm}')
st.write(f'Mean Absolute Error (MAE): {mae_bi_lstm}')
st.write(f'Mean Squared Error (MSE): {mse_bi_lstm}')

# Input for the number of days to predict using the bidirectional LSTM model
num_days_to_predict_bi_lstm = st.number_input('Enter the number of days to predict (Bidirectional LSTM Model):', min_value=1, value=10, step=1)

# Make predictions for the next 'num_days_to_predict_bi_lstm' days using the bidirectional LSTM model
predicted_stock_prices_bi_lstm = []

for _ in range(num_days_to_predict_bi_lstm):
    # Make prediction using the bidirectional LSTM model
    predicted_scaled_bi_lstm = model_bi_lstm.predict(x_test_bi_lstm[-1].reshape(1, step_size, 1))

    # Inverse transform to get the predicted stock price
    predicted_stock_price_bi_lstm = scaler.inverse_transform(predicted_scaled_bi_lstm.reshape(-1, 1))

    # Append the predicted stock price to the list
    predicted_stock_prices_bi_lstm.append(predicted_stock_price_bi_lstm[0, 0])

    # Update x_test_bi_lstm for the next prediction
    x_test_bi_lstm = np.roll(x_test_bi_lstm, -1, axis=0)  # Shift the existing data to make room for the new prediction
    x_test_bi_lstm[-1, :, 0] = predicted_scaled_bi_lstm[0, 0]  # Add the new prediction to x_test_bi_lstm

# Create an index for counting
predicted_index_bi_lstm = np.arange(1, 1 + num_days_to_predict_bi_lstm)

# Plotting results for the bidirectional LSTM model
st.subheader('Predicted Stock Prices for the Next Days (Bidirectional LSTM Model)')
fig_predictions_bi_lstm = plt.figure(figsize=(12, 6))
plt.plot(predicted_index_bi_lstm, predicted_stock_prices_bi_lstm, label='Predicted Stock Prices (Bidirectional LSTM Model)', marker='o')
plt.title('Predicted Stock Prices for the Next Days (Bidirectional LSTM Model)')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig_predictions_bi_lstm)

# Display predicted stock prices for up to 5 days in a table
st.subheader('Predicted Stock Prices for Up to 5 Days')
predicted_table_data_bi_lstm = {'Day': np.arange(1, min(6, num_days_to_predict_bi_lstm + 1)),
                                'LSTM Model Prediction': predicted_stock_prices[:5],
                                'Bidirectional LSTM Model Prediction': predicted_stock_prices_bi_lstm[:5]
                                ,'GRU Model Prediction': predicted_stock_prices_gru[:5]}
predicted_table_df_bi_lstm = pd.DataFrame(predicted_table_data_bi_lstm)
st.table(predicted_table_df_bi_lstm)
