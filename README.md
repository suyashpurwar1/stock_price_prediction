📈 Stock Price Prediction with Streamlit 🚀

Project Description

Stock Price Prediction is an interactive application built with Streamlit, allowing users to explore historical stock prices and predict future values using state-of-the-art machine learning models (LSTM, GRU, Bidirectional LSTM). The app is designed to provide a user-friendly interface for stock analysis and forecasting.

Features 🌟

Interactive User Interface:
Users can input a stock ticker, select start and end dates, and choose the column (Close, Open, High, Low, Adj Close) for analysis.

Data Visualization:
Historical stock prices are displayed along with 50-day, 100-day, and 200-day Moving Averages.

Model Predictions:
Utilizes pre-trained machine learning models (LSTM, GRU, Bidirectional LSTM) to predict future stock prices.

Evaluation Metrics:
Provides evaluation metrics (R² Score, Mean Absolute Error, Mean Squared Error) for the machine learning models.

Dynamic Charting:
Users can visualize actual vs predicted stock prices for different timeframes and select the number of days for future predictions.


Setup and Usage 🛠️
Installation:
Clone the repository: git clone https://github.com/suyashpurwar1/stock_price_prediction.git
Install dependencies: pip install -r requirements.txt
Run the Streamlit App:
Execute the Streamlit app: streamlit run stock_app.py

Open your browser and navigate to http://localhost:8501 to interact with the application.


Dependencies 📦:
-Streamlit
-NumPy
-Pandas
-Matplotlib
-Scikit-learn
-TensorFlow
-yfinance


Project Structure 🏗️
stock_app.py: Main Streamlit app script.
LSTM_unit_64.h5, GRU_unit_64.h5, Bi_Directional_LSTM_unit_64.h5: Pre-trained machine learning models.
requirements.txt: List of project dependencies.

Acknowledgments 🙌
Data source: Yahoo Finance
Machine Learning models: LSTM, GRU, Bidirectional LSTM

Contributing 🤝
Contributions are welcome! Feel free to open issues or submit pull requests.
