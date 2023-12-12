from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="stock_price_prediction",
    version="0.0.1",
    author="Suyash Purwar",
    description="Stock Price Prediction using LSTM, GRU, and Bidirectional LSTM models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/suyashpurwar1/stock_price_prediction",
    author_email="your-email@example.com",  # Please update with your email address
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.6",
    install_requires=[
        'numpy==1.21.2',
        'pandas==1.3.3',
        'streamlit==1.18.0',
        'matplotlib==3.4.3',
        'scikit-learn==0.24.2',
        'tensorflow==2.6.0',
        'yfinance==0.1.64'
    ],
)
