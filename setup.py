from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

PROJECT_NAME = "stock_price_prediction"
AUTHOR_NAME = "suyashpurwar1"
SRC_DIR = "src"
LIST_OF_REQUIREMENTS = [
    'streamlit',
    'numpy',
    'pandas',
    'matplotlib',
    'scikit-learn',
    'yfinance',
    'tensorflow'
]

setup(
    name=PROJECT_NAME,
    version="0.0.1",
    author=AUTHOR_NAME,
    description="Stock Price Prediction using Machine Learning",
    long_description="Stock price predictor ",
    long_description_content_type="text/markdown",
    url=f"https://github.com/suyashpurwar1/stock_price_prediction",
    author_email="your-email@example.com",
    packages=find_packages(include=[f"{SRC_DIR}"]),
    license="MIT",
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)
