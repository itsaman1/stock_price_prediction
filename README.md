 # Stock Price Prediction Using LSTM

## Overview
This project implements a stock price prediction application using Long Short-Term Memory (LSTM) neural networks. The goal is to predict future stock prices based on historical data, allowing users to make informed investment decisions.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Data Visualization](#data-visualization)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Features
- Fetch historical stock price data using Yahoo Finance.
- Visualize stock prices with moving averages.
- Predict future stock prices using an LSTM model.
- Interactive web interface built with Streamlit.
- User input for any stock ticker symbol.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Keras
- TensorFlow
- Streamlit
- Yahoo Finance (yfinance)

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
2. Install the required libraries
   ```bash
   pip install -r requirements.txt

## Usage
To run the application, use the following command:
   ```bash
   streamlit run app.py
```

## Data Visualization
The application provides various visualizations, including:

1. Closing price vs. time chart.
2. Closing price with 100-day and 200-day moving averages.
3. Prediction vs. original price chart.

## Model Training
The LSTM model is trained on 70% of the historical data and tested on the remaining 30%. The model architecture consists of multiple LSTM layers and dropout layers to prevent overfitting.

The model is saved as keras_model.h5, which can be loaded for making predictions.

## Contributing
Contributions are welcome! If you have suggestions for improvements or features, feel free to create a pull request or open an issue.

## Acknowledgments
1. Thanks to the authors of the libraries used in this project.
2. Inspired by various online resources on stock price prediction and LSTM models.

### Instructions for Usage
- **Replace `yourusername`**: Ensure you replace `yourusername` with your actual GitHub username in the clone URL.
- **Add a LICENSE file**: Include a LICENSE file in your repository to specify the licensing details for your project.
- **Create a `requirements.txt` file**: This file should list all the dependencies used in your project, allowing users to install them easily.

