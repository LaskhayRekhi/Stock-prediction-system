import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model
model = load_model(r'C:\Users\laksh\Downloads\project_ml\Stock_Predictions.keras')

# Streamlit UI
st.title('Stock Market Predictor')

# Input field for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')

start = '2015-01-01'
end = datetime.today().strftime('%Y-%m-%d')  # Get today's date

try:
    data = yf.download(stock, start, end)

    if not data.empty:
        stock_name = yf.Ticker(stock).info['longName']
        st.subheader(f'Stock Name: {stock_name}')

        st.subheader('Stock Data')
        st.write(data)

        # Plot moving averages
        st.subheader('Price vs Moving Averages')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['Close'], label='Close Price')
        ax.plot(data['Close'].rolling(50).mean(), label='MA50')
        ax.plot(data['Close'].rolling(100).mean(), label='MA100')
        ax.plot(data['Close'].rolling(200).mean(), label='MA200')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)

        # Prepare test data
        data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
        data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])
        scaler = MinMaxScaler(feature_range=(0,1))
        past_100_days = data_train.tail(100)
        data_test = pd.concat([past_100_days, data_test], ignore_index=True)
        data_test_scale = scaler.fit_transform(data_test)

        x = []
        y = []
        for i in range(100, data_test_scale.shape[0]):
            x.append(data_test_scale[i-100:i])
            y.append(data_test_scale[i,0])
        x, y = np.array(x), np.array(y)

        # Make predictions
        predictions = model.predict(x)
        predictions = predictions * (1/scaler.scale_)
        y = y * (1/scaler.scale_)

        # Display predictions
        st.subheader('Original Price vs Predicted Price')
        prediction_df = pd.DataFrame({'Original Price': y.flatten(), 'Predicted Price': predictions.flatten()})
        st.write(prediction_df)

        # Plot predictions vs actual
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(predictions, 'r', label='Predicted Price')
        ax.plot(y, 'g', label='Original Price')
        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)


    else:
        st.write('No data available for the given stock symbol.')

except Exception as e:
    st.error(f'Error fetching data: {str(e)}')
