# 📈 Stock Price Prediction App

This project is a web-based stock price prediction app built using **Streamlit**, **TensorFlow/Keras**, and **scikit-learn**. It uses historical stock data to predict the next day's closing price using a trained machine learning model (LSTM-based).

## 🚀 Features

- Predict stock closing prices based on historical data  
- Clean & simple Streamlit web interface  
- Uses trained LSTM model (`keras`) for accurate forecasting  
- Data preprocessing using `MinMaxScaler`  
- Easy to deploy and use locally

## 🛠️ Tech Stack

- Python 3.11
- Streamlit
- Pandas
- Numpy
- scikit-learn
- TensorFlow / Keras
- Matplotlib

## 📸 Screenshots

*(Add screenshots of your app here if possible)*

## 🧠 Model Info

The model used is a Long Short-Term Memory (LSTM) neural network, trained on historical stock prices using past `n` days to predict the next day.

## 📂 Project Structure

stockprediction/
│
├── app.py # Streamlit frontend
├── model.h5 # Trained LSTM model
├── scaler.pkl # Saved MinMaxScaler for inverse transform
├── data/ # (Optional) Folder for stock data
└── requirements.txt # Required Python packages

markdown
Copy
Edit

## ▶️ How to Run

Make sure you have Python 3.11 and pip installed.

1. **Install dependencies**  
```bash
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py
📦 Installation Notes
Make sure you have TensorFlow installed:

bash
Copy
Edit
pip install tensorflow
If any module gives error, install manually using:

bash
Copy
Edit
pip install streamlit pandas numpy scikit-learn matplotlib
