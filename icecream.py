import streamlit as st
import pickle
import pandas as pd

# connection to mongo db
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://zahid:zahid1234@cluster0.sltur.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# create database 
db = client['IceCream']

# create collection
collection = db['icecream_pred']

# load pickle file
def load_model():
    with open('ice-cream.pkl', 'rb') as file:
        model = pickle.load(file)
        return model

# processing for model
def processing_input_data(data):
    df = pd.DataFrame([data])
    return df

# prediction of model
def predict_data(data):
    model = load_model()
    processed_data = processing_input_data(data)
    return model.predict(processed_data)

def main():
    st.title('ICE-CREAM Prediction')
    temp = st.slider('Temperature (°C)', -4.662263, 4.899032, 0.0)
        
    user_data = {
        'Temperature (°C)' : temp,
    }
    
    if st.button('Predict'):
        predict = predict_data(user_data)
        predicted_val = round(float(predict[0]), 3)
        
        st.success(f"Sales Prediction : {predicted_val}")
        
        # add to the collection
        user_data['Predicted Sales'] = predicted_val
        collection.insert_one(user_data)
        
if __name__ == '__main__':
    main()