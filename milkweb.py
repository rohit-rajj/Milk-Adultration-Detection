import numpy as np
import pandas as pd
import pickle
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

loaded_model = joblib.load("decision_model.joblib.dat")

def milkquality_prediction(input_data):
    # Normalize input_data
    input_data[0] = (float(input_data[0]) - 3.0)/(9.5 - 3.0)
    input_data[1] = (float(input_data[1]) - 34)/(90 - 34)
    input_data[-1] = (float(input_data[-1]) - 240)/(255 - 240)

    # Convert input_data to numpy array and reshape
    id_np_array = np.asarray(input_data)
    id_reshaped = id_np_array.reshape(1,-1)
    prediction = loaded_model.predict(id_reshaped)
    if(prediction[0]==False):
        return "MILK IS ADULTERATED"
    else:
        return "MILK IS NOT ADULTERATED"
                 
def main():
    
    st.title('Milk Quality Prediction')
    
    pH = st.text_input('PH Level')
    Temprature = st.text_input('Temperature')	
    Taste = st.text_input('Taste Level')
    Odor = st.text_input('Odour Level')
    Fat = st.text_input('Fat Level')
    Turbidity = st.text_input('Turbidity')	
    Colour = st.text_input('Color Level of Milk')
    
    predictionn = ''
    
    if st.button('Predict'):
        predictionn = milkquality_prediction([pH, Temprature, Taste, Odor, Fat, Turbidity, Colour
])  
    st.success(predictionn)
    
if __name__=='__main__':
    main()