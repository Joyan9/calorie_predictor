# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 19:24:49 2022

@author: joyan
"""

import numpy as np
import pickle
import streamlit as st

model = pickle.load(open('C:/Users/joyan/OneDrive/Desktop/Machine learning course/Calorie prediction/trained_model.sav','rb'))

def calorie_prediction(input_data):
    input_data_array = np.asarray(input_data)
    input_data_reshaped = input_data_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    return prediction[0]

def main():
    #giving a title to our page
    st.title("Calorie prediction")
    #getting input from user
    #gender (female:1, male:0),age, height(in cm), weight(kg), duration(mins),
    #heart_rate(bpm), body_temp(celcius)
    
    gender = st.text_input("Gender, if Female enter 1 else enter 0")
    age = st.text_input("Age")
    height = st.text_input("Height in cm")
    weight = st.text_input("Weight in kg")
    duration = st.text_input("Exercise duration in minutes")
    heart_rate = st.text_input("Average heart rate during exercise")
    body_temp = st.text_input("Average body temperature in centigrade during exercise")
    
    #code for prediction
    calories = 0
    
    
    #button for predition
    if st.button("Predict Calories"):
        calories = calorie_prediction([gender,age,height,weight,duration,heart_rate,body_temp])
        
    st.success(calories)
    
if __name__ == "__main__":
    main()
