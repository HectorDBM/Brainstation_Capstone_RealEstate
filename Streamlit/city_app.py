#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# load the training model
city_model = joblib.load('city_xgb_model.joblib')


def main():
    # Streamlit UI
    st.title('US Real Estate Forcast Model, city level')
    
    # Input variables
    Unit_Type = st.number_input('Enter Unit Type Encoded Number: ', min_value=1, max_value=6, step=1)
    st.text('Note, the unit type is encoded with the following numbers:  1-Bed= 1, 2-Bed= 2, 3-Bed= 3,4-Bed= 4, 5-Bed= 5, Condo/Co-op= 6')
    Int_rate = st.number_input('Interest Rate in Decimal Form:', min_value=0.00, max_value=1.00, step=0.01)
    st.text('Interest rate ranges from 0.00 to 1.00')
    GDP = st.number_input('GDP [Billion of USD]: ')
    Unemp_Rate = st.number_input('Unemployment Rate in Decimal From: ', min_value=0.00, max_value=1.00, step=0.01)
    st.text('Unemployment rate ranges from 0.00 to 1.00')
    month = st.number_input('Month Number: ', min_value=1, max_value=12, step=1)
    st.text('Note: January= 1, February= 2, ... December =12')
    loc_encoded = st.number_input('Enter Location Encoded Number: ', min_value=0, max_value=31, step=1)
    st.markdown(""" Location encoded Number: Atlanta, GA = 0 || Austin, TX = 1  || Baltimore, MD = 2 || Boston, MA = 3 ||
                Charlotte, NC = 4 || Chicago, IL = 5 || Cincinnati, OH = 6 || Columbus, OH = 7 || Dallas, TX = 8 || Denver, CO = 9 || 
                Detroit, MI = 10 || Houston, TX = 11 || Kansas City, MO = 12 || Las Vegas, NV = 13 || Los Angeles, CA = 14 || 
                Miami, FL = 15 || Minneapolis, MN = 16 || New York, NY = 17 || Orlando, FL = 18 || Philadelphia, PA = 19 || 
                Phoenix, AZ = 20 || Pittsburgh, PA = 21 || Portland, = OR	22 || Riverside, CA = 23 || Sacramento, CA = 24 || 
                San Antonio, TX = 25 || San Diego, CA = 26 || San Francisco, CA = 27 || Seattle, WA = 28 || St. Louis, MO = 29 ||
                Tampa, FL = 30 || Washington, DC = 31
                """)
    
    # Prediction
    if st.button('Predict'):
        make_prediction = city_model.predict([[Unit_Type, Int_rate, GDP, Unemp_Rate, month,loc_encoded]])
        output = round(make_prediction[0],1)
        st.success('You can expect to buy/sell your property for ${} USD'.format(output))
        
if __name__=='__main__':
    main()
    

