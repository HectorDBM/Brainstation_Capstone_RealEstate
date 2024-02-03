#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# load the training model
nation_model = joblib.load('nation_xgb_model.joblib')


def main():
    # Streamlit UI
    st.title('US Real Estate Forcast Model, nation level')
    
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
    
    # Prediction
    if st.button('Predict'):
        make_prediction = nation_model.predict([[Unit_Type, Int_rate, GDP, Unemp_Rate, month]])
        output = round(make_prediction[0],1)
        st.success('You can expect to buy/sell your property for ${} USD'.format(output))
        
if __name__=='__main__':
    main()
    

