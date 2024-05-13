import streamlit as st

import eda
import predict

#setting page configuration
navigation = st.sidebar.selectbox("Select Page", 
                                  options=['EDA','Predict'])

st.sidebar.write('# About')
st.sidebar.write('''
This page is created to predict customer to receive insurance or not
                 ''')
if navigation =='EDA':
     eda.run()
else:
     predict.run()