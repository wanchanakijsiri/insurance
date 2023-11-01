
import streamlit as st
import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
import pickle

#Load model
with open('model_Prediction.pkl', 'rb') as file:
    # Load the data from the file
    model, sex_encoder, smoker_encoder, region_encoder = pickle.load(file)
    
# Streamlit app
st.title("Insurance Expense Prediction")

# Get user input for each variable
age = st.slider('Please enter age:', 0, 100, 25)
sex = st.selectbox('PLease select sex:', ['male', 'female'])
bmi = st.slider('Please enter bmi:', 0, 100, 25)
children = st.slider('Please enter children:', 0, 5, 0)
smoker = st.selectbox('PLease select smoker:', ['yes', 'no'])
region = st.selectbox('PLease select region:', ['southwest', 'southeast','northwest','northeast'])

# Create a DataFrame for the user input
user_input = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region],
    })

# Categorical Data Encoding
user_input['sex'] = sex_encoder.transform(user_input['sex'])
user_input['smoker'] = smoker_encoder.transform(user_input['smoker'])
user_input['region'] = region_encoder.transform(user_input['region'])

# Predicting
prediction = model.predict(user_input)

# Display Result
st.subheader('Prediction Result:')
st.write('Insurance expense:', prediction[0])
