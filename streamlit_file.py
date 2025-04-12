#!/usr/bin/env python
# coding: utf-8

# In[14]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model, Scaler, and Feature Names
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")  # List of 65 expected features

# Streamlit UI
st.title("📊 Telecom Churn Prediction App")
st.sidebar.header("Enter Customer Details")

# Numerical Inputs
account_length = st.sidebar.number_input("Account Length (Days)", min_value=0, max_value=500, value=100)
voice_plan = st.sidebar.selectbox("Voice Plan", ["No", "Yes"])
voice_messages = st.sidebar.number_input("Voice Messages", min_value=0, max_value=100, value=10)
intl_plan = st.sidebar.selectbox("International Plan", ["No", "Yes"])
intl_mins = st.sidebar.number_input("International Minutes", min_value=0.0, max_value=50.0, value=10.0)
intl_calls = st.sidebar.number_input("International Calls", min_value=0, max_value=20, value=5)
day_mins = st.sidebar.number_input("Day Minutes", min_value=0.0, max_value=500.0, value=200.0)
day_calls = st.sidebar.number_input("Day Calls", min_value=0, max_value=200, value=100)
eve_mins = st.sidebar.number_input("Evening Minutes", min_value=0.0, max_value=500.0, value=180.0)
eve_calls = st.sidebar.number_input("Evening Calls", min_value=0, max_value=200, value=100)
night_mins = st.sidebar.number_input("Night Minutes", min_value=0.0, max_value=500.0, value=200.0)
night_calls = st.sidebar.number_input("Night Calls", min_value=0, max_value=200, value=100)
customer_calls = st.sidebar.number_input("Customer Service Calls", min_value=0, max_value=20, value=2)

# Convert categorical inputs
voice_plan = 1 if voice_plan == "Yes" else 0
intl_plan = 1 if intl_plan == "Yes" else 0

# State Selection (One-Hot Encoding)
states = ['AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA',
          'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK',
          'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

selected_state = st.sidebar.selectbox("Select State", states)

# One-hot encode the selected state
state_features = {f"state_{state}": 1 if state == selected_state else 0 for state in states}

# One-hot encoding for Area Code
area_codes = ["area.code_area_code_415", "area.code_area_code_510"]
selected_area_code = st.sidebar.selectbox("Select Area Code", ["415", "510"])
area_code_features = {col: 1 if col == f"area.code_area_code_{selected_area_code}" else 0 for col in area_codes}

# Create input dictionary
input_dict = {
    "account.length": account_length,
    "voice.plan": voice_plan,
    "voice.messages": voice_messages,
    "intl.plan": intl_plan,
    "intl.mins": intl_mins,
    "intl.calls": intl_calls,
    "day.mins": day_mins,
    "day.calls": day_calls,
    "eve.mins": eve_mins,
    "eve.calls": eve_calls,
    "night.mins": night_mins,
    "night.calls": night_calls,
    "customer.calls": customer_calls,
}

# Combine all features
input_dict.update(state_features)
input_dict.update(area_code_features)

# Convert to DataFrame and Reindex
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=feature_names, fill_value=0)  # Ensures correct feature order

# Scale numerical features
numerical_columns = ["account.length", "voice.messages", "intl.mins", "intl.calls",
                     "day.mins", "day.calls", "eve.mins", "eve.calls",
                     "night.mins", "night.calls", "customer.calls"]

input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

# Convert to NumPy array for prediction
input_data = input_df.values

# Prediction Button
if st.sidebar.button("Predict Churn"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]
    if prediction[0] == 1:
        st.error(f"⚠️ The customer is likely to churn.\n\n**Churn Probability: {probability[0]:.2%}**")
    else:
        st.success(f"✅ The customer is unlikely to churn.\n\n**Churn Probability: {probability[0]:.2%}**")


# In[ ]:




