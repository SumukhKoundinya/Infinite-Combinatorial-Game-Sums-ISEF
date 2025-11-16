import streamlit as st
import numpy as np
import pickle
import pandas as pd

model = pickle.load(open("finite_model.pkl", "rb"))
scaler = pickle.load(open("finite_scaler.pkl", "rb"))

st.title("Finite Incomplete Information VEX Game AI")

p1_action = st.selectbox("Player 1 Action", list(range(5)))
p2_action = st.selectbox("Player 2 Action", list(range(5)))

state_1 = st.checkbox("State 1 On/Off", value=False)
state_2 = st.checkbox("State 2 On/Off", value=False)
state_3 = st.checkbox("State 3 On/Off", value=False)

input_data = np.array([[p1_action, p2_action, int(state_1), int(state_2), int(state_3)]])

if st.button("Predict Winner"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Winner: Player {'1' if prediction == 1 else '2'}")

    st.bar_chart(pd.Series(model.feature_importances_, index=['p1_action', 'p2_action', 'state_1', 'state_2', 'state_3']))
