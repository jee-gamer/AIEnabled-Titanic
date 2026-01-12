import streamlit as st
import joblib
import pandas as pd

from app_api import Passenger

st.set_page_config(page_title="Titanic Survival Demo", page_icon="🚢", layout="centered")
st.title("🚢 Titanic Survival Predictor")

@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")

model = load_model()

st.subheader("Passengar Features")
c1, c2, c3 = st.columns(3)

with c1:
    PassengerName = st.text_input("Passenger Name", value="Kingston Taylor")
    Pclass = st.selectbox("Pclass", [1,2,3], index=1)
    Sex = st.selectbox("Sex", ["male", "female"], index=1)
    HasPrefix = st.selectbox("HasPrefix", [0,1], index=1)

with c2:
    Age = st.number_input("Age", min_value=0.0, max_value=100.0, value=28.0, step=1.0)
    SibSp= st.number_input("SibSp", min_value=0, max_value=10, value=0, step=1)
    Parch = st.number_input("Parch", min_value=0, max_value=10, value=0, step=1)
    TicketIsLine = st.selectbox("TicketIsLine", [0,1], index=0)

with c3:
    TicketNumber = st.number_input("TicketNumber", min_value=0, value=17599, step=1)
    TicketLength = st.number_input("TicketLength", min_value=0, value=5, step=1)
    Fare = st.number_input("Fare", min_value=0.0, value=26.0, step=0.1)
    Embarked = st.selectbox("Embarked", ["S","C","Q"], index=0)

if st.button("Predict"):
    cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "HasPrefix", "TicketNumber", "TicketIsLine", "TicketLength", "Fare", "Embarked"]
    X = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch,
                       HasPrefix, TicketNumber, TicketIsLine, TicketLength, Fare, Embarked]], columns=cols)
    prob = model.predict_proba(X)[0,1]
    pred = int(prob >= 0.5)

    st.markdown("----")
    st.subheader("Result: %s"% PassengerName)
    st.write(f"**Prediction:** {'✅ Survive' if pred==1 else '❌ Not survive'}")
    st.write(f"**Probability of survival:** {prob:.2%}")
    st.caption("Educational demo. Not for real-world decision-making.")