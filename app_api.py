from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Titanic Survival API", version="1.0")

model = joblib.load('titanic_model.pkl')

class Passenger(BaseModel):
    Pclass: int = Field(ge=1, le=3)
    Sex: str
    Age: float | None = None
    SibSp: int
    Parch: int
    HasPrefix: int #0/1
    TicketNumber: int
    TicketLength: int #0/1
    TicketIsLine: int
    Fare: float | None = None
    Embarked: str


@app.post("/predict")
def predict_survival(p: Passenger):
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'HasPrefix',
            'TicketNumber', 'TicketLength', 'TicketIsLine', 'Fare', 'Embarked']
    X = pd.DataFrame([[getattr(p, col) for col in cols]], columns=cols)
    prob = model.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)
    return {"prediction": pred, 
            "prob_survive": round(prob, 4)}