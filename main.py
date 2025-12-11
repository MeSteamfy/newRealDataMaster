# ====================================================================
# main.py - API FastAPI pour la prédiction (Q9)
# ====================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
from typing import List

# --- Configuration ---
# Le pipeline Q7/Q8
PIPELINE_FILENAME = "credit_scoring_pipeline.pkl" 
NB_EXPECTED_FEATURES = 13  # Nombre de colonnes d'entrée du pipeline

app = FastAPI(title="Credit Scoring API")

class CreditApplication(BaseModel):
    Seniority: float
    Home: float
    Time: float
    Age: float
    Marital: float
    Records: float
    Job: float
    Expenses: float
    Income: float
    Assets: float
    Debt: float
    Amount: float
    Price: float

    class Config:
        json_schema_extra = {
            "example": {
                "Seniority": 9.0,
                "Home": 1.0,
                "Time": 60.0,
                "Age": 30.0,
                "Marital": 0.0,
                "Records": 1.0,
                "Job": 1.0,
                "Expenses": 73.0,
                "Income": 129.0,
                "Assets": 0.0,
                "Debt": 0.0,
                "Amount": 800.0,
                "Price": 846.0
            }
        }

# --- Chargement du Pipeline ---
@app.on_event("startup")
def load_pipeline():
    """Charge le pipeline ML au démarrage de l'API."""
    if not os.path.exists(PIPELINE_FILENAME):
        # Le pipeline est créé à la fin de la Q8 (orchestration)
        raise HTTPException(status_code=500, detail=f"Pipeline file not found: {PIPELINE_FILENAME}")

    try:
        with open(PIPELINE_FILENAME, 'rb') as file:
            app.state.pipeline = pickle.load(file)
        print("Pipeline chargé avec succès.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du pipeline: {e}")

# --- Endpoint de Prédiction ---
@app.post("/predict/")
def predict_status(application: CreditApplication):
    # 1. On récupère le pipeline stocké dans l'état de l'application
    pipeline = app.state.pipeline

    # 2. On transforme l'objet reçu en une liste ordonnée
    features_list = [
        application.Seniority,
        application.Home,
        application.Time,
        application.Age,
        application.Marital,
        application.Records,
        application.Job,
        application.Expenses,
        application.Income,
        application.Assets,
        application.Debt,
        application.Amount,
        application.Price
    ]

    # 3. Prédiction
    try:
        # Le modèle attend une liste de listes (2D array)
        prediction = pipeline.predict([features_list])
        probabilities = pipeline.predict_proba([features_list])

        return {
            "prediction": int(prediction[0]),
            "status_message": "Crédit Accordé" if prediction[0] == 1 else "Crédit Refusé",
            "probability_refuse": float(probabilities[0][0]),
            "probability_accord": float(probabilities[0][1])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {str(e)}")

