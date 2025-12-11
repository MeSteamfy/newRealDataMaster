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

# --- Modèle de données d'entrée ---
class FeatureInput(BaseModel):
    features: List[float]
    
    class Config:
        schema_extra = {
            "example": {
                "features": [9.0, 1.0, 60.0, 30.0, 0.0, 1.0, 1.0, 73.0, 129.0, 0.0, 0.0, 800.0, 846.0] 
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
def predict_status(input_data: FeatureInput):
    """Renvoie la prédiction de statut (0 ou 1) et les probabilités associées."""
    
    if len(input_data.features) != NB_EXPECTED_FEATURES:
        raise HTTPException(status_code=400, detail=f"Expected {NB_EXPECTED_FEATURES} features, but received {len(input_data.features)}")

    # Convertir l'entrée en array numpy 2D
    data_array = np.array(input_data.features).reshape(1, -1)
    
    pipeline = app.state.pipeline
    
    try:
        prediction = pipeline.predict(data_array)[0]
        # Vérifier si le modèle supporte predict_proba (important pour certains modèles comme le CART)
        if hasattr(pipeline.named_steps['final_classifier'], 'predict_proba'):
            probabilities = pipeline.predict_proba(data_array)[0]
        else:
            # Si le modèle ne supporte pas, on renvoie une probabilité par défaut ou une estimation
            probabilities = [0.5, 0.5] 
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {e}")

    result = {
        "prediction": int(prediction),
        "status_message": "Crédit Accordé (1)" if prediction == 1.0 else "Crédit Refusé (0)",
        "probability_refuse": float(probabilities[0]),
        "probability_accord": float(probabilities[1])
    }
    
    
    return result