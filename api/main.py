import sys
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =============================================================================
# ASCII HEADER - STYLE LIVRAISON PRO
# =============================================================================
#  _____                 _         ______ _             _   _____       _   _                 
# |  ___|               (_)        |  ___(_)           | | |  __ \     | | | |                
# | |__ __  ____ _ _ __  _ _ __    | |_   _ _ __   __ _| | | |  \/_   _| |_| |__  _   _ _ __  
# |  __|\ \/ / _` | '_ \| | '_ \   |  _| | | '_ \ / _` | | | | __| | | | __| '_ \| | | | '_ \ 
# | |___ >  < (_| | | | | | | | |  | |   | | | | | (_| | | | |_\ \ |_| | |_| | | | |_| | |_) |
# \____//_/\_\__,_|_| |_|_|_| |_|  \_|   |_|_| |_|\__,_|_|  \____/\__,_|\__|_| |_|\__,_| .__/ 
#                                                                                      | |    
#                                                                                      |_|    
# 
# Titre        : Serveur FastAPI pour la Prédiction des Loyers
# Auteur       : Adam Beloucif et Emilien MORICE
# Projet       : Examen Final Python Data Science
# Date         : 2026-02-26
# Description  : Exposition du modèle Random Forest via un endpoint /predict.
# =============================================================================

# Configuration Console UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# -----------------------------------------------------------------------------
# Configuration du Logging Professionnel
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Chargement du modèle et du scaler (POURQUOI : Éviter de recharger à chaque requête)
try:
    logger.info("📦 Chargement du modèle Random Forest (model.pkl)...")
    model = joblib.load("model.pkl")
    logger.info("📦 Chargement du Scaler (scaler.pkl)...")
    scaler = joblib.load("scaler.pkl")
    logger.info("✅ Modèles chargés en mémoire avec succès.")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement des modèles : {e}")
    sys.exit(1)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Rent Prediction API",
    description="API permettant d'estimer le prix d'un loyer aux États-Unis.",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# Modèles de Validation (Pydantic)
# -----------------------------------------------------------------------------
# POURQUOI : Assurer la validation stricte des inputs de l'utilisateur. 
# Si une variable est manquante ou aberrante, FastAPI renverra une erreur HTTP 422 claire.
class ApartmentFeatures(BaseModel):
    bathrooms: float = Field(..., description="Nombre de salles de bain", example=1.0)
    bedrooms: float = Field(..., description="Nombre de chambres", example=2.0)
    square_feet: float = Field(..., description="Surface en pieds carrés", example=1050.5)
    latitude: float = Field(..., description="Latitude géographique", example=38.905)
    longitude: float = Field(..., description="Longitude géographique", example=-76.986)

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health", summary="Vérifier la santé de l'API")
def health_check():
    """
    Retourne le statut de l'API. Très utile pour les Load Balancers ou Kubernetes.
    """
    return {"status": "ok", "message": "API opérationnelle \U0001f680"}


@app.post("/predict", summary="Estimer le loyer d'un appartement")
def predict_rent(payload: ApartmentFeatures):
    """
    Reçoit les caractéristiques d'un appartement, les met à l'échelle via le scaler,
    puis interroge le modèle de Machine Learning pour une prédiction de loyer.
    """
    logger.info(f"🔍 Requête de prédiction reçue : {payload.model_dump()}")
    
    try:
        # 1. Mise en forme de la donnée comme attendu par sklearn
        input_data = pd.DataFrame([payload.model_dump()])
        
        # 2. Application du Scaling (Pre-Processing obligatoire)
        input_scaled = scaler.transform(input_data)
        
        # 3. Prédiction
        predicted_price = model.predict(input_scaled)[0]
        
        logger.info(f"✅ Prédiction réussie : {predicted_price:.2f} $")
        
        return {
            "prediction_usd": round(float(predicted_price), 2),
            "currency": "USD",
            "model_version": "RandomForest_1.0"
        }
    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur lors de la prédiction.")
