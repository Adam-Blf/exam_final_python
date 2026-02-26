import sys
import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

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
# Titre        : Entraînement des modèles (Supervisés & Non Supervisés)
# Auteur       : Adam Beloucif et Emilien MORICE
# Projet       : Examen Final Python Data Science
# Date         : 2026-02-26
# Description  : Script principal pour mon projet de fin d'année. Ici on nettoie nos 
#                données et on entraîne quelques modèles vus en cours (Régression, 
#                Arbre, Random Forest).
# =============================================================================

# Forcer l'encodage de la console Windows en UTF-8 pour supporter les emojis et caractères spéciaux
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

# Création du répertoire d'output s'il n'existe pas
os.makedirs("output", exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# TRAVAIL 1 : CHARGEMENT ET NETTOYAGE DES DONNÉES
# ═════════════════════════════════════════════════════════════════════════════

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Charge les données depuis un CSV et nettoie les anomalies.
    POURQUOI : Comme vu en TD, il y a souvent des valeurs bizarres dans les vrais datasets 
    (loyers à 1$). Les retirer aide nos modèles à mieux apprendre la réalité du marché.
    """
    logger.info("📦 Je commence par charger le fichier CSV de base...")
    try:
        df = pd.read_csv(filepath, sep=';', encoding='cp1252')
        logger.info(f"✅ Données chargées avec succès : {df.shape[0]} lignes.")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement : {e}")
        sys.exit(1)

    # 1. Sélection des features pertinentes (Feature Selection)
    # POURQUOI : On va faire simple et utiliser les variables numériques évidentes 
    # (bathrooms, bedrooms, surface et latitude/longitude) pour pas s'embrouiller avec le texte.
    features = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude', 'price']
    df = df[features]
    
    # 2. Nettoyage des valeurs manquantes
    # POURQUOI : Les algos de sklearn de base gèrent pas les valeurs nulles (les NaN). 
    # Vu qu'on a beaucoup de données, jeter quelques lignes incomplètes c'est plus sûr qu'essayer de les deviner.
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"🔍 Nettoyage NaN : {initial_len - len(df)} lignes supprimées.")

    # 3. Traitement des Outliers (Valeurs Aberrantes)
    # POURQUOI : Un loyer à 1$ ça bousille completement nos moyennes. 
    # On garde juste les apparts normaux entre 300$ et 10,000$, et plus de 200 sqft.
    df = df[(df['price'] > 300) & (df['price'] < 10000)]
    df = df[(df['square_feet'] > 200) & (df['square_feet'] < 10000)]
    
    logger.info(f"📊 Données finalisées après filtrage des Outliers : {len(df)} annonces conservées.")
    
    return df

# ═════════════════════════════════════════════════════════════════════════════
# TRAVAIL 2 : PRE-PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_data(df: pd.DataFrame):
    """
    Sépare les données entre X (les features) et y (ce qu'on veut prédire, le prix).
    POURQUOI : Les mètres carrés (milliers) et les latitudes (dizaines) ont des échelles très différentes !
    Le StandardScaler met tout à plat pour que les modèles apprennent proportionnellement sans favoriser la surface à tort.
    """
    logger.info("⚙️ Démarrage du Pre-Processing : on sépare train/test et on applique un petit StandardScaler...")
    
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Validation du split classique 80/20 pour garder assez de données d'apprentissage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Sauvegarde du scaler pour pouvoir transformer les inputs de l'API plus tard
    joblib.dump(scaler, 'scaler.pkl')
    logger.info("✅ Scaler sauvegardé sous 'scaler.pkl'")
    
    # Création d'un dataset "Features" global pour l'appentissage Non-Supervisé
    X_scaled_full = scaler.fit_transform(X)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X, X_scaled_full

# ═════════════════════════════════════════════════════════════════════════════
# TRAVAIL 3 : MODÈLES SUPERVISÉS (PRÉDICTION DU PRIX)
# ═════════════════════════════════════════════════════════════════════════════

def train_supervised_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Entraîne trois modèles supervisés et on garde le gagnant.
    POURQUOI : C'est la démarche super classique du cours : on compare d'abord un modèle basique 
    (Régression Linéaire) face à des modèles plus lourds comme Random Forest pour voir l'impact de la complexité.
    """
    logger.info("🚀 C'est parti pour l'entraînement de nos modèles de Machine Learning...")
    
    models = {
        "Régression Linéaire": LinearRegression(),
        "Arbre de Décision": DecisionTreeRegressor(random_state=42),
        "Forêt Aléatoire (Random Forest)": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_r2 = -np.inf
    best_model_name = ""
    
    for name, model in models.items():
        logger.info(f"⏳ Entraînement du modèle : {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # MÉTRIQUES
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        logger.info(f"   📈 {name} -> RMSE: {rmse:.2f} | R²: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name
            
    logger.info(f"🏆 Le meilleur modèle supervisé est '{best_model_name}' avec R²={best_r2:.4f}")
    
    # Feature Importances (exclusif aux modèles type Random Forest/Decision Tree)
    # POURQUOI : Mon prof m'a dit qu'il faut toujours pouvoir expliquer son modèle. 
    # Afficher l'importance des variables montre ce qui compte vraiment pour bien prédire un loyer de façon concrète.
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
        plt.title('Importance des Features (Analyse Business)', fontsize=14)
        plt.xlabel('Importance Relative')
        plt.ylabel('Variables explicatives')
        plt.tight_layout()
        plt.savefig('output/feature_importance.png')
        plt.close()
        logger.info("📸 Graphique 'feature_importance.png' généré dans output/.")
        
    # Sauvegarde du meilleur modèle
    joblib.dump(best_model, 'model.pkl')
    logger.info("📁 Modèle final sauvegardé sous 'model.pkl'")
    
    return best_model

# ═════════════════════════════════════════════════════════════════════════════
# TRAVAIL 4 : MODÈLE NON SUPERVISÉ (CLUSTERING GÉOGRAPHIQUE / IMMOBILIER)
# ═════════════════════════════════════════════════════════════════════════════

def train_unsupervised_model(X_scaled_full, X_original):
    """
    Petit modèle K-Means pour voir des tendances dans le marché.
    POURQUOI : Histoire de rajouter une touche non-supervisée au projet ! Ça nous permet de 
    segmenter le marché de façon automatique, peut-être qu'il regroupe de lui-même les petits studios vs les villas.
    """
    logger.info("🧠 Et pour finir, on lance l'entraînement d'un petit clustering K-Means non supervisé...")
    
    # Hypothèse métier : 4 types d'appartements principaux
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled_full)
    
    X_original['Cluster'] = clusters
    
    # Analyse visuelle des clusters selon la Surface et le nombre de Chambres
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=X_original, x='square_feet', y='bedrooms', hue='Cluster', palette='Set1', alpha=0.6)
    plt.title('Clustering K-Means : Segmentation des biens', fontsize=14)
    plt.xlabel('Surface (sq ft)')
    plt.ylabel('Nombre de Chambres')
    plt.tight_layout()
    plt.savefig('output/clustering_analysis.png')
    plt.close()
    logger.info("📸 Graphique 'clustering_analysis.png' généré dans output/.")

# ═════════════════════════════════════════════════════════════════════════════
# EXÉCUTION PRINCIPALE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    filepath = "apartments_for_rent_classified_10K.csv"
    
    # 1. Chargement et nettoyage
    df_clean = load_and_clean_data(filepath)
    
    # 2. Pré-traitement
    X_train, X_test, y_train, y_test, X, X_scaled_full = preprocess_data(df_clean)
    
    # 3. Modélisation Supervisée
    best_model = train_supervised_models(X_train, X_test, y_train, y_test, X.columns)
    
    # 4. Modélisation Non Supervisée
    train_unsupervised_model(X_scaled_full, X)
    
    logger.info("🎉 Fin de l'entraînement avec succès. Tous les livrables sont générés !")
