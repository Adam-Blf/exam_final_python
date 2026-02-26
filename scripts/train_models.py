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
# Description  : Nettoyage, Pre-processing, Entraînement de 3 modèles supervisés et 
#                1 cluster non supervisé, évaluation détaillée, et exports des poids.
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
    Charge les données depuis un CSV et applique les règles de nettoyage.
    POURQUOI : Le jeu de données initial contient beaucoup de valeurs aberrantes 
    (prix irréalistes) et des valeurs manquantes qui fausseraient l'apprentissage 
    de nos modèles immobiliers.
    """
    logger.info("📦 Démarrage du chargement des données brutes...")
    try:
        df = pd.read_csv(filepath, sep=';', encoding='cp1252')
        logger.info(f"✅ Données chargées avec succès : {df.shape[0]} lignes.")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement : {e}")
        sys.exit(1)

    # 1. Sélection des features pertinentes (Feature Selection métier)
    # POURQUOI : Simplifier le modèle en écartant les colonnes non structurées (textes)
    # et conserver celles directement liées au prix du logement selon l'intuition métier.
    features = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude', 'price']
    df = df[features]
    
    # 2. Nettoyage des valeurs manquantes
    # POURQUOI : Les algorithmes comme la régression linéaire ou Random Forest 
    # ne supportent pas les NaN. La suppression est la méthode la plus sûre si le volume est faible.
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"🔍 Nettoyage NaN : {initial_len - len(df)} lignes supprimées.")

    # 3. Traitement des Outliers (Valeurs Aberrantes)
    # POURQUOI : Un loyer à 1$ ou 1M$ biaise la moyenne et les prédictions. 
    # On isole les biens "standards" (Ex: Loyer entre 300$ et 10,000$, surface > 200 sqft).
    df = df[(df['price'] > 300) & (df['price'] < 10000)]
    df = df[(df['square_feet'] > 200) & (df['square_feet'] < 10000)]
    
    logger.info(f"📊 Données finalisées après filtrage des Outliers : {len(df)} annonces conservées.")
    
    return df

# ═════════════════════════════════════════════════════════════════════════════
# TRAVAIL 2 : PRE-PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_data(df: pd.DataFrame):
    """
    Sépare les données en variables explicatives (X) et cible (y), 
    puis applique une normalisation.
    POURQUOI : Les différentes variables n'ont pas la même échelle (prix vs latitude).
    La normalisation (StandardScaler) aide les modèles basés sur les distances et optimise 
    la convergence des algorithmes.
    """
    logger.info("⚙️ Démarrage du Pre-Processing (Split & Scaling)...")
    
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
    Entraîne trois modèles supervisés différents et sélectionne le meilleur.
    POURQUOI : Permet de comparer un modèle linéaire simple avec des modèles non linéaires 
    arborescents qui captent mieux les complexités métier.
    """
    logger.info("🚀 Démarrage de l'entraînement des modèles supervisés...")
    
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
    
    # Feature Importances (exclusif aux modèles ensemblistes comme Random Forest)
    # POURQUOI : Pour le reporting business, il faut expliquer quelles sont les variables 
    # qui font grimper le prix du loyer.
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
    Entraîne un modèle K-Means pour grouper les annonces.
    POURQUOI : Découvrir des 'segments' de biens immobiliers sans indications préalables.
    Par exemple, segmenter en "Biens de luxe", "Biens familiaux", etc.
    """
    logger.info("🧠 Démarrage de l'entraînement du modèle Non-Supervisé (K-Means)...")
    
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
