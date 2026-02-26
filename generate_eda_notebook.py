import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# Analyse Exploratoire des Données (EDA) - Prix des Loyers aux USA

**Projet**: Examen Final Python - Data Science
**Auteurs**: Adam Beloucif et Emilien MORICE

## Problématique Business
Le but de ce projet est d'analyser les annonces immobilières afin de comprendre les facteurs qui influencent le prix des loyers aux États-Unis, puis de concevoir un modèle de Machine Learning capable d'estimer ce prix pour de nouveaux biens.

### Objectifs de cette étape :
1. Explorer le jeu de données `apartments_for_rent_classified_10K.csv`.
2. Gérer les valeurs manquantes et aberrantes.
3. Étudier la distribution de la variable cible (`price`).
4. Identifier les corrélations entre les variables explicatives et le prix.
"""

code_import = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Configuration visuelle globale
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
"""

text_load = """\
## 1. Chargement et Aperçu des données
"""

code_load = """\
# Lecture du dataset
# NB: Le separateur est un point-virgule et l'encodage est cp1252 d'après notre analyse préliminaire.
df = pd.read_csv('apartments_for_rent_classified_10K.csv', sep=';', encoding='cp1252')

print(f"Dimensions du dataset : {df.shape}")
display(df.head())
"""

text_structure = """\
## 2. Structure et Statistiques Descriptives
"""

code_structure = """\
# Informations générales
display(df.info())

# Statistiques descriptives
display(df.describe())
"""

text_missing = """\
## 3. Analyse des Valeurs Manquantes
"""

code_missing = """\
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({'Total Manquants': missing_values, 'Pourcentage (%)': missing_percent})
display(missing_data[missing_data['Total Manquants'] > 0].sort_values(by='Pourcentage (%)', ascending=False))

# Visualisation des valeurs manquantes
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Carte de chaleur des valeurs manquantes', fontsize=16)
plt.show()
"""

text_target = """\
## 4. Analyse de la Variable Cible (Price)
"""

code_target = """\
plt.figure(figsize=(14, 5))

# Histogramme du prix
plt.subplot(1, 2, 1)
sns.histplot(df['price'], bins=50, kde=True, color='blue')
plt.title('Distribution des Loyer ($)', fontsize=14)
plt.xlabel('Prix')
plt.ylabel('Fréquence')

# Boîte à moustaches (Boxplot) pour identifier les outliers
plt.subplot(1, 2, 2)
sns.boxplot(x=df['price'], color='lightblue')
plt.title('Boxplot des Loyer ($) - Outliers', fontsize=14)
plt.xlabel('Prix')

plt.tight_layout()
plt.show()

print("Skewness (Asymétrie) :", df['price'].skew())
print("Kurtosis (Aplatissement) :", df['price'].kurt())
"""

text_bivariate = """\
## 5. Analyse Bivariée & Corrélations
"""

code_bivariate = """\
# Filtrage des variables numériques
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

plt.figure(figsize=(12, 8))
correlation_matrix = df[numeric_cols].corr()

# Masque pour la matrice triangulaire supérieure
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
plt.title('Matrice de Corrélation', fontsize=16)
plt.show()
"""

code_scatter = """\
# Relation entre la surface (square_feet) et le prix
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='square_feet', y='price', alpha=0.5, color='darkorange')
plt.title('Relation entre la surface (sq ft) et le Loyer ($)', fontsize=14)
plt.xlabel('Surface (Square Feet)')
plt.ylabel('Prix du Loyer ($)')
plt.show()
"""

text_conclusion = """\
## 6. Conclusions pour la Modélisation

1. **Valeurs manquantes** : Les colonnes avec un très fort taux de valeurs manquantes (ex: `address`, `amenities`) devront être ignorées ou traitées spécifiquement.
2. **Outliers** : La distribution du `price` est asymétrique avec des valeurs extrêmes à droite. Un filtrage (ex: isoler les loyers entre 200$ et 10,000$) ou une transformation logarithmique est nécessaire. De même pour `square_feet`.
3. **Features pertinentes** : Les variables `square_feet`, `bedrooms`, `bathrooms`, `latitude` et `longitude` montrent le plus fort potentiel explicatif ou logique métier. Les variables textuelles (`title`, `body`) ne seront pas utilisées pour le modèle de base, bien qu'elles pourraient être utiles avec du NLP.
4. **Catégorielles** : `pets_allowed` ou `state` peuvent être encodées.
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_import),
    nbf.v4.new_markdown_cell(text_load),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_markdown_cell(text_structure),
    nbf.v4.new_code_cell(code_structure),
    nbf.v4.new_markdown_cell(text_missing),
    nbf.v4.new_code_cell(code_missing),
    nbf.v4.new_markdown_cell(text_target),
    nbf.v4.new_code_cell(code_target),
    nbf.v4.new_markdown_cell(text_bivariate),
    nbf.v4.new_code_cell(code_bivariate),
    nbf.v4.new_code_cell(code_scatter),
    nbf.v4.new_markdown_cell(text_conclusion),
]

with open('eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("eda.ipynb généré avec succès.")
