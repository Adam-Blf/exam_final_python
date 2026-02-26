# Utilisation d'une image Python optimisée et légère
FROM python:3.10-slim

# Définition du répertoire de travail dans le container
WORKDIR /app

# Copie des fichiers de configuration
COPY requirements.txt .

# Installation des dépendances avec nettoyage du cache pour réduire l'image
RUN pip install --no-cache-dir -r requirements.txt

# Copie intégrale du projet métier
COPY . .

# Exposition du port sur lequel FastAPI va écouter
EXPOSE 8000

# Commande d'exécution du serveur Uvicorn avec host ouvert pour Docker
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
