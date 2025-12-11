# Fichier principal

Le fichier principal est [celui ci](./newTPAlgoDataMaster.ipynb)

# Atelier Apprentissage Supervisé - Credit Scoring

Ce projet implémente un pipeline complet de Machine Learning pour le scoring de crédit, allant de l'analyse exploratoire à la mise en production via une API FastAPI.

## 📋 Prérequis

- **Python** (version 3.9 ou supérieure)
- **Poetry** (Gestionnaire de dépendances)

Si Poetry n'est pas installé sur votre machine :

```bash
# Windows : 
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
$Env:Path += ";$Env:APPDATA\Python\Scripts"
poetry --version

# Mac
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
source ~/.zshrc
poetry --version

# Linux
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
poetry --version


## Clonez le dépôt et accédez au dossier :

git clone git@github.com:MeSteamfy/newRealDataMaster.git
cd newRealDataMaster
```

## Installez l'environnement virtuel et les dépendances :

```bash
poetry lock
poetry install
```

## Lancez Jupyter via Poetry
```bash
poetry run jupyter notebook
```

Ouvrir et exécuter le notebook :
    Dans l'interface web qui s'ouvre, cliquez sur le fichier newTPAlgoDataMaster.ipynb
    Dans le menu du notebook, sélectionnez Kernel > Restart & Run All.
    Cette étape va sauvegarder le fichier credit_scoring_pipeline.pkl sur votre disque.

## Lancement de l'API

### Démarrez le serveur Uvicorn
```bash
poetry run uvicorn main:app --reload
```

L'API est maintenant accessible. Vous pouvez tester les endpoints via la documentation interactive :
    URL : http://127.0.0.1:8000/docs

**Exemple de requête**
```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict/](http://127.0.0.1:8000/predict/)' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": [9.0, 1.0, 60.0, 30.0, 0.0, 1.0, 1.0, 73.0, 129.0, 0.0, 0.0, 800.0, 846.0]
}'
```







