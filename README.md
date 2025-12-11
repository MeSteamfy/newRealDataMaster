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

Ouvrir et exécuter le notebook : <br>
    Dans l'interface web qui s'ouvre, cliquez sur le fichier **newTPAlgoDataMaster.ipynb**<br>
    Dans le menu du notebook, sélectionnez Kernel > Restart & Run All.<br>
    Cette étape va sauvegarder le fichier **credit_scoring_pipeline.pkl** sur votre disque.<br>

## Lancement de l'API

### Démarrez le serveur Uvicorn
```bash
poetry run uvicorn main:app --reload
```

L'API est maintenant accessible. Vous pouvez tester les endpoints via la documentation interactive :
    URL : http://127.0.0.1:8000/docs

**📖 Utilisation de l'API**

L'endpoint principal est /predict/. Il attend une requête POST contenant un vecteur de 13 caractéristiques (features) décrivant le client.  

**Format de la requête**
Le corps de la requête (Body) doit être au format JSON :
```json
{
  "features": [valeur1, valeur2, ..., valeur13]
}
```

**À quoi correspondent les 13 valeurs ?**

L'ordre des données est crucial pour la prédiction. Voici la correspondance des index :
- 0 : [Nom Feature 1] (ex: Statut du compte)
- 1 : [Nom Feature 2] (ex: Durée du crédit en mois)
- 2 : [Nom Feature 3] (ex: Historique des crédits)
- ...
- 12 : [Nom Feature 13] (ex: Montant du crédit)

**Exemple de test**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": [9.0, 1.0, 60.0, 30.0, 0.0, 1.0, 1.0, 73.0, 129.0, 0.0, 0.0, 800.0, 846.0]
}'
```









